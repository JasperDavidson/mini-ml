#include <numeric>
#include <algorithm>

#include "ops.h"
#include "tensor.h"

Tensor broadcast_op(const Tensor& t1, const Tensor& t2, float (*combine_fn)(float, float)) {
    // Create the broadcasted shape
    std::vector<int> broadcast_shape = Tensor::broadcast_shapes(t1.shape, t2.shape);

    // Compute the data with broadcasted shape
    std::vector<float> output_data;
    int total_size = std::accumulate(broadcast_shape.begin(), broadcast_shape.end(), 1, std::multiplies<float>());

    for (int i = 0; i < total_size; ++i) {
        int temp_idx = i;
        std::vector<int> multi_idx;

        // Get the multidimensional index for the one dimensional index
        for (int i = broadcast_shape.size() - 1; i >= 0; --i) {
            int dim = broadcast_shape[i];
            multi_idx.insert(multi_idx.begin(), temp_idx % dim);
            temp_idx = (int) (temp_idx / dim);
        }

        // Each tensor needs it's own flattened index to access for each index in the broadcasted space
        // If the shape is every <= 1 that means it effectively doesn't contain any information --> encode 0 for flatten
        std::vector<int> flat_idx_1;
        std::vector<int> flat_idx_2;

        for (int i = 0; i < multi_idx.size(); ++i) {
            if (t1.shape[i] > 1) {
                flat_idx_1.push_back(multi_idx[i]);
            } else {
                flat_idx_1.push_back(0);
            }

            if (t2.shape[i] > 1) {
                flat_idx_2.push_back(multi_idx[i]);
            } else {
                flat_idx_2.push_back(0);
            }
        }

        int flat_index_1 = t1.flatten_index(flat_idx_1);
        int flat_index_2 = t2.flatten_index(flat_idx_2);

        // Perform the elementwise operation
        output_data.push_back(combine_fn(t1.data[flat_index_1], t2.data[flat_index_2]));
    }

    return Tensor(output_data, broadcast_shape, true);
}

Tensor add(Tensor t1, Tensor t2) {
    Tensor output_t = broadcast_op(t1, t2, [](float a, float b) { return a + b; });

    auto grad_fn = [](const Tensor& upstream_grad) {
        return std::vector<Tensor>{upstream_grad, upstream_grad};
    };

    output_t.grad_fn = grad_fn;
    output_t.parents = {t1, t2};

    return output_t;
}

Tensor mul(Tensor t1, Tensor t2) {
    auto mul = [](float a, float b) { return a * b; };
    Tensor output_t = broadcast_op(t1, t2, mul);

    auto grad_fn = [&t1, &t2, &mul](const Tensor& upstream_grad) {
        return std::vector<Tensor>{
            broadcast_op(t2, upstream_grad, mul),
            broadcast_op(t1, upstream_grad, mul)
        };
    };

    output_t.grad_fn = grad_fn;
    output_t.parents = {t1, t2};

    return output_t;
}

Tensor neg(Tensor t) {
    auto neg = [](float a) { return -a; };

    Tensor output_t(t.data, t.shape, true);
    std::transform(output_t.data.begin(), output_t.data.end(), output_t.data.begin(), neg);

    auto grad_fn = [&neg](const Tensor& upstream_grad) {
        Tensor neg_upstream = upstream_grad;
        std::transform(neg_upstream.data.begin(), neg_upstream.data.end(), neg_upstream.data.begin(), neg);

        return std::vector<Tensor>{neg_upstream};
    };

    output_t.grad_fn = grad_fn;
    output_t.parents = std::vector<Tensor>{t};

    return output_t;
}

Tensor inv(Tensor t) {
    auto inv = [](float a) { return 1 / a; };

    Tensor output_t(t.data, t.shape, true);
    std::transform(output_t.data.begin(), output_t.data.end(), output_t.data.begin(), inv);

    auto grad_fn = [](const Tensor& upstream_grad) {
        auto inv_deriv = [](float a) { return -1 / (a * a); };

        Tensor inv_upstream = upstream_grad;
        std::transform(inv_upstream.data.begin(), inv_upstream.data.end(), inv_upstream.data.begin(), inv_deriv);
        
        return std::vector<Tensor>{inv_upstream};
    };

    output_t.grad_fn = grad_fn;
    output_t.parents = std::vector<Tensor>{t};

    return output_t;
}