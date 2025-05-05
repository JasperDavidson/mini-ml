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

std::shared_ptr<Tensor> add(std::shared_ptr<Tensor> t1, std::shared_ptr<Tensor> t2) {
    auto add_fn = [](float a, float b) { return a + b; };
    std::shared_ptr<Tensor> output_t = std::make_shared<Tensor>(broadcast_op(*t1, *t2, add_fn));

    auto grad_fn = [](Tensor& upstream_grad) {
        return std::vector<std::unique_ptr<Tensor>>{
            std::make_unique<Tensor>(upstream_grad),
            std::make_unique<Tensor>(upstream_grad)
        };
    };

    t1->grad_count++;    
    t2->grad_count++;
    output_t->grad_fn = grad_fn;
    output_t->parents = std::vector<std::shared_ptr<Tensor>>{t1, t2};

    return output_t;
}

std::shared_ptr<Tensor> mul(std::shared_ptr<Tensor> t1, std::shared_ptr<Tensor> t2) {
    auto mul_fn = [](float a, float b) { return a * b; };
    std::shared_ptr<Tensor> output_t = std::make_shared<Tensor>(broadcast_op(*t1, *t2, mul_fn));

    auto grad_fn = [t1, t2, mul_fn](Tensor& upstream_grad) {
        return std::vector<std::unique_ptr<Tensor>>{
            std::make_unique<Tensor>(broadcast_op(*t2, upstream_grad, mul_fn)),
            std::make_unique<Tensor>(broadcast_op(*t1, upstream_grad, mul_fn))
        };
    };

    t1->grad_count++;
    t2->grad_count++;
    output_t->grad_fn = grad_fn;
    output_t->parents = std::vector<std::shared_ptr<Tensor>>{t1, t2};

    return output_t;
}

std::shared_ptr<Tensor> neg(std::shared_ptr<Tensor> t) {
    auto neg_fn = [](float a) { return -a; };

    std::shared_ptr<Tensor> output_t = std::make_shared<Tensor>(*t);
    std::transform(output_t->data.begin(), output_t->data.end(), output_t->data.begin(), neg_fn);

    auto grad_fn = [neg_fn](Tensor& upstream_grad) {
        std::unique_ptr<Tensor> neg_upstream = std::make_unique<Tensor>(upstream_grad);
        std::transform(neg_upstream->data.begin(), neg_upstream->data.end(), neg_upstream->data.begin(), neg_fn);

        return std::vector<std::unique_ptr<Tensor>>{std::move(neg_upstream)};
    };

    t->grad_count++;
    output_t->grad_fn = grad_fn;
    output_t->parents = std::vector<std::shared_ptr<Tensor>>{t};

    return output_t;
}

std::shared_ptr<Tensor> inv(std::shared_ptr<Tensor> t) {
    auto inv_fn = [](float a) { return 1 / a; };

    std::shared_ptr<Tensor> output_t = std::make_shared<Tensor>(*t);
    std::transform(output_t->data.begin(), output_t->data.end(), output_t->data.begin(), inv_fn);

    auto grad_fn = [](Tensor& upstream_grad) {
        auto inv_deriv = [](float a) { return -1 / (a * a); };

        std::unique_ptr<Tensor> inv_upstream = std::make_unique<Tensor>(upstream_grad);
        std::transform(inv_upstream->data.begin(), inv_upstream->data.end(), inv_upstream->data.begin(), inv_deriv);
        
        return std::vector<std::unique_ptr<Tensor>>{std::move(inv_upstream)};
    };

    t->grad_count++;
    output_t->grad_fn = grad_fn;
    output_t->parents = std::vector<std::shared_ptr<Tensor>>{t};

    return output_t;
}