#include "tensor.h"

#include <vector>
#include <functional>
#include <stdexcept>

Tensor::Tensor(std::vector<float> data, std::vector<int> shape, bool requires_grad) : data(data), shape(shape), requires_grad(requires_grad) {
    strides = _compute_strides();
}

std::vector<int> Tensor::_compute_strides() {
    std::vector<int> strides{1};
    int stride = 1;

    for (auto it = shape.rbegin(); it != shape.rend(); ++it) {
        stride *= *it;
        strides.push_back(stride);
    }

    return strides;
}

int Tensor::flatten_index(const std::vector<int>& multi_idx) const {
    int flat_idx = 0;

    for (int i = 0; i < multi_idx.size(); ++i) {
        flat_idx += multi_idx[i] * strides[i];
    }

    return flat_idx;
}

std::vector<int> Tensor::broadcast_shapes(std::vector<int> shape_one, std::vector<int> shape_two) {
    std::vector<int> broadcast_shape;

    auto prepend_left = [](std::vector<int>& vec, int num) {
        for (int i = 0; i < num; ++i) {
            vec.insert(vec.begin(), 1);
        }
    };

    if (shape_one.size() < shape_two.size()) {
        int prepend = shape_two.size() - shape_one.size();
        prepend_left(shape_one, prepend);
    } else if (shape_two.size() < shape_one.size()) {
        int prepend = shape_one.size() - shape_two.size();
        prepend_left(shape_two, prepend);
    }

    auto dim1 = shape_one.begin(); auto dim2 = shape_two.begin();
    for(; dim1 != shape_one.end() && dim2 != shape_two.end(); ++dim1, ++dim2) {
        if (*dim1 == *dim2) {
            // Doesn't matter which one we push back
            broadcast_shape.push_back(*dim1);
        } else if (*dim1 == 1) {
            broadcast_shape.push_back(*dim2);
        } else if (*dim2 == 1) {
            broadcast_shape.push_back(*dim1);
        } else {
            throw std::invalid_argument("Invalid tensor arguments: Cannot complete broadcasting for operation");
        }
    }

    return broadcast_shape;
}

Tensor Tensor::ones_like() {
    std::vector<float> data{};

    for (int i = 0; i < strides[0] * shape[0]; ++i) {
        data.push_back(1);
    }

    return Tensor(data, shape, true);
}