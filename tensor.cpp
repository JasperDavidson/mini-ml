#include "tensor.h"

#include <vector>
#include <functional>
#include <stdexcept>

// Tensor constructor
Tensor::Tensor(float data, std::vector<int> shape, bool requires_grad) : data(data), shape(shape), requires_grad(requires_grad) {
    strides = _compute_strides();
}

// Compute the stride for each dimension --> how many elements to progress in 1D array to get to the next section of the same dimension
std::vector<int> Tensor::_compute_strides() {
    std::vector<int> strides{1};
    int stride = 1;

    for (auto it = shape.rbegin(); it != shape.rend(); ++it) {
        stride *= *it;
        strides.insert(strides.begin(), stride);
    }

    return strides;
}

// Calculate the flattened index based on a multi-dimensional index and shape
int Tensor::flatten_index(const std::vector<int>& multi_idx) {
    int flat_idx = -1;

    for (int i = 0; i < multi_idx.size(); ++i) {
        flat_idx += multi_idx[i] * strides[i];
    }

    return flat_idx;
}

std::vector<int> Tensor::broadcast_shapes(const std::vector<int>& shape_one, const std::vector<int>& shape_two) {
    std::vector<int> broadcast_shape;

    if (shape_one.size() < shape_two.size()) {
        for (int i = 0; i < shape_two.size() - shape_one.size(); ++i) {
            broadcast_shape.push_back(1);
        }
    } else if (shape_two.size() < shape_one.size()) {
        for (int i = 0; i < shape_one.size() - shape_two.size(); ++i) {
            broadcast_shape.push_back(1);
        }
    }

    auto dim1 = shape_one.begin(); auto dim2 = shape_two.begin();
    for (; dim1 != shape_one.end() && dim2 != shape_two.end(); ++dim1, ++dim2) {
        if (*dim1 == *dim2) {
            // Doesn't matter which one we push back
            broadcast_shape.push_back(*dim1);
        } else if (*dim1 == 1) {
            broadcast_shape.push_back(*dim2);
        } else if (*dim2 == 1) {
            broadcast_shape.push_back(*dim1);
        } else {
            throw std::invalid_argument("Invalid broadcast size: Operation on two tensors not possible");
        }
    }
}