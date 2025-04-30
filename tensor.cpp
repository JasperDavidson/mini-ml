#include "tensor.h"

#include <vector>
#include <functional>

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