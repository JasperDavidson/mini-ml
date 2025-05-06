#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <functional>
#include <ranges>

class Tensor {
public:
    // Core Tensor components
    std::vector<float> data;
    std::vector<int> shape;
    bool requires_grad;
    std::vector<int> strides;

    // Autodiff internal requirements
    std::shared_ptr<Tensor> grad;
    std::function<std::vector<std::shared_ptr<Tensor>>(std::shared_ptr<Tensor>)> grad_fn;
    std::vector<std::shared_ptr<Tensor>> parents;
    int grad_count = 0;

    // Tensor constructors
    Tensor(std::vector<float> data, std::vector<int> shape, bool requires_grad = false);

    // Calculate the flattened index based on a multi-dimensional index and shape
    int flatten_index(const std::vector<int>& multi_idx) const;

    // Calculate the broadcasted shape for a tensor operation
    // A broadcasted shape is one that accommodates for all tensor dimensions each way
    static std::vector<int> broadcast_shapes(std::vector<int> shape_one, std::vector<int> shape_two);

    // Utility function to generate a tensor of a given size full of 1s
    Tensor ones_like();

private:
    // Compute the stride for each dimension --> how many elements to progress in 1D array to get to the next section of the same dimension
    std::vector<int> _compute_strides();
};

#endif