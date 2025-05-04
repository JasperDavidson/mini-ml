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
    float grad;
    std::function<std::vector<Tensor>(const Tensor&)> grad_fn;
    std::vector<Tensor> parents;
    int grad_count = 0;

    // Tensor constructors
    Tensor(std::vector<float> data, std::vector<int> shape, bool requires_grad = false);

    // Calculate the flattened index based on a multi-dimensional index and shape
    int flatten_index(const std::vector<int>& multi_idx) const;

    // Calculate the broadcasted shape for a tensor operation
    // A broadcasted shape is one that accommodates for all tensor dimensions each way
    static std::vector<int> broadcast_shapes(std::vector<int> shape_one, std::vector<int> shape_two);

private:
    // Compute the stride for each dimension --> how many elements to progress in 1D array to get to the next section of the same dimension
    std::vector<int> _compute_strides();
};