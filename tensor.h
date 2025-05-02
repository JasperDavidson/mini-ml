#include <vector>
#include <functional>
#include <ranges>

class Tensor {
public:
    // Core Tensor components
    float data;
    std::vector<int> shape;
    bool requires_grad;
    std::vector<int> strides;

    Tensor(float data = 0.0f, std::vector<int> shape, bool requires_grad = false);

    int flatten_index(const std::vector<int>& multi_idx);
    static std::vector<int> broadcast_shapes(const std::vector<int>& shape_one, const std::vector<int>& shape_two);

private:
    // Autodiff internal requirements
    float grad;
    float prev_grad;
    std::function<std::tuple<float, float>(Tensor&, float)> grad_fn;
    std::vector<Tensor> parents;

    std::vector<int> _compute_strides();
};