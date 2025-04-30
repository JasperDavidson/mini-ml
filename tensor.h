#include <vector>
#include <functional>
#include <ranges>

class Tensor {
public:
    Tensor(float data = 0.0f, std::vector<int> shape, bool requires_grad = false);

    // Core Tensor components
    float data;
    std::vector<int> shape;
    bool requires_grad;
    std::vector<int> strides;

private:
    // Autodiff internal requirements
    float grad;
    float prev_grad;
    std::function<std::tuple<float, float>(Tensor&, float)> grad_fn;
    std::vector<Tensor> parents;

    std::vector<int> _compute_strides();
};