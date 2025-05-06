#include "autodiff.h"
#include "ops.h"

#include <queue>

auto add = [](float a, float b) { return a + b; };

void backprop(std::vector<std::shared_ptr<Tensor>> roots) {
    std::queue<std::shared_ptr<Tensor>> tensor_queue;

    for (auto& root : roots) {
        root->grad = std::make_unique<Tensor>(root->ones_like());
        tensor_queue.push(root);
    }

    while (!tensor_queue.empty()) {
        std::shared_ptr<Tensor> current = tensor_queue.front();
        std::vector<std::shared_ptr<Tensor>> parent_grads = current->grad_fn(current->grad);
        tensor_queue.pop();

        for (int i = 0; i < current->parents.size(); ++i) {
            std::shared_ptr<Tensor> parent = current->parents[i];

            if (!parent->grad) {
                parent->grad = parent_grads[i];
            } else {
                *parent->grad = broadcast_op(*parent->grad, *parent_grads[i], add);
            }

            parent->grad_count--;

            if (parent->grad_count <= 0) {
                tensor_queue.push(parent);
            }
        }
    }
}