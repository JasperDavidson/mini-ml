#include "autodiff.h"

#include <queue>

Tensor backprop(std::vector<Tensor> roots) {
    std::queue<std::shared_ptr<Tensor>> tensor_queue;

    for (auto& root : roots) {
        root.grad = std::make_unique<Tensor>(root.ones_like());
        tensor_queue.push(std::make_shared<Tensor>(root));
    }

    while (!tensor_queue.empty()) {
        std::shared_ptr<Tensor> current = tensor_queue.front();
        tensor_queue.pop();

        for (std::shared_ptr<Tensor> parent : current->parents) {
            if (parent->grad_count == 0) {
                tensor_queue.push(parent);
            } else {
                if (!parent->grad) {

                } else {
                    
                }

                parent->grad_count--;
            }
        }
    }
}