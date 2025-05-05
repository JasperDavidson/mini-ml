#include "tensor.h"
#include "ops.h"

#include <iostream>

int main() {
    std::shared_ptr<Tensor> t1 = std::make_shared<Tensor>(Tensor({1.0}, {1}, true));
    std::shared_ptr<Tensor> t2 = std::make_shared<Tensor>(Tensor({1.0}, {1}, true));
    
    std::shared_ptr<Tensor> t3 = add(t1, t2);
    
    for (float num : t3->data) {
        std::cout << num << '\n';
    }
    
    return 0;
}