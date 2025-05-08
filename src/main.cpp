#include "tensor.h"
#include "ops.h"
#include "autodiff.h"

#include <iostream>
#include <cmath>

int main() {
    std::shared_ptr<Tensor> t1 = std::make_shared<Tensor>(Tensor({2.0, 2.0, 2.0, 2.0}, {2, 2}, true));
    std::shared_ptr<Tensor> t2 = std::make_shared<Tensor>(Tensor({1.0, 1.0}, {2, 1}, true));
    std::shared_ptr<Tensor> t3 = std::make_shared<Tensor>(Tensor({2.0, 2.0, 2.0, 2.0}, {2, 2}, true));
    
    std::shared_ptr<Tensor> add_op = add_ops(t1, t2);
    std::shared_ptr<Tensor> mul_op = mul_ops(add_op, t3);

    for (float num : mul_op->data) {
        std::cout << num << '\n';
    }

    backward({mul_op});

    std::cout << "t1 grad: \n";

    for (int num : (*t1->grad).data) {
        std::cout << num << '\n';
    }

    std::cout << '\n';

    std::cout << "t2 grad: \n";

    for (int num : (*t2->grad).data) {
        std::cout << num << '\n';
    }

    std::cout << '\n';

    std::cout << "t3 grad: \n";

    for (int num : (*t3->grad).data) {
        std::cout << num << '\n';
    }

    std::cout << '\n';

    return 0;
}