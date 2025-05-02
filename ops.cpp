#include "ops.h"

Tensor broadcast_op(Tensor t1, Tensor t2, float (*f)(float, float)) {
    std::vector<int> broadcast_shape = Tensor::broadcast_shapes(t1.shape, t2.shape);
}