#include "tensor.h"

/*
Basic Operations for Tensors
*/

Tensor broadcast_op(Tensor t1, Tensor t2, float (*f)(float, float));