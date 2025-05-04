#include "tensor.h"

/*
Basic Operations for Tensors
*/

Tensor broadcast_op(const Tensor& t1, const Tensor& t2, float (*f)(float, float));

// Computes the output tensor, creates the grad_fn, and adds the nodes' parents
Tensor add(Tensor t1, Tensor t2);
Tensor mul(Tensor t1, Tensor t2);
Tensor neg(Tensor t);
Tensor inv(Tensor t);