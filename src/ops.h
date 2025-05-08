#ifndef OPS_H
#define OPS_H

#include "tensor.h"

/*
Basic Operations for Tensors
*/

Tensor broadcast_op(const Tensor& t1, const Tensor& t2, float (*f)(float, float));

// Computes the output tensor, creates the grad_fn, and adds the nodes' parents
std::shared_ptr<Tensor> add_ops(std::shared_ptr<Tensor> t1, std::shared_ptr<Tensor> t2);
std::shared_ptr<Tensor> mul_ops(std::shared_ptr<Tensor> t1, std::shared_ptr<Tensor> t2);
std::shared_ptr<Tensor> neg_ops(std::shared_ptr<Tensor> t);
std::shared_ptr<Tensor> inv_ops(std::shared_ptr<Tensor> t);

#endif