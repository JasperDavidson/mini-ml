#ifndef AUTODIFF_H
#define AUTODIFF_H

#include "tensor.h"

#include <queue>

void backprop(std::vector<Tensor> roots);

#endif