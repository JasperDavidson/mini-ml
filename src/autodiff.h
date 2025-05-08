#ifndef AUTODIFF_H
#define AUTODIFF_H

#include "tensor.h"

#include <queue>

void backward(std::vector<std::shared_ptr<Tensor>> roots);

#endif