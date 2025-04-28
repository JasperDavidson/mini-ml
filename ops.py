import math

from tensor import Tensor
from typing import Callable

"""
Basic operations for Tensors
"""

# If two tensors are of different shapes, we need to mold them into the same shape in order to perform the operation --> "broadcasting"
def broadcast_op(t1: Tensor, t2: Tensor, combine_fn: Callable[[float], float]) -> Tensor:
    # Create the broadcasted shape
    broadcast_shape = Tensor.broadcast_shapes(t1.shape, t2.shape)

    # Compute the data w/broadcasted shape
    output_data = []

    total_size = math.prod(broadcast_shape)

    for idx in range(total_size):
        # Compute the multidimensional address in the broadcasted shape
        temp_idx = idx
        multi_idx = []
        for dim in reversed(broadcast_shape):
            multi_idx.append((temp_idx % dim,))
            temp_idx //= dim
        multi_idx.reverse()

        flat_idx_1 = tuple(multi_idx[i] if t1.shape[i] > 1 else 0 for i in range(len(multi_idx)))
        flat_idx_2 = tuple(multi_idx[i] if t2.shape[i] > 1 else 0 for i in range(len(multi_idx)))

        flat_index_1 = t1.flatten_index(flat_idx_1)
        flat_index_2 = t2.flatten_index(flat_idx_2)

        output_data.append(combine_fn(t1.data[flat_index_1], t2.data[flat_index_2]))
    
    return Tensor(output_data, broadcast_shape, requires_grad=True)

def add(t1: Tensor, t2: Tensor) -> Tensor:
    out = broadcast_op(t1, t2, lambda x, y: x + y)

    if out.requires_grad:
        # Derivative of addition operation is always one
        def grad_fn(grad_output):
            grad_t1 = grad_output
            grad_t2 = grad_output

            return grad_t1, grad_t2

        out._grad_fn = grad_fn
        out._parents = [t1, t2]

    return out

def mul(t1: Tensor, t2: Tensor) -> Tensor:
    out = broadcast_op(t1, t2, lambda x, y: x * y)

    if out.requires_grad:
        # Derivative of multiplication leaves behind all variables except the variable the derivative is w.r.t.
        def grad_fn(grad_output):
            grad_t1 = t1.data * grad_output
            grad_t2 = t2.data * grad_output

            return grad_t1, grad_t2

        out._grad_fn = grad_fn
        out._parents = [t1, t2]

    return out

def neg(t: Tensor) -> Tensor:
    out = Tensor(-t.data, t.requires_grad)

    if out.requires_grad:
        # Derivative of multiplication is always -1
        def grad_fn(grad_output):
            grad_t = -grad_output

            return grad_t

        out._grad_fn = grad_fn
        out._parents = [t]

    return out

def inv(t: Tensor) -> Tensor:
    out = Tensor(1 / t.data, t.requires_grad)

    if out.requires_grad:
        # Derivative of 1/x is -1/x^2
        def grad_fn(grad_output):
            grad_t = (-1 / (np.square(t.data))) * grad_output

            return grad_t

        out._grad_fn = grad_fn
        out._parents = [t]

    return out

def exp(t: Tensor) -> Tensor:
    t = t if isinstance(Tensor) else Tensor(t, requires_grad=True)

def eq(t1: Tensor, t2: Tensor) -> Tensor:
    comp_result = t1.data == t2.data

    return Tensor(float(comp_result))