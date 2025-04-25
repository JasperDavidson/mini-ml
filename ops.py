from tensor import Tensor
from typing import Union

"""
Basic operations for Tensors
"""

# Helper method to setup and compute the basic "out" tensor from operations
def dual_op_setup(t1: Tensor, t2: Union[Tensor, float, int]) -> Tensor:
    t2 = t2 if isinstance(t2, Tensor) else Tensor(t2)

    data = t1.data + t2.data
    requires_grad = t1.requires_grad or t2.requires_grad

    return Tensor(data, requires_grad)

def add(t1: Union[Tensor, float, int], t2: Union[Tensor, float, int]) -> Tensor:
    out = dual_op_setup(t1, t2)

    if out.requires_grad:
        # Derivative of addition operation is always one
        def grad_fn(grad_output):
            grad_t1 = grad_output
            grad_t2 = grad_output

            return grad_t1, grad_t2

        out._grad_fn = grad_fn
        out._parents = [t1, t2]

    return out

def mul(t1: Union[Tensor, float, int], t2: Union[Tensor, float, int]) -> Tensor:
    out = dual_op_setup(t1, t2)

    if out.requires_grad:
        # Derivative of multiplication leaves behind all variables except the variable the derivative is w.r.t.
        def grad_fn(grad_output):
            grad_t1 = t1.data * grad_output
            grad_t2 = t2.data * grad_output

            return grad_t1, grad_t2

        out._grad_fn = grad_fn
        out._parents = [t1, t2]

    return out

def neg(t: Union[Tensor, float, int]) -> Tensor:
    t = t if isinstance(t, Tensor) else Tensor(t)
    out = Tensor(-t.data, t.requires_grad)

    if out.requires_grad:
        # Derivative of multiplication is always -1
        def grad_fn(grad_output):
            grad_t = -grad_output

            return grad_t

        out._grad_fn = grad_fn
        out._parents = [t]

    return out

def inv(t: Union[Tensor, float, int]) -> Tensor:
    t = t if isinstance(t, Tensor) else Tensor(t)
    out = Tensor(1 / t.data, t.requires_grad)

    if out.requires_grad:
        # Derivative of 1/x is -1/x^2
        def grad_fn(grad_output):
            grad_t = (-1 / (np.square(t.data))) * grad_output

            return grad_t

        out._grad_fn = grad_fn
        out._parents = [t]