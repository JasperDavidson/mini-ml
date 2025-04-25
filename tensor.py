import numpy as np
from typing import Callable, Tuple, Optional, List, Union

class Tensor:
    def __init__(self, data: Union[float, int, np.ndarray], requires_grad: bool = False):
        # Core tensor components
        self.data = np.array(data)
        self.requires_grad = requires_grad
        self._grad = None

        # Autodiff internal requirements
        self._grad_fn = None
        self._parents = []

    """
    Basic pythonic operations for Tensors
    """

    # Helper method to setup and compute the basic "out" tensor from operations
    def dual_op_setup(t1: "Tensor", t2: Union["Tensor", float, int]) -> "Tensor":
        t2 = t2 if instanceof(t2, Tensor) else Tensor(t2)

        data = t1.data + t2.data
        requires_grad = t1.requires_grad or t2.requires_grad

        return Tensor(data, requires_grad)

    def __add__(self, other: Union["Tensor", float, int]) -> "Tensor":
        out = dual_op_setup(self, other)

        if out.requires_grad:
            # Derivative of addition operation is always one
            def grad_fn(grad_output):
                grad_self = grad_output
                grad_other = grad_output

                return grad_self, grad_other

            out._grad_fn = grad_fn
            out._parents = [self, other]

        return out

    def __mul__(self, other: Union["Tensor", float, int]) -> "Tensor":
        out = dual_op_setup(self, other)

        if out.requires_grad:
            # Derivative of multiplication leaves behind all variables except the variable the derivative is w.r.t.
            def grad_fn(grad_output):
                grad_self = other.data * grad_output
                grad_other = self.data * grad_output

                return grad_self, grad_other

            out._grad_fn = grad_fn
            out._parents = [self, other]

        return out
