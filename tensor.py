import numpy as np
from typing import Optional, Union

class Tensor:
    def __init__(self, data: Union[float, int, np.ndarray], requires_grad: Optional[bool] = False):
        # Core tensor components
        self.data = np.array(data)
        self.requires_grad = requires_grad
        self._grad = None

        # Autodiff internal requirements
        self._grad_fn = None
        self._parents = []

    """
    Basic operations for Tensors
    """

    # Helper method to setup and compute the basic "out" tensor from operations
    def dual_op_setup(t1: Tensor, t2: Union[Tensor, float, int]) -> Tensor:
        t2 = t2 if instanceof(t2, Tensor) else Tensor(t2)

        data = t1.data + t2.data
        requires_grad = t1.requires_grad or t2.requires_grad

        return Tensor(data, requires_grad)

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def neg(t: Union[Tensor, float, int]) -> Tensor:
        t = t if instanceof(t, Tensor) else Tensor(t)
        out = Tensor(-t.data, t.requires_grad)

        if out.requires_grad:
            # Derivative of multiplication is always -1
            def grad_fn(grad_output):
                grad_t = -grad_output

                return grad_t

            out._grad_fn = grad_fn
            out._parents = [t]

        return out

    @staticmethod
    def inv(t: Union[Tensor, float, int]) -> Tensor:
        t = t if instanceof(t, Tensor) else Tensor(t)
        out = Tensor(1 / t.data, t.requires_grad)

        if out.requires_grad:
            # Derivative of 1/x is -1/x^2
            def grad_fn(grad_output):
                grad_t = (-1 / (np.square(t.data))) * grad_output

                return grad_t

            out._grad_fn = grad_fn
            out._parents = [t]
