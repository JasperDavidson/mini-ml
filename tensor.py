import numpy as np
from typing import Optional, Union

class Tensor:
    def __init__(self, data: Union[float, int, np.ndarray], requires_grad: Optional[bool] = False):
        # Core tensor components
        self.data = np.array(data)
        self.requires_grad = requires_grad
        self._grad = None
        self._prev_grad = None

        # Autodiff internal requirements
        self._grad_fn = None
        self._parents = []