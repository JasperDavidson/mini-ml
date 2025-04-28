from typing import Optional, Iterable, Tuple, List

class Tensor:
    def __init__(self, data: Iterable[float], shape: Optional[Tuple[int, ...]] = (1,), requires_grad: Optional[bool] = False):
        # Core tensor components
        self.data = data
        self.shape = shape
        self.requires_grad = requires_grad
        self.strides = self._compute_strides()

        # Autodiff internal requirements
        self._grad = None
        self._prev_grad = None
        self._grad_fn = None
        self._parents = []

    # Compute the stride for each dimension --> how many elements to progress in 1D array to get to the next section of the same dimension
    def _compute_strides(self) -> Tuple[int, ...]:
        strides = (1,)
        stride = 1

        for dim in self.shape:
            stride *= dim
            strides += (stride,)
        
        # Returns the strides such that it reflects the order of the shape
        return reversed(strides)


    # Calculate the flattened index based on the multi-dimensional index and shape
    def flatten_index(self, multi_idx: Iterable[int]) -> int:
        flat_idx = -1

        for (i, idx) in enumerate(multi_idx):
            flat_idx += self.strides[i] * idx
        
        return flat_idx

    @staticmethod
    # Compute the broadcasted shape result of performing on operation on two tensors
    # Takes the tensor shapes as input
    def broadcast_shapes(shape_one: Tuple[int, ...], shape_two: Tuple[int, ...]) -> Tuple[int, ...]:
        broadcast_shape = ()

        if (len(shape_one) < len(shape_two)):
            broadcast_shape += (1,) * (len(shape_two) - len(shape_one))
        elif (len(shape_two) < len(shape_one)):
            broadcast_shape += (1,) * (len(shape_one) - len(shape_two))

        for (dim1, dim2) in zip(shape_one, shape_two):
            if dim1 == dim2:
                # Doesn't matter which one we add
                broadcast_shape += (dim1,)
            elif dim2 == 1:
                broadcast_shape += (dim1,)
            elif dim1 == 1:
                broadcast_shape += (dim2,)
            else:
                raise ValueError("The two tensors cannot be operated on: broadcast error")
        
        return broadcast_shape