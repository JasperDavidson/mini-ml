from collections import deque
from tensor import Tensor
from typing import Iterable

"""
For autodifferentiation, we need to have a topological graph we can trace back the grads from, accumulating them until we reach the end
"""
def build_topo(tensor: Tensor) -> Iterable[Tensor]:
    topo_list = []

    # Need 'visited' in case one operation result is used as input to more than one other operation
    visited = set()

    queue = deque()
    queue.appendleft(tensor)

    while queue:
        next_op = queue.pop()

        if next_op not in visited:
            visited.add(next_op)
            topo_list.append(next_op)

            for parent in next_op._parents:
                queue.appendleft(parent)

    return topo_list

"""
Backpropagation which updates the gradients for each computation
"""
def backprop(topo_list: Iterable[Tensor]):
    for tensor in topo_list:
        grads = tensor._grad_fn(tensor._prev_grad) if tensor._grad_fn else None

        if grads is None:
            continue

        tensor._grad = sum(grads) if isinstance(grads, tuple) else grads

        for parent in tensor._parents:
            if parent._prev_grad is None:
                parent._prev_grad = tensor._grad
            else:
                parent._prev_grad += tensor._grad