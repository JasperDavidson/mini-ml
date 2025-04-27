import pytest
import numpy as np

from tensor import Tensor
from hypothesis import given, strategies as st
import ops

@given(st.floats(), st.floats())
def test_add(a: float, b: float):
    x = Tensor(a, False)
    y = Tensor(b, False)
    z = Tensor(a + b, False)

    assert ops.eq(ops.add(x, y), z)

@given(st.floats(), st.floats())
def test_mul(a: float, b: float):
    x = Tensor(a, False)
    y = Tensor(b, False)
    z = Tensor(a * b, False)

    assert ops.eq(ops.mul(x, y), z)

@given(st.floats(), st.floats())
def test_sub(a: float, b: float):
    x = Tensor(a, False)
    y = Tensor(b, False)
    z = Tensor(a - b, False)

    assert ops.eq(ops.add(x, ops.neg(y)), z)

@given(st.floats(), st.floats().filter(lambda n: n != 0))
def test_div(a: float, b: float):
    x = Tensor(a, False)
    y = Tensor(b, False)
    z = Tensor(a / b, False)

    assert ops.eq(ops.mul(x, ops.inv(y)), z)