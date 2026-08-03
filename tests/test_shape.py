import numpy as np
import pytest

from onearray.shape import dim, infer_shape


@pytest.mark.parametrize(
    "value, shape",
    [
        (3, ()),
        ([], (0,)),
        ([1, 2, 3], (3,)),
        (((1, 2), (3, 4)), (2, 2)),
        (np.zeros((2, 3)), (2, 3)),
    ],
)
def test_infer_shape(value, shape):
    assert infer_shape(value) == shape
    assert dim(value) == len(shape)


@pytest.mark.parametrize("value", ["123", [1, "two"], [[1], [2, 3]]])
def test_invalid_values_have_no_shape(value):
    assert infer_shape(value) is None
    with pytest.raises(TypeError):
        dim(value)


def test_reusing_a_child_sequence_is_not_a_cycle():
    child = [1, 2]
    assert infer_shape([child, child]) == (2, 2)


def test_recursive_sequences_have_no_shape():
    recursive = []
    recursive.append(recursive)
    assert infer_shape(recursive) is None
