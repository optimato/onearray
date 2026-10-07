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


@pytest.mark.parametrize("value", [1, np.array(1), np.array("text"), "text"])
def test_length_rejects_scalars_and_invalid_values(value):
    from onearray.shape import len as array_len

    with pytest.raises(TypeError):
        array_len(value)


def test_dimension_rejects_non_numeric_arrays():
    with pytest.raises(TypeError):
        dim(np.array(["text"]))


@pytest.mark.parametrize(
    "value, expected", [([], 0), ([[1, 2]], 1), (np.zeros((3, 2)), 3)]
)
def test_length_of_non_scalar_arrays(value, expected):
    from onearray.shape import len as array_len

    assert array_len(value) == expected
