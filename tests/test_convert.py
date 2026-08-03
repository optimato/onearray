import numpy as np
import pytest

from onearray.convert import to_list, zeros_like


@pytest.mark.parametrize(
    "value, expected",
    [
        (3, 3),
        (np.float32(1.5), 1.5),
        (np.array(3), 3),
        (((1, 2), (3, 4)), [[1, 2], [3, 4]]),
        (np.array([[1, 2], [3, 4]]), [[1, 2], [3, 4]]),
    ],
)
def test_to_list_returns_python_values(value, expected):
    result = to_list(value)
    assert result == expected
    if not isinstance(expected, list):
        assert type(result) is type(expected)


@pytest.mark.parametrize("value", ["123", [[1], [2, 3]], np.array(["one"])])
def test_to_list_rejects_non_array_like_values(value):
    with pytest.raises(TypeError):
        to_list(value)


@pytest.mark.parametrize("dtype", [np.bool_, np.int32, np.float64, np.complex64])
def test_zeros_like_preserves_numpy_shape_and_dtype(dtype):
    source = np.ones((2, 3), dtype=dtype)
    result = zeros_like(source)

    assert result.shape == source.shape
    assert result.dtype == source.dtype
    assert not result.any()


@pytest.mark.parametrize("value", [[1, 2], np.array(["one", "two"])])
def test_zeros_like_requires_an_array_container(value):
    with pytest.raises(TypeError):
        zeros_like(value)
