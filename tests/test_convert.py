import numpy as np
import pytest

import onearray as oa


def test_array_returns_independent_numpy_storage():
    source = np.array([1, 2])

    result = oa.array(source)
    result[0] = 99

    assert result.tolist() == [99, 2]
    assert source.tolist() == [1, 2]


def test_asarray_reuses_numpy_storage_when_possible():
    source = np.array([1, 2])

    assert oa.asarray(source) is source


@pytest.mark.parametrize(
    "value, expected",
    [
        (3, np.array(3)),
        ([1, 2], np.array([1, 2])),
        (((1, 2), (3, 4)), np.array([[1, 2], [3, 4]])),
    ],
)
def test_asarray_accepts_valid_array_like_values(value, expected):
    np.testing.assert_array_equal(oa.asarray(value), expected)


def test_to_list_handles_arrays_scalars_and_nested_sequences():
    assert oa.to_list(np.array([[1, 2], [3, 4]])) == [[1, 2], [3, 4]]
    assert oa.to_list(np.array(3)) == 3
    assert oa.to_list(((1, 2), (3, 4))) == [[1, 2], [3, 4]]


def test_zeros_like_preserves_numpy_shape_and_dtype():
    source = np.array([[1, 2]], dtype=np.int16)

    result = oa.zeros_like(source)

    assert result.dtype == source.dtype
    assert result.shape == source.shape
    np.testing.assert_array_equal(result, np.zeros((1, 2), dtype=np.int16))


def test_add_axis_without_positions_returns_independent_copy():
    source = np.array([1, 2])

    result = oa.add_axis(source)
    result[0] = 99

    assert result.shape == (2,)
    assert source.tolist() == [1, 2]


@pytest.mark.parametrize(
    "axes, expected_shape",
    [
        ((1,), (2, 1, 3)),
        ((0, 2), (1, 2, 1, 3)),
        ((-1,), (2, 3, 1)),
        ((-3, -1), (2, 1, 3, 1)),
    ],
)
def test_add_axis_inserts_at_final_positions(axes, expected_shape):
    result = oa.add_axis(np.zeros((2, 3)), *axes)

    assert result.shape == expected_shape


@pytest.mark.parametrize(
    "axes, error",
    [
        ((0, -1), ValueError),
        ((0, 0), ValueError),
        ((3,), ValueError),
        ((-4,), ValueError),
        (("0",), TypeError),
    ],
)
def test_add_axis_rejects_invalid_positions(axes, error):
    with pytest.raises(error):
        oa.add_axis(np.zeros((2, 3)), *axes)


@pytest.mark.parametrize("factory", [oa.array, oa.asarray])
def test_custom_numeric_types_require_explicit_conversion(factory):
    from fractions import Fraction
    from decimal import Decimal
    from onearray.validation import is_array_like

    for value in (Fraction(1, 2), Decimal("0.5")):
        assert not is_array_like(value)
        for input_value in (value, [value]):
            with pytest.raises(ValueError):
                factory(input_value)
        assert factory(float(value)).item() == 0.5
