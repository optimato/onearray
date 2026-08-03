import numpy as np
import pytest

from onearray.validation import (
    is_array,
    is_array_like,
    is_numberlike,
    is_real,
    is_real_nonnegative,
    is_real_positive,
)


@pytest.mark.parametrize(
    "value",
    [True, 1, 1.5, 1j, np.int64(1), np.float32(1), np.bool_(True)],
)
def test_numberlike_accepts_numeric_scalars(value):
    assert is_numberlike(value)


@pytest.mark.parametrize(
    "value",
    [None, "1", b"1", np.str_("1"), np.datetime64("2026-01-01")],
)
def test_numberlike_rejects_other_values(value):
    assert not is_numberlike(value)


def test_array_means_numeric_backend_container():
    assert is_array(np.array([1, 2]))
    assert is_array(np.array([True, False]))

    assert not is_array([1, 2])
    assert not is_array(np.array(["one", "two"]))


@pytest.mark.parametrize(
    "value, expected",
    [
        (1, True),
        ([1, 2], True),
        ([[1, 2], [3, 4]], True),
        ([[1], [2, 3]], False),
        ([1, "two"], False),
        ("123", False),
    ],
)
def test_array_like_requires_one_rectangular_numeric_shape(value, expected):
    assert is_array_like(value) is expected


@pytest.mark.parametrize(
    "predicate, accepted, rejected",
    [
        (is_real, [0, -1, 2.5], [1, float("inf")]),
        (is_real_positive, [0.5, 2], [0, 1]),
        (is_real_nonnegative, [0, 2], [-1, 2]),
    ],
)
def test_real_predicates(predicate, accepted, rejected):
    assert predicate(accepted)
    assert not predicate(rejected)
    assert predicate([])  # Universal predicates are true for empty inputs.


@pytest.mark.parametrize("value", [True, 1j])
def test_real_predicates_reject_boolean_and_complex_values(value):
    assert not is_real(value)


def test_real_predicates_raise_for_non_array_like_values():
    with pytest.raises(TypeError):
        is_real([1, "two"])
