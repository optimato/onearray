import pytest

torch = pytest.importorskip("torch")

from onearray.validation import (  # noqa: E402
    is_array,
    is_array_like,
    is_real,
    is_real_nonnegative,
    is_real_positive,
)


@pytest.mark.parametrize("dtype", [torch.bool, torch.int64, torch.float32, torch.complex64])
def test_torch_numeric_and_boolean_tensors_are_arrays(dtype):
    tensor = torch.tensor([0, 1], dtype=dtype)
    assert is_array(tensor)
    assert is_array_like(tensor)


def test_torch_real_predicates():
    assert is_real(torch.tensor([-1, 0, 2.5]))
    assert is_real_positive(torch.tensor([0.5, 2]))
    assert is_real_nonnegative(torch.tensor([0, 2]))

    assert not is_real(torch.tensor([1, torch.inf]))
    assert not is_real_positive(torch.tensor([0, 1]))
    assert not is_real_nonnegative(torch.tensor([-1, 2]))


@pytest.mark.parametrize("dtype", [torch.bool, torch.complex64])
def test_torch_boolean_and_complex_values_are_not_real(dtype):
    assert not is_real(torch.tensor([1], dtype=dtype))


@pytest.mark.parametrize(
    "predicate", [is_real, is_real_positive, is_real_nonnegative]
)
def test_torch_real_predicates_accept_empty_tensors(predicate):
    assert predicate(torch.tensor([]))
