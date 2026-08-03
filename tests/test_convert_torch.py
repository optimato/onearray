import pytest

torch = pytest.importorskip("torch")

from onearray.convert import to_list, zeros_like  # noqa: E402


def test_to_list_converts_torch_tensors_to_python_values():
    assert to_list(torch.tensor(3)) == 3
    assert to_list(torch.tensor([[1, 2], [3, 4]])) == [[1, 2], [3, 4]]


@pytest.mark.parametrize("dtype", [torch.bool, torch.int64, torch.float32])
def test_zeros_like_preserves_torch_shape_dtype_and_device(dtype):
    source = torch.ones((2, 3), dtype=dtype)
    result = zeros_like(source)

    assert result.shape == source.shape
    assert result.dtype == source.dtype
    assert result.device == source.device
    assert not result.any().item()
