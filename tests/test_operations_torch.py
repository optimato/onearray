import pytest

torch = pytest.importorskip("torch")

import onearray as oa


def test_conversion_and_axis_helpers_preserve_torch_backend():
    source = torch.tensor([1, 2])

    assert oa.asarray(source, mode="torch") is source

    copied = oa.array(source, mode="torch")
    copied[0] = 99
    assert source.tolist() == [1, 2]

    unchanged = oa.add_axis(source)
    unchanged[0] = 99
    assert source.tolist() == [1, 2]

    expanded = oa.add_axis(torch.zeros((2, 3)), -3, -1)
    assert expanded.shape == (2, 1, 3, 1)


def test_math_and_fourier_helpers_preserve_torch_backend():
    source = torch.tensor([1.0, 2.0, 3.0, 4.0])

    assert isinstance(oa.sum(source), torch.Tensor)
    assert oa.sum(source).item() == 10.0
    assert torch.equal(oa.abs(-source), source)
    assert torch.allclose(oa.exp(source), torch.exp(source))

    transformed = oa.fft(source)
    assert isinstance(transformed, torch.Tensor)
    assert torch.allclose(oa.ifft(transformed).real, source)
