import pytest

import onearray as oa

torch = pytest.importorskip("torch")


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


def test_multidimensional_fftfreq_preserves_torch_meshgrid_container():
    result = oa.fftfreq((2, 3), d=(0.5, 1.0), mode="torch")
    expected = torch.meshgrid(
        torch.fft.fftfreq(2, 0.5), torch.fft.fftfreq(3, 1.0), indexing="ij"
    )

    assert isinstance(result, tuple)
    assert len(result) == len(expected)
    assert all(
        torch.equal(actual, expected_component)
        for actual, expected_component in zip(result, expected)
    )


def test_fftfreq_cuda_mode_checks_cuda_availability(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="torch_cuda"):
        oa.fftfreq(4, mode="torch_cuda")


@pytest.mark.parametrize("factory", [oa.array, oa.asarray])
@pytest.mark.parametrize("value", [[], (), [[], []]])
@pytest.mark.parametrize("mode", ["torch", "torch_cuda"])
def test_empty_sequences_convert_to_torch(factory, value, mode):
    if mode == "torch_cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    result = factory(value, mode=mode)
    assert result.numel() == 0
    assert tuple(result.shape) == ((2, 0) if value == [[], []] else (0,))
    assert result.device.type == ("cuda" if mode == "torch_cuda" else "cpu")


def test_scalar_tensor_has_no_length():
    from onearray.shape import len as array_len

    with pytest.raises(TypeError):
        array_len(torch.tensor(1))
