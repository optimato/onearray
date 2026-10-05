import numpy as np
import pytest

import onearray as oa


def test_math_helpers_preserve_numpy_backend_and_values():
    source = np.array([-1.0, 0.0, 1.0])

    assert isinstance(oa.sum(source), np.floating)
    assert oa.sum(source) == 0.0
    np.testing.assert_allclose(oa.exp(source), np.exp(source))
    np.testing.assert_array_equal(oa.abs(source), np.array([1.0, 0.0, 1.0]))


@pytest.mark.parametrize("operation", [oa.sum, oa.exp, oa.abs])
def test_math_helpers_reject_non_array_inputs(operation):
    with pytest.raises(TypeError):
        operation([1, 2])


def test_fft_and_ifft_round_trip_numpy_values():
    source = np.array([1.0, 2.0, 3.0, 4.0])

    transformed = oa.fft(source)

    assert isinstance(transformed, np.ndarray)
    np.testing.assert_allclose(oa.ifft(transformed), source)


def test_fft_helpers_match_numpy_for_axes_and_shifts():
    source = np.arange(6).reshape(2, 3)

    np.testing.assert_allclose(oa.fft(source, axis=1), np.fft.fft(source, axis=1))
    np.testing.assert_array_equal(oa.fftshift(source), np.fft.fftshift(source))
    np.testing.assert_array_equal(oa.ifftshift(source), np.fft.ifftshift(source))


def test_fftfreq_matches_numpy():
    np.testing.assert_allclose(oa.fftfreq(4, d=0.5), np.fft.fftfreq(4, d=0.5))
