import numpy as np
import torch
from .errors import ArrayError
from . import checks

__all__ = ["fft", "ifft", "fftfreq", "fftshift", "ifftshift"]


def fft(arr, axis=None):
    """
    Compute the Fast Fourier Transform of the input array/tensor in 1D or 2D.

    Parameters
    ----------
    arr: numpy.ndarray or torch.Tensor
        Input array
    axis: int or tuple of int, optional
        Axis or axes over which to compute the FFT, by default None (all axes)

    Returns
    -------
    numpy.ndarray or torch.Tensor
        FFT of input with same type as input

    Raises
    ------
    TypeError
        If input is not a numpy array or torch tensor
    """
    if not isinstance(arr, (np.ndarray, torch.Tensor)):
        raise TypeError("Input must be a numpy array or torch tensor")

    if axis is None:
        axis = tuple(range(arr.ndim))

    if isinstance(arr, np.ndarray):
        return (
            np.fft.fft2(arr, axes=axis)
            if len(axis) == 2
            else np.fft.fft(arr, axis=axis)
        )
    else:
        return (
            torch.fft.fft2(arr, dim=axis)
            if len(axis) == 2
            else torch.fft.fft(arr, dim=axis)
        )


def ifft(arr, axis=None):
    """
    Compute the Inverse Fast Fourier Transform of the input array/tensor in 1D or 2D.

    Parameters
    ----------
    arr: numpy.ndarray or torch.Tensor
        Input array
    axis: int or tuple of int, optional
        Axis or axes over which to compute the IFFT, by default None (all axes)

    Returns
    -------
    numpy.ndarray or torch.Tensor
        IFFT of input with same type as input

    Raises
    ------
    TypeError
        If input is not a numpy array or torch tensor
    """
    if not isinstance(arr, (np.ndarray, torch.Tensor)):
        raise TypeError("Input must be a numpy array or torch tensor")

    if axis is None:
        axis = tuple(range(arr.ndim))
    if isinstance(arr, np.ndarray):
        return (
            np.fft.ifft2(arr, axes=axis)
            if len(axis) == 2
            else np.fft.ifft(arr, axis=axis)
        )
    else:
        return (
            torch.fft.ifft2(arr, dim=axis)
            if len(axis) == 2
            else torch.fft.ifft(arr, dim=axis)
        )


def fftfreq(n, d=1.0, mode="numpy"):
    """
    Return the Discrete Fourier Transform sample frequencies.
    Works in 1D or 2D.

    Parameters
    ----------
    n: int or tuple of int
        Window length or shape for 2D
    d: float or tuple of float, optional
        Sample spacing (inverse of sampling rate), by default 1.0
    mode: str, optional
        'numpy', 'torch', or 'torch_cuda' output type, by default 'numpy'

    Returns
    -------
    np.ndarray or torch.Tensor
        Array of frequencies with specified type

    Raises
    ------
    ValueError
        If mode is not valid or dimensions don't match
    """
    if not checks.is_valid_array_mode(mode):
        raise ValueError(ArrayError.INVALID_ARRAY_MODE.value)

    # Handle 2D case
    if isinstance(n, tuple):
        # If d is not a tuple, create a tuple of same length as n with d value
        if not isinstance(d, tuple):
            d = tuple(d for _ in range(len(n)))
        elif len(d) != len(n):
            raise ValueError("For 2D, both n and d must be tuples of same length")

        if mode == "torch" or mode == "torch_cuda":
            freqs = [fftfreq(ni, di, mode) for ni, di in zip(n, d)]
            freq_grid = torch.meshgrid(freqs, indexing="ij")
            return freq_grid
        else:
            return np.meshgrid(
                *(np.fft.fftfreq(ni, di) for ni, di in zip(n, d)), indexing="ij"
            )

    # Original 1D case
    if mode == "torch" or mode == "torch_cuda":
        val = 1.0 / (n * d)
        results = torch.empty(n)
        N = (n - 1) // 2 + 1
        results[:N] = torch.arange(0, N)
        results[N:] = torch.arange(-(n // 2), 0)
        results = results * val
        return results.cuda() if mode == "torch_cuda" else results
    else:
        return np.fft.fftfreq(n, d)


def fftshift(arr, axes=None):
    """
    Shift the zero-frequency component to the center of the spectrum.

    This function swaps half-spaces for all axes listed (defaults to all).
    The output has the DC (zero-frequency) component in the center of the array.

    Parameters
    ----------
    arr: numpy.ndarray or torch.Tensor
        Input array
    axes: int or tuple of int, optional
        Axes over which to shift. Default is None, which shifts all axes.

    Returns
    -------
    numpy.ndarray or torch.Tensor
        The shifted array, same type as input

    Raises
    ------
    TypeError
        If input is not a numpy array or torch tensor
    """
    if not isinstance(arr, (np.ndarray, torch.Tensor)):
        raise TypeError("Input must be a numpy array or torch tensor")

    if isinstance(arr, np.ndarray):
        return np.fft.fftshift(arr, axes=axes)
    else:
        # torch.fft.fftshift uses 'dim' instead of 'axes'
        return torch.fft.fftshift(arr, dim=axes)


def ifftshift(arr, axes=None):
    """
    Inverse of fftshift. Shift the zero-frequency component back to the beginning.

    This function is the inverse of fftshift. Although identical for even-length arrays,
    the functions differ by one sample for odd-length arrays.

    Parameters
    ----------
    arr: numpy.ndarray or torch.Tensor
        Input array
    axes: int or tuple of int, optional
        Axes over which to shift. Default is None, which shifts all axes.

    Returns
    -------
    numpy.ndarray or torch.Tensor
        The shifted array, same type as input

    Raises
    ------
    TypeError
        If input is not a numpy array or torch tensor
    """
    if not isinstance(arr, (np.ndarray, torch.Tensor)):
        raise TypeError("Input must be a numpy array or torch tensor")

    if isinstance(arr, np.ndarray):
        return np.fft.ifftshift(arr, axes=axes)
    else:
        # torch.fft.ifftshift uses 'dim' instead of 'axes'
        return torch.fft.ifftshift(arr, dim=axes)
