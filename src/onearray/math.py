from ._backend import (
    numpy_array_types,
    require_numpy,
    require_torch,
    torch_tensor_types,
)

__all__ = ["sum", "exp", "abs"]


def sum(arr, axis=None, keepdims=False):
    """
    Compute the sum of the elements along the specified axis.

    Parameters
    ----------
    arr : numpy.ndarray or torch.Tensor
        The input array or tensor.
    axis : int, optional
        The axis along which to compute the sum, by default None (sum over all dimensions).
    keepdims : bool, optional
        Whether to keep the dimensions of the input array, by default False.

    Returns
    -------
    numpy.ndarray or torch.Tensor
        The sum of the elements along the specified axis.

    Raises
    ------
    TypeError
        If the input is not a numpy array or torch tensor.
    """

    if isinstance(arr, numpy_array_types):
        np = require_numpy()
        return np.sum(arr, axis=axis, keepdims=keepdims)
    elif isinstance(arr, torch_tensor_types):
        torch = require_torch()
        return torch.sum(arr, dim=axis, keepdim=keepdims)
    else:
        raise TypeError("Input must be a numpy array or torch tensor")


def exp(arr):
    """
    Compute the exponential of all elements in the input array or tensor.

    Parameters
    ----------
    arr : numpy.ndarray or torch.Tensor
        The input array or tensor.

    Returns
    -------
    numpy.ndarray or torch.Tensor
        The exponential of the input array or tensor.

    Raises
    ------
    TypeError
        If the input is not a numpy array or torch tensor.
    """

    if isinstance(arr, numpy_array_types):
        np = require_numpy()
        return np.exp(arr)
    elif isinstance(arr, torch_tensor_types):
        torch = require_torch()
        return torch.exp(arr)
    else:
        raise TypeError("Input must be a numpy array or torch tensor")


def abs(arr):
    """
    Compute the absolute value of all elements in the input array or tensor.

    Parameters
    ----------
    arr : numpy.ndarray or torch.Tensor
        The input array or tensor.

    Returns
    -------
    numpy.ndarray or torch.Tensor
        The absolute value of the input array or tensor.

    Raises
    ------
    TypeError
        If the input is not a numpy array or torch tensor.
    """

    if isinstance(arr, numpy_array_types):
        np = require_numpy()
        return np.abs(arr)
    elif isinstance(arr, torch_tensor_types):
        torch = require_torch()
        return torch.abs(arr)
    else:
        raise TypeError("Input must be a numpy array or torch tensor")
