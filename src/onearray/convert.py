from collections.abc import Sequence

from ._backend import (
    numpy_array_types,
    require_numpy,
    require_torch,
    torch_tensor_types,
)
from .errors import ArrayError
from .validation import is_array, is_array_like, is_valid_array_mode

__all__ = ["array", "add_axis", "to_list", "zeros_like"]


def array(arr, mode="numpy"):
    """
    Convert input to numpy or torch tensor based on mode.

    Allowed input types are:
    - sequence of numbers
    - numpy ndarray
    - torch Tensor
    - sequence of sequences (ND array)
    - sequence of torch Tensors
    - sequence of numpy ndarrays

    Parameters
    ----------
    arr : Sequence, np.ndarray, or torch.Tensor
        The input array to convert.
    mode : str, optional
        The target array mode ('numpy', 'torch', 'torch_cuda'), by default 'numpy'

    Returns
    -------
    np.ndarray or torch.Tensor
        The converted array.

    Raises
    ------
    ValueError
        If the input array is of an invalid type.
    ValueError
        If the array mode is invalid.
    RuntimeError
        If CUDA is not available when requested.
    """

    if not is_array_like(arr):
        raise ValueError(ArrayError.INVALID_ARRAY_TYPE.value)

    if not is_valid_array_mode(mode):
        raise ValueError(ArrayError.INVALID_ARRAY_MODE.value)

    if mode == "numpy":
        np = require_numpy()
        if isinstance(arr, torch_tensor_types):
            return arr.cpu().detach().numpy()
        elif isinstance(arr, numpy_array_types):
            return arr
        elif isinstance(arr, Sequence) and not isinstance(arr, (str, bytes, bytearray)):
            if not is_array_like(arr):
                raise ValueError(ArrayError.INVALID_ARRAY_TYPE.value)
            return np.array(arr)
        else:
            raise ValueError(ArrayError.INVALID_ARRAY_TYPE.value)

    elif mode == "torch":
        torch = require_torch()
        if isinstance(arr, numpy_array_types):
            return torch.tensor(arr.copy())
        elif isinstance(arr, Sequence) and not isinstance(arr, (str, bytes, bytearray)):
            if not is_array_like(arr):
                raise ValueError(ArrayError.INVALID_ARRAY_TYPE.value)
            if all(is_array(x) for x in arr):
                return torch.stack(
                    [
                        (
                            x
                            if isinstance(x, torch_tensor_types) and x.device.type == "cpu"
                            else (
                                x.cpu()
                                if isinstance(x, torch_tensor_types)
                                else torch.tensor(x)
                            )
                        )
                        for x in arr
                    ]
                )
            return torch.tensor(arr)
        elif isinstance(arr, torch_tensor_types) and arr.is_cuda:
            # If the tensor is on GPU, move it to CPU first
            return arr.cpu()
        elif isinstance(arr, torch_tensor_types):
            return arr
        else:
            raise ValueError(ArrayError.INVALID_ARRAY_TYPE.value)

    elif mode == "torch_cuda":
        torch = require_torch()
        if not torch.cuda.is_available():
            raise RuntimeError(ArrayError.CUDA_NOT_AVAILABLE.value)
        if isinstance(arr, numpy_array_types):
            return torch.tensor(arr.copy(), device="cuda")
        elif isinstance(arr, Sequence) and not isinstance(arr, (str, bytes, bytearray)):
            if not is_array_like(arr):
                raise ValueError(ArrayError.INVALID_ARRAY_TYPE.value)
            if all(is_array(x) for x in arr):
                return torch.stack(
                    [
                        (
                            x
                            if isinstance(x, torch_tensor_types) and x.is_cuda
                            else (
                                x.cuda()
                                if isinstance(x, torch_tensor_types)
                                else torch.tensor(x, device="cuda")
                            )
                        )
                        for x in arr
                    ]
                )
            return torch.tensor(arr).cuda()
        elif isinstance(arr, torch_tensor_types) and arr.is_cuda:
            # If the tensor is already on GPU, return it as is
            return arr
        elif isinstance(arr, torch_tensor_types):
            # Move the tensor to GPU
            return arr.cuda()
        else:
            raise ValueError(ArrayError.INVALID_ARRAY_TYPE.value)

    else:
        raise ValueError(ArrayError.INVALID_ARRAY_MODE.value)


def add_axis(arr, *axes):
    """
    Add new axes to array or tensor at specified positions.

    Parameters
    ----------
    arr : np.ndarray or torch.Tensor
        The input array or tensor.
    *axes : int
        Variable number of integer positions where to add new axes.
    Returns
    -------
    np.ndarray or torch.Tensor
        The array or tensor with new axes added.

    Raises
    ------
    TypeError
        If the input is not a numpy array or torch tensor.
    ValueError
        If the input array is not a valid shape.
    ValueError
        If the number of axes is too large.


    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([1, 2, 3])
    >>> _add_axis(x, 1)  # equivalent to x[:, np.newaxis]
    array([[1],
           [2],
           [3]])
    >>> _add_axis(x, 0, 2)  # equivalent to x[np.newaxis, :, np.newaxis]
    array([[[1],
            [2],
            [3]]])
    """
    if not isinstance(arr, numpy_array_types + torch_tensor_types):
        raise TypeError("Input must be a numpy array or torch tensor")

    if max(axes) >= len(axes) + arr.ndim:
        raise ValueError("Too many dimensions")

    # Sort axes in descending order to avoid shifting positions
    if all(x >= 0 for x in axes):
        axes = sorted(axes)  # , reverse=True)
    elif all(x < 0 for x in axes):
        axes = sorted(axes, reverse=True)
    else:
        raise ValueError("All axes must be either positive or negative")

    # Create a copy to avoid modifying the original
    if isinstance(arr, numpy_array_types):
        result = arr.copy()
    elif isinstance(arr, torch_tensor_types):
        result = arr.clone()
    else:
        raise TypeError("Input must be a numpy array or torch tensor")

    if isinstance(result, numpy_array_types):
        np = require_numpy()
        for axis in axes:
            result = np.expand_dims(result, axis=axis)
    elif isinstance(result, torch_tensor_types):
        torch = require_torch()
        for axis in axes:
            result = torch.unsqueeze(result, dim=axis)
    else:
        raise TypeError("Input must be a numpy array or torch tensor")

    return result


def to_list(arr):
    """
    Convert a numpy array, torch tensor, or list to a Python list.

    Parameters
    ----------
    arr : list, np.ndarray, or torch.Tensor
        The input array or tensor.

    Returns
    -------
    list
        The converted Python list.

    Raises
    ------
    TypeError
        If the input is not a list, numpy array, or torch tensor.
    """
    if isinstance(arr, list):
        return arr
    elif isinstance(arr, numpy_array_types):
        return arr.tolist()
    elif isinstance(arr, torch_tensor_types):
        tensor = arr.detach()
        if tensor.device.type == "cpu":
            return tensor.tolist()
        return tensor.cpu().tolist()
    else:
        raise TypeError("Input must be a list, numpy array, or torch tensor")


def zeros_like(arr):
    """
    Create an array/tensor of zeros with the same shape and type as the input.

    Parameters
    ----------
    arr: np.ndarray or torch.Tensor
        Input array

    Returns
    -------
    np.ndarray or torch.Tensor
        Array of zeros with same shape and type as input

    Raises
    ------
    TypeError
        If input is not a numpy array or torch tensor
    """
    if isinstance(arr, numpy_array_types):
        np = require_numpy()
        return np.zeros_like(arr)
    elif isinstance(arr, torch_tensor_types):
        torch = require_torch()
        return torch.zeros_like(arr)
    else:
        raise TypeError("Input must be a numpy array or torch tensor")
