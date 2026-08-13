from collections.abc import Sequence

from ._backend import (
    numpy_array_types,
    require_numpy,
    require_torch,
    torch_tensor_types,
)
from .errors import ArrayError
from .validation import is_array, is_array_like, is_valid_array_mode

__all__ = ["add_axis", "array", "asarray", "to_list", "zeros_like"]


def array(arr, mode="numpy"):
    """
    Convert input to numpy or torch tensor based on mode.

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
    ret = asarray(arr, mode=mode)
    if isinstance(ret, numpy_array_types):
        return ret.copy()
    elif isinstance(ret, torch_tensor_types):
        return ret.clone()
    else:
        raise TypeError("Unexpected return type from asarray function")


def asarray(arr, mode="numpy"):
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
            # if not is_array_like(arr):
            #     raise ValueError(ArrayError.INVALID_ARRAY_TYPE.value)
            return np.asarray(arr)
        else:
            raise ValueError(ArrayError.INVALID_ARRAY_TYPE.value)

    elif mode == "torch":
        torch = require_torch()
        if isinstance(arr, numpy_array_types):
            return torch.as_tensor(arr)
        elif isinstance(arr, Sequence) and not isinstance(arr, (str, bytes, bytearray)):
            # if not is_array_like(arr):
            #     raise ValueError(ArrayError.INVALID_ARRAY_TYPE.value)
            if all(is_array(x) for x in arr):
                return torch.stack(
                    [
                        (
                            x
                            if isinstance(x, torch_tensor_types)
                            and x.device.type == "cpu"
                            else (
                                x.cpu()
                                if isinstance(x, torch_tensor_types)
                                else torch.as_tensor(x)
                            )
                        )
                        for x in arr
                    ]
                )
            return torch.as_tensor(arr)
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
            return torch.as_tensor(arr, device="cuda")
        elif isinstance(arr, Sequence) and not isinstance(arr, (str, bytes, bytearray)):
            # if not is_array_like(arr):
            #     raise ValueError(ArrayError.INVALID_ARRAY_TYPE.value)
            if all(is_array(x) for x in arr):
                return torch.stack(
                    [
                        (
                            x
                            if isinstance(x, torch_tensor_types) and x.is_cuda
                            else (
                                x.cuda()
                                if isinstance(x, torch_tensor_types)
                                else torch.as_tensor(x, device="cuda")
                            )
                        )
                        for x in arr
                    ]
                )
            return torch.as_tensor(arr, device="cuda")
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
    is_numpy = isinstance(arr, numpy_array_types)
    is_torch = isinstance(arr, torch_tensor_types)
    if not (is_numpy or is_torch):
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
    if is_numpy:
        result = arr.copy()
    elif is_torch:
        result = arr.clone()

    if is_numpy:
        np = require_numpy()
        for axis in axes:
            result = np.expand_dims(result, axis=axis)
    elif is_torch:
        torch = require_torch()
        for axis in axes:
            result = torch.unsqueeze(result, dim=axis)

    return result


def to_list(arr):
    """
    Convert an array-like input to nested Python lists.

    Parameters
    ----------
    arr : array-like
        The input array-like object.

    Returns
    -------
    list | scalar
        The converted Python list (or a scalar for 0-d inputs).

    Raises
    ------
    TypeError
        If the input is not array-like.
    """
    if not is_array_like(arr):
        raise TypeError(ArrayError.INVALID_ARRAY_TYPE.value)

    def _to_list(value):
        if isinstance(value, numpy_array_types):
            return value.tolist()
        if isinstance(value, torch_tensor_types):
            tensor = value.detach()
            if tensor.device.type != "cpu":
                tensor = tensor.cpu()
            return tensor.tolist()
        if isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            return [_to_list(item) for item in value]
        return value

    return _to_list(arr)


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
