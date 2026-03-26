from . import checks
from .errors import ArrayError
import numpy as np
import torch

__all__ = ["array", "add_axis", "to_list", "zeros_like"]


def array(arr, mode="numpy"):
    """
    Convert input to numpy or torch tensor based on mode.

    Allowed input types are:
    - list of numbers
    - numpy ndarray
    - torch Tensor
    - list of lists (ND array)
    - list of torch Tensors
    - list of numpy ndarrays

    Parameters
    ----------
    arr : list, np.ndarray, or torch.Tensor
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

    if not checks.is_array(arr):
        raise ValueError(ArrayError.INVALID_ARRAY_TYPE.value)

    if not checks.is_valid_array_mode(mode):
        raise ValueError(ArrayError.INVALID_ARRAY_MODE.value)

    if mode == "numpy":
        if isinstance(arr, torch.Tensor):
            return arr.cpu().detach().numpy()
        elif isinstance(arr, np.ndarray):
            return arr
        elif isinstance(arr, list):
            return np.array(arr)
        else:
            raise ValueError(ArrayError.INVALID_ARRAY_TYPE.value)

    elif mode == "torch":
        if isinstance(arr, np.ndarray):
            return torch.tensor(arr.copy())
        elif isinstance(arr, list):
            if checks.is_numeric_array(arr):
                return torch.tensor(arr)
            elif all(checks.is_numeric_array(x) for x in arr):
                return torch.stack(
                    [
                        (
                            x
                            if isinstance(x, torch.Tensor) and x.is_cpu
                            else (
                                x.cpu()
                                if isinstance(x, torch.Tensor)
                                else torch.tensor(x)
                            )
                        )
                        for x in arr
                    ]
                )
            else:
                raise ValueError(ArrayError.INVALID_ARRAY_TYPE.value)
        elif isinstance(arr, torch.Tensor) and arr.is_cuda:
            # If the tensor is on GPU, move it to CPU first
            return arr.cpu()
        elif isinstance(arr, torch.Tensor):
            return arr
        else:
            raise ValueError(ArrayError.INVALID_ARRAY_TYPE.value)

    elif mode == "torch_cuda" and torch.cuda.is_available():
        if isinstance(arr, np.ndarray):
            return torch.tensor(arr.copy(), device="cuda")
        elif isinstance(arr, list):
            if checks.is_numeric_array(arr):
                return torch.tensor(arr).cuda()
            elif all(checks.is_numeric_array(x) for x in arr):
                return torch.stack(
                    [
                        (
                            x
                            if isinstance(x, torch.Tensor) and x.is_cuda
                            else (
                                x.cuda()
                                if isinstance(x, torch.Tensor)
                                else torch.tensor(x).cuda()
                            )
                        )
                        for x in arr
                    ]
                )
            else:
                raise ValueError(ArrayError.INVALID_ARRAY_TYPE.value)
        elif isinstance(arr, torch.Tensor) and arr.is_cuda:
            # If the tensor is already on GPU, return it as is
            return arr
        elif isinstance(arr, torch.Tensor):
            # Move the tensor to GPU
            return arr.cuda()
        else:
            raise ValueError(ArrayError.INVALID_ARRAY_TYPE.value)

    elif mode == "torch_cuda" and not torch.cuda.is_available():
        raise RuntimeError(ArrayError.CUDA_NOT_AVAILABLE.value)

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
    if not isinstance(arr, (np.ndarray, torch.Tensor)):
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
    result = arr.copy() if isinstance(arr, np.ndarray) else arr.clone()

    if isinstance(result, np.ndarray):
        for axis in axes:
            result = np.expand_dims(result, axis=axis)
    else:  # torch.Tensor
        for axis in axes:
            result = torch.unsqueeze(result, dim=axis)

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
    elif isinstance(arr, np.ndarray):
        return arr.tolist()
    elif isinstance(arr, torch.Tensor):
        if arr.requires_grad:
            return arr.cpu().detach().numpy().tolist()
        return arr.cpu().numpy().tolist()
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
    if isinstance(arr, np.ndarray):
        return np.zeros_like(arr)
    elif isinstance(arr, torch.Tensor):
        return torch.zeros_like(arr)
    else:
        raise TypeError("Input must be a numpy array or torch tensor")
