from typing import Any
import numpy as np
import torch
from collections.abc import Sequence
from numbers import Number
import warnings

__all__ = [
    "is_numeric",
    "is_positive_numeric",
    "is_array",
    "is_array_of",
    "is_numeric_array",
    "is_positive_numeric_array",
    "array_equal",
    "is_valid_array_mode",
]


def is_numeric(x: Any) -> bool:
    """Helper function to check if input is a numeric value"""
    warnings.warn(
        "is_numeric is deprecated and will be removed in a future release. "
        "Please use is_numberlike from onearray.validation instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    from .shape import dim

    return isinstance(x, Number) or (is_numeric_array(x) and dim(x) == 0)


def is_positive_numeric(x: Any) -> bool:
    """
    Helper function to check if input is a non-negative numeric value

    Notes
    -----
    - Despite the name, this function checks for non-negativity (>= 0), not strictly positive (> 0).
    - This function is deprecated; use `is_real_nonnegative` from `onearray.validation` instead.
    """
    warnings.warn(
        "is_positive_numeric is deprecated and will be removed in a future release. "
        "Please use is_real_nonnegative from onearray.validation instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return is_numeric(x) and x >= 0


def is_array(arr: Any) -> bool:
    """Check if the input is a list, numpy array, or torch tensor"""
    warnings.warn(
        "is_array is deprecated and will be removed in a future release. "
        "Please use is_array or is_array_like from onearray.validation instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return isinstance(arr, (list, np.ndarray, torch.Tensor))


def is_array_of(arr: Any, dtype: type) -> bool:
    """Check if the input is an array of a specific type"""
    if isinstance(arr, np.ndarray):
        return np.issubdtype(arr.dtype, dtype)
    elif isinstance(arr, torch.Tensor):
        # Handle torch tensor types
        if dtype in (int, np.int32, np.int64):
            return arr.dtype in (torch.int32, torch.int64)
        elif dtype in (float, np.float32, np.float64):
            return arr.dtype in (torch.float32, torch.float64)
        else:
            return arr.dtype == dtype
    elif isinstance(arr, list):
        return all(isinstance(i, dtype) for i in arr)
    else:
        return False


def is_numeric_array(arr: Any) -> bool:
    """Helper function to check if input is a numeric array/tensor"""
    warnings.warn(
        "is_numeric_array is deprecated and will be removed in a future release. "
        "Please use is_array or is_array_like from onearray.validation instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    if isinstance(arr, list):
        return all(isinstance(x, Number) for x in arr) or all(
            isinstance(x, list) and is_numeric_array(x) for x in arr
        )
    elif isinstance(arr, np.ndarray):
        return np.issubdtype(arr.dtype, np.number)
    elif isinstance(arr, torch.Tensor):
        return torch.is_floating_point(arr) or arr.dtype in (torch.int32, torch.int64)
    else:
        return False


def is_positive_numeric_array(arr: Any) -> bool:
    """
    Helper function to check if input is a non-negative numeric array/tensor

    Notes
    -----
    - Despite the name, this function checks for non-negativity (>= 0), not strictly positive (> 0).
    - This function is deprecated; use `is_real_nonnegative` from `onearray.validation` instead.
    """
    warnings.warn(
        "is_positive_numeric_array is deprecated and will be removed in a future release. "
        "Please use is_real_nonnegative from onearray.validation instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    if not is_numeric_array(arr):
        return False
    elif isinstance(arr, list):
        if all(isinstance(x, list) for x in arr):
            return all(is_positive_numeric_array(x) for x in arr) or all(
                is_positive_numeric(x) for x in arr
            )
        elif all(isinstance(x, (int, float)) for x in arr):
            return all(is_positive_numeric(x) for x in arr)
        else:
            return False
    if isinstance(arr, np.ndarray):
        return bool(np.all(arr >= 0))
    elif isinstance(arr, torch.Tensor):
        return bool(torch.all(arr >= 0).item())
    else:
        return False


def array_equal(arr1, arr2) -> bool:
    """Helper function to compare arrays/tensors regardless of their type"""
    # Convert lists to numpy arrays first
    if isinstance(arr1, Sequence) and not isinstance(arr1, (np.ndarray, torch.Tensor)):
        arr1 = np.array(arr1)
    if isinstance(arr2, Sequence) and not isinstance(arr2, (np.ndarray, torch.Tensor)):
        arr2 = np.array(arr2)

    if isinstance(arr1, torch.Tensor) and isinstance(arr2, torch.Tensor):
        return torch.equal(arr1, arr2)
    elif isinstance(arr1, np.ndarray) and isinstance(arr2, np.ndarray):
        return np.array_equal(arr1, arr2)
    elif isinstance(arr1, torch.Tensor):
        return torch.equal(arr1, torch.as_tensor(arr2))
    elif isinstance(arr2, torch.Tensor):
        return torch.equal(torch.as_tensor(arr1), arr2)
    else:
        return np.array_equal(np.array(arr1), np.array(arr2))


def is_valid_array_mode(mode: str) -> bool:
    """Check if the provided mode is valid"""
    warnings.warn(
        "is_valid_array_mode is deprecated and will be removed in a future release. "
        "Please use is_valid_array_mode from onearray.validation instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return mode in ["numpy", "torch", "torch_cuda"]
