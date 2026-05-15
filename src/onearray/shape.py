from __future__ import annotations

from typing import Any
from collections.abc import Sequence
import builtins

from ._backend import numpy_array_types, torch_tensor_types
from .validation import _is_string_like, is_array, is_numberlike
from .errors import ArrayError


__all__ = ["dim", "infer_shape"]


def infer_shape(x: Any) -> tuple[int, ...] | None:
    """
    Infer the rectangular shape of an array-like object.

    Parameters
    ----------
    x : Any
        Object to test.

    Returns
    -------
    tuple[int, ...] | None
        The inferred shape if `x` is array-like, i.e. if it can be unambiguously
        converted to a rectangular array container. Returns ``None`` if
        the shape cannot be determined unambiguously.

    Notes
    -----
    The following rules define array-likeness for the purpose of shape inference:

    - Numeric or boolean scalars are considered array-like and have shape ``()``.
    - Supported array containers contribute their existing shape.
    - A Python sequence is array-like if all of its elements are array-like and
      have the same inferred shape; in this case the resulting shape is
      ``(len(sequence),) + child_shape``.
    - String-like objects (``str``, ``bytes``, ``bytearray``) are not considered
      array-like and are rejected.

    This function performs *structural* validation only. It does not allocate
    arrays or perform any data conversion.
    """
    return _infer_shape_recursive(x, seen=set())


def _infer_shape_recursive(x: Any, seen: set[int]) -> tuple[int, ...] | None:
    # Reject string-like sequences early
    if _is_string_like(x):
        return None

    # Scalar
    if is_numberlike(x):
        return ()

    # Array container (numeric/bool dtype already enforced by is_array)
    if is_array(x):
        return tuple(x.shape)  # np.ndarray / torch.Tensor

    # Python sequence: must be stackable with equal child shapes
    if isinstance(x, Sequence):
        oid = id(x)
        if oid in seen:
            return None
        seen.add(oid)

        n = builtins.len(x)
        if n == 0:
            # Policy choice: treat empty sequence as an empty 1D array
            return (0,)

        first_shape: tuple[int, ...] | None = None
        for item in x:
            shp = _infer_shape_recursive(item, seen)
            if shp is None:
                return None
            if first_shape is None:
                first_shape = shp
            elif shp != first_shape:
                return None

        assert (
            first_shape is not None
        )  # n > 0 guarantees this, but helps the type checker

        # first_shape is guaranteed to be a tuple here
        return (n,) + first_shape

    return None


def dim(arr):
    """
    Get the number of dimensions of the input array/tensor.
    If the input is a nested array, return the number of dimensions of the innermost arrays plus one.

    Parameters
    ----------
    arr : list, np.ndarray, or torch.Tensor
        The input array or tensor.

    Returns
    -------
    int
        The number of dimensions of the input array or tensor.

    Raises
    ------
    TypeError
        If the input is not a list, numpy array, or torch tensor, or if the nested array structure is invalid.
    """
    if isinstance(arr, numpy_array_types):
        return arr.ndim
    elif isinstance(arr, torch_tensor_types):
        return arr.ndim
    elif isinstance(arr, list):
        shape = _infer_shape_recursive(arr, seen=set())
        if shape is None:
            raise TypeError(ArrayError.INVALID_ARRAY_TYPE.value)
        return len(shape)
    else:
        raise TypeError(ArrayError.INVALID_ARRAY_TYPE.value)


def len(arr):
    """
    Get the length of the input array/tensor
    If the input is a nested array, return the length of the outermost array.
    If the input is a 1D array, return its length.
    If the input is a multi-dimensional array, return the size of the first dimension.

    Parameters
    ----------
    arr : list, np.ndarray, or torch.Tensor
        The input array or tensor.

    Returns
    -------
    int
        The length of the input array or tensor.

    Raises
    ------
    TypeError
        If the input is not a list, numpy array, or torch tensor.
    """
    if isinstance(arr, numpy_array_types):
        return arr.shape[0]
    elif isinstance(arr, torch_tensor_types):
        return arr.shape[0]
    elif isinstance(arr, list):
        return builtins.len(arr)
    else:
        raise TypeError(ArrayError.INVALID_ARRAY_TYPE.value)

