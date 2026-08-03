from __future__ import annotations

from typing import Any
from numbers import Number, Real
from collections.abc import Callable, Sequence
import math

from ._backend import (
    numpy_array_types,
    numpy_bool_types,
    numpy_scalar_types,
    require_numpy,
    require_torch,
    torch_tensor_types,
)
from .types import ArrayLike

__all__ = [
    "is_numberlike",
    "is_array",
    "is_array_like",
    "all_real_and",
    "is_real",
    "is_real_positive",
    "is_real_nonnegative",
    "is_valid_array_mode",
]


def is_numberlike(x: Any) -> bool:
    """
    Check whether an object is a numeric or boolean scalar.

    Parameters
    ----------
    x : Any
        Object to test.

    Returns
    -------
    bool
        True if `x` is a supported numeric or boolean scalar; False otherwise.

    Notes
    -----
    This function is used for runtime validation at API boundaries.

    Currently accepted scalar types include:

      - Python numeric scalars (including ``bool``, ``int``, ``float``, ``complex``,
        and other ``numbers.Number`` implementations)
      - NumPy numeric scalars (subclasses of ``numpy.number``)
      - NumPy boolean scalar (``numpy.bool_``)

    Non-numeric NumPy scalar types (e.g. ``numpy.str_``, ``numpy.datetime64``,
    ``numpy.timedelta64``) are rejected. Support for additional scalar types may be
    added in the future.
    """
    return isinstance(x, Number) or isinstance(x, numpy_scalar_types)


def _is_numpy_numeric_or_bool_dtype(dtype: Any) -> bool:
    """
    Check whether a NumPy dtype is numeric or boolean.

    Parameters
    ----------
    dtype : numpy.dtype
        NumPy dtype to test.

    Returns
    -------
    bool
        True if the dtype is numeric or boolean; False otherwise.
    """
    np = require_numpy()
    return np.issubdtype(dtype, np.number) or np.issubdtype(dtype, np.bool_)


def _is_torch_numeric_or_bool_dtype(dtype: Any) -> bool:
    """
    Check whether a torch dtype is numeric or boolean.

    Parameters
    ----------
    dtype : torch.dtype
        Torch dtype to test.

    Returns
    -------
    bool
        True if the dtype is numeric or boolean; False otherwise.

    Notes
    -----
    Currently accepted torch dtypes include boolean, floating-point, complex, and
    integer types. Support for additional backends may be added in the future.
    """
    torch = require_torch()
    if dtype == torch.bool:
        return True
    if dtype.is_floating_point:
        return True
    if dtype.is_complex:
        return True
    return dtype in (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
        torch.uint16,
        torch.uint32,
        torch.uint64,
    )


def is_array(x: Any) -> bool:
    """
    Check whether an object is a supported rectangular array container with numeric or boolean dtype.

    Parameters
    ----------
    x : Any
        Object to test.

    Returns
    -------
    bool
        True if `x` is a supported array container and its dtype is numeric or
        boolean; False otherwise.

    Notes
    -----
    This function is intentionally strict: it only returns True for recognized
    array containers, not for Python sequences.

    Currently supported array containers include NumPy ndarrays and torch Tensors.
    Support for additional backends may be added in the future.
    """
    if isinstance(x, numpy_array_types):
        return _is_numpy_numeric_or_bool_dtype(x.dtype)
    if isinstance(x, torch_tensor_types):
        return _is_torch_numeric_or_bool_dtype(x.dtype)
    return False


def _is_string_like(x: Any) -> bool:
    return isinstance(x, (str, bytes, bytearray))


def is_array_like(x: Any) -> bool:
    """
    Check whether an object is array-like.

    Parameters
    ----------
    x : Any
        Object to test.

    Returns
    -------
    bool
        True if `x` is array-like; False otherwise.

    Notes
    -----
    An object is considered array-like if it can be unambiguously converted to
    a rectangular numeric or boolean array container.

    In particular:

      - Supported array containers are array-like.
      - Numeric or boolean scalars are array-like.
      - Python sequences are array-like if all elements are array-like and their
        shapes are compatible for stacking into a rectangular array.
      - String-like objects (``str``, ``bytes``, ``bytearray``) are not considered
        array-like.

    This function is intended for runtime validation at API boundaries.
    """
    from .shape import infer_shape

    return infer_shape(x) is not None


def all_real_and(x: ArrayLike, predicate: Callable[[Any], Any]) -> bool:
    """
    Check whether all real values contained in an array-like object satisfy
    a given predicate.

    The function returns ``True`` if and only if every scalar value contained
    in ``x`` is a finite real number and ``predicate`` evaluates to ``True``
    on that value. Boolean and complex values are rejected and cause the
    function to return ``False``.

    Supported input types include real numeric scalars, NumPy arrays,
    PyTorch tensors, and recursive Python sequences thereof. String-like
    objects are not considered array-like.

    Parameters
    ----------
    x : ArrayLike
        Array-like object to test.
    predicate : callable
        A predicate applied elementwise to the values contained in ``x``.

        The predicate must be compatible with the input type:

        - For scalar inputs, it must return a truthy or falsy value.
        - For NumPy arrays, it must return a boolean array broadcastable
          to the shape of ``x``.
        - For PyTorch tensors, it must return a boolean tensor broadcastable
          to the shape of ``x``.

    Returns
    -------
    bool
        ``True`` if all contained values are finite real numbers and satisfy
        ``predicate``; ``False`` otherwise.

    Raises
    ------
    TypeError
        If ``x`` is not array-like.
    ValueError
        If a self-referential sequence is encountered.

    Notes
    -----
    - Boolean values are not considered real and always result in ``False``.
    - Complex-valued arrays or tensors result in ``False``.
    - Integer arrays and tensors are treated as finite.
    - Python sequences are traversed recursively.
    """
    if not is_array_like(x):
        raise TypeError(f"Object of type {type(x).__name__} is not array-like")

    # ---- scalar ----
    if isinstance(x, bool) or isinstance(x, numpy_bool_types):
        return False
    if isinstance(x, Real):
        return bool(math.isfinite(x) and predicate(x))

    # ---- NumPy ----
    if isinstance(x, numpy_array_types):
        np = require_numpy()
        if np.iscomplexobj(x) or x.dtype == np.bool_:
            return False
        return bool(np.all(np.isfinite(x) & predicate(x)))

    # ---- Torch ----
    if isinstance(x, torch_tensor_types):
        torch = require_torch()
        if x.is_complex() or x.dtype == torch.bool:
            return False

        if x.is_floating_point():
            finite = torch.isfinite(x)
        else:
            # integers are always finite
            finite = torch.ones_like(x, dtype=torch.bool)

        return bool((finite & predicate(x)).all().item())

    # ---- recursive sequences ----
    if isinstance(x, Sequence) and not isinstance(x, (str, bytes, bytearray)):
        seen: set[int] = set()
        return _all_real_and_seq(x, predicate, seen)

    return False


def _all_real_and_seq(
    x: Sequence[Any],
    predicate: Callable[[Any], Any],
    seen: set[int],
) -> bool:
    oid = id(x)
    if oid in seen:
        raise ValueError("Self-referential sequence encountered")
    seen.add(oid)
    return all(all_real_and(item, predicate) for item in x)


def is_real(x: ArrayLike) -> bool:
    """
    Check whether all values contained in an array-like object are finite reals.

    The function returns ``True`` if and only if every scalar value contained in
    ``x`` is a finite real number. Boolean and complex values are rejected and
    cause the function to return ``False``.

    Parameters
    ----------
    x : ArrayLike
        Array-like object to test.

    Returns
    -------
    bool
        ``True`` if all contained values are finite real numbers; ``False``
        otherwise.

    Raises
    ------
    TypeError
        If ``x`` is not array-like.
    ValueError
        If a self-referential sequence is encountered.

    Notes
    -----
    - Python sequences are traversed recursively.
    - This is equivalent to ``all_real_and(x, lambda _: True)``.
    """
    return all_real_and(x, lambda _: True)


def is_real_positive(x: ArrayLike) -> bool:
    """
    Check whether all values contained in an array-like object are finite positive reals.

    The function returns ``True`` if and only if every scalar value contained in
    ``x`` is a finite real number greater than zero. Boolean and complex values
    are rejected and cause the function to return ``False``.

    Parameters
    ----------
    x : ArrayLike
        Array-like object to test.

    Returns
    -------
    bool
        ``True`` if all contained values are finite positive real numbers;
        ``False`` otherwise.

    Raises
    ------
    TypeError
        If ``x`` is not array-like.
    ValueError
        If a self-referential sequence is encountered.

    Notes
    -----
    - Python sequences are traversed recursively.
    - This is equivalent to ``all_real_and(x, lambda v: v > 0)``.
    """
    return all_real_and(x, lambda v: v > 0)


def is_real_nonnegative(x: ArrayLike) -> bool:
    """
    Check whether all values contained in an array-like object are finite nonnegative reals.

    The function returns ``True`` if and only if every scalar value contained in
    ``x`` is a finite real number greater than or equal to zero. Boolean and
    complex values are rejected and cause the function to return ``False``.

    Parameters
    ----------
    x : ArrayLike
        Array-like object to test.

    Returns
    -------
    bool
        ``True`` if all contained values are finite nonnegative real numbers;
        ``False`` otherwise.

    Raises
    ------
    TypeError
        If ``x`` is not array-like.
    ValueError
        If a self-referential sequence is encountered.

    Notes
    -----
    - Python sequences are traversed recursively.
    - This is equivalent to ``all_real_and(x, lambda v: v >= 0)``.
    """
    return all_real_and(x, lambda v: v >= 0)


def is_valid_array_mode(mode: str) -> bool:
    """
    Check if the provided mode is valid.

    Parameters
    ----------
    mode : str
        Mode to check.

    Returns
    -------
    bool
        True if the mode is valid; False otherwise.

    Notes
    -----
    Valid modes are "numpy", "torch", and "torch_cuda".
    """
    return mode in ["numpy", "torch", "torch_cuda"]
