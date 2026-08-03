# src/onearray/types.py
"""
Type definitions for OneArray.

This module defines core type aliases used throughout the OneArray library,
including numeric types, array-like inputs, and backend modes.

These type aliases facilitate static type checking and improve code clarity
by providing meaningful names for commonly used type combinations.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, TypeAlias, TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt
    import torch

__all__ = [
    "NumberLike",
    "Array",
    "ArrayLike",
    "ArrayMode",
]

# ---------------------------------------------------------------------------
# Core numeric & array-like types
# ---------------------------------------------------------------------------

# Python and NumPy scalar numbers (real, complex, boolean) but NOT np.str_, np.datetime64, etc.
if TYPE_CHECKING:
    NumberLike: TypeAlias = bool | int | float | complex | np.number | np.bool_
    """
    Type alias for scalar numeric types.

    Includes Python built-in types and NumPy scalar types.
    """
else:
    NumberLike: TypeAlias = bool | int | float | complex | Any
    """
    Type alias for scalar numeric types.

    Includes Python built-in types and NumPy scalar types.
    """

# Normalized array type used internally by onearray
if TYPE_CHECKING:
    Array: TypeAlias = npt.NDArray[np.number | np.bool_] | torch.Tensor
    """
    Type alias for array container types.

    Includes NumPy ndarrays with numeric or boolean dtypes, and PyTorch tensors.

    Notes
    -----
    - This type does not include other array-like types such as lists or tuples.
    - This type does not include NumPy arrays with non-numeric dtypes such as strings or datetime64.
    - This type does not include scalar numeric types; see `NumberLike` for that.
    """
else:
    Array: TypeAlias = Any
    """
    Type alias for array container types.

    Includes NumPy ndarrays with numeric or boolean dtypes, and PyTorch tensors.

    Notes
    -----
    - This type does not include other array-like types such as lists or tuples.
    - This type does not include NumPy arrays with non-numeric dtypes such as strings or datetime64.
    - This type does not include scalar numeric types; see `NumberLike` for that.
    """

# User-facing "anything that can be turned into an Array".
# This is intentionally broad; runtime checks may still reject invalid values.
ArrayLike: TypeAlias = NumberLike | Array | Sequence["ArrayLike"]
"""
Type alias for array-like inputs.

Includes scalar numeric types, array containers, and nested sequences thereof.
This type represents any input that can be converted into a valid array
container by onearray utilities.

Notes
-----
- This type is intentionally broad to accommodate various input formats.
- Runtime validation may still reject certain inputs that cannot be converted
  into valid arrays (e.g., sequences with inconsistent shapes).
"""

# Backend choice for array conversion
ArrayMode: TypeAlias = Literal["numpy", "torch", "torch_cuda"]
"""
Type alias for specifying array backend modes.

Indicates which array library to use when converting array-like inputs.
Supported values are:
- `"numpy"`: Use NumPy arrays.
- `"torch"`: Use PyTorch tensors on CPU.
- `"torch_cuda"`: Use PyTorch tensors on CUDA-enabled GPU.
"""
