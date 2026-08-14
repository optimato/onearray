"""
OneArray: Backend-agnostic array utilities and numerical primitives.

This package provides a unified interface for working with array containers 
across multiple numerical backends, while enforcing a consistent set of shape 
and dtype invariants.

The following concepts are central to OneArray:

- **array container**: a rectangular array object holding numeric or boolean data,
  provided by a supported backend.
- **array-like**: any object that can be unambiguously converted into an array
  container, such as numeric scalars, existing array containers, or nested Python
  sequences with compatible shapes.

OneArray exposes three main categories of functionality:

1. **Normalization and conversion**
   Utilities to convert array-like inputs into array containers, to infer shapes,
   and to move data between supported backends while preserving semantics.

2. **Backend-agnostic numerical operations**
   A collection of mathematical, logical, and signal-processing functions
   (e.g. elementwise operations, FFTs) that operate uniformly on array containers,
   dispatching to the appropriate backend implementation while preserving a
   consistent user-facing API.

3. **Shape-aware helpers**
   Functions that reason about array structure (such as shape inference and
   validation) without performing allocation or computation.

Functions in this module are intended to be used at API boundaries and in
numerical kernels alike. Once an input has been normalized through this layer,
downstream code may assume it operates on well-formed array containers with
predictable shape and dtype behavior, independent of the underlying backend.
"""

from .checks import (
    is_numeric,
    is_positive_numeric,
    is_array,
    is_array_of,
    is_numeric_array,
    is_positive_numeric_array,
    array_equal,
    is_valid_array_mode,
)

from .shape import dim, len  # noqa: F401

from .convert import array, asarray, add_axis, to_list, zeros_like

from .math import sum, exp, abs

from .fourier import fft, ifft, fftfreq, fftshift, ifftshift

__all__ = (
    [
        "is_numeric",
        "is_positive_numeric",
        "is_array",
        "is_array_of",
        "is_numeric_array",
        "is_positive_numeric_array",
        "array_equal",
        "is_valid_array_mode",
    ]
    + [
        "dim"  # len has been left out to avoid conflict with built-in len()
        # it can still be accessed via from array.shape import len
    ]
    + ["array", "asarray", "add_axis", "to_list", "zeros_like"]
    + ["sum", "exp", "abs"]
    + ["fft", "ifft", "fftfreq", "fftshift", "ifftshift"]
)
