from __future__ import annotations

from .errors import ArrayError

try:
    import numpy as np

    HAS_NUMPY = True
    numpy_array_types = (np.ndarray,)
    numpy_scalar_types = (np.number, np.bool_)
    numpy_bool_types = (np.bool_,)
except ModuleNotFoundError:  # pragma: no cover - depends on environment
    np = None
    HAS_NUMPY = False
    numpy_array_types = ()
    numpy_scalar_types = ()
    numpy_bool_types = ()

try:
    import torch

    HAS_TORCH = True
    torch_tensor_types = (torch.Tensor,)
except ModuleNotFoundError:  # pragma: no cover - depends on environment
    torch = None
    HAS_TORCH = False
    torch_tensor_types = ()


def numpy_available() -> bool:
    return HAS_NUMPY


def torch_available() -> bool:
    return HAS_TORCH


def require_numpy():
    if not HAS_NUMPY:
        raise ModuleNotFoundError(ArrayError.NUMPY_NOT_AVAILABLE.value)
    return np


def require_torch():
    if not HAS_TORCH:
        raise ModuleNotFoundError(ArrayError.TORCH_NOT_AVAILABLE.value)
    return torch
