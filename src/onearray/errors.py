from enum import Enum


class ArrayError(Enum):
    """
    Enum for errors related to array operations.
    """

    INVALID_ARRAY_TYPE = "arr must be a list, NumPy array, or torch tensor."
    INVALID_ARRAY_MODE = "mode must be either 'numpy', 'torch', or 'torch_cuda'."
    CUDA_NOT_AVAILABLE = "'torch_cuda' mode was requested but CUDA is not available. Use 'torch' mode instead."
    NUMPY_NOT_AVAILABLE = "NumPy is required for this operation. Install numpy to use NumPy-backed functionality."
    TORCH_NOT_AVAILABLE = "PyTorch is required for this operation. Install torch to use PyTorch-backed functionality."
