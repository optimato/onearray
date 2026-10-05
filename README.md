# OneArray

[![Code style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Lint](https://github.com/optimato/onearray/actions/workflows/lint.yml/badge.svg)](https://github.com/optimato/onearray/actions/workflows/lint.yml)
[![Tests](https://github.com/optimato/onearray/actions/workflows/tests.yml/badge.svg)](https://github.com/optimato/onearray/actions/workflows/tests.yml)
[![Documentation](https://img.shields.io/badge/docs-online-blue.svg)](https://optimato.github.io/onearray/)
[![Python](https://img.shields.io/badge/python-%E2%89%A53.10-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

Backend-agnostic array utilities with a familiar NumPy-style API.

OneArray lets you write numerical code once and use it with either NumPy arrays
or PyTorch tensors. Operations dispatch from the input type and return values in
the same backend—there is no wrapper array class to learn.

## Installation

OneArray requires Python 3.10 or newer. Install it with the backend you need:

```bash
pip install "onearray[numpy]"
pip install "onearray[torch]"
```

To work on the project locally:

```bash
git clone https://github.com/optimato/onearray.git
cd onearray
pip install -e ".[test]"
```

## Quick start

```python
import onearray as oa

x_np = oa.array([1.0, 2.0, 3.0], mode="numpy")
x_torch = oa.array([1.0, 2.0, 3.0], mode="torch")

oa.exp(x_np)     # numpy.ndarray
oa.exp(x_torch)  # torch.Tensor
```

Numerical operations do not silently move data between backends. Use `array`
or `asarray` when you explicitly want to create or convert an array:

```python
x = oa.asarray([1, 2, 3], mode="numpy")
y = oa.fft(x)
values = oa.to_list(y)
```

## Features

- NumPy and PyTorch support, including CUDA tensors where PyTorch supports the operation
- Explicit array creation and conversion with `array` and `asarray`
- Backend-preserving math helpers: `sum`, `exp`, and `abs`
- Fourier helpers: `fft`, `ifft`, `fftfreq`, `fftshift`, and `ifftshift`
- Shape, axis, list-conversion, and zero-allocation helpers
- Validation for arrays, scalars, and rectangular nested sequences

## Documentation

Read the full documentation at [optimato.github.io/onearray](https://optimato.github.io/onearray/).

## Running tests

```bash
pytest
```

## License

OneArray is released under the [MIT License](LICENSE).
