# Arrays and backends

OneArray works directly with NumPy arrays and PyTorch tensors. It does not wrap
them in a new array class.

NumPy and PyTorch are the **backends** currently supported by OneArray. In other
words, they are the libraries that store the data and perform the calculations.
OneArray selects the appropriate implementation from the value it receives:

```python
import numpy as np
import torch

import onearray as oa


x_np = np.array([1.0, 2.0, 3.0])
x_torch = torch.tensor([1.0, 2.0, 3.0])

y_np = oa.exp(x_np)
y_torch = oa.exp(x_torch)
```

`y_np` is a NumPy array and `y_torch` is a PyTorch tensor. The same OneArray
operation handles both inputs; the input determines which implementation runs.

## Array containers

OneArray calls NumPy arrays and PyTorch tensors **array containers**. More
precisely, an array container is either:

- a NumPy `ndarray` with a numeric or boolean dtype; or
- a PyTorch `Tensor` with a numeric or boolean dtype.

The container carries information that matters during a calculation: its
backend, shape, dtype and, for PyTorch tensors, device. OneArray preserves that
backend unless you explicitly call a conversion utility.

An array container can be zero-dimensional. For example, `np.array(3.0)` and
`torch.tensor(3.0)` are array containers even though their shape is `()`.

## Array-like values

Python values do not need to be array containers to be understood by OneArray.
A value is **array-like** when it can be converted unambiguously into a
rectangular numeric or boolean array.

Array-like values include:

- numeric and boolean scalars;
- existing array containers; and
- nested Python sequences whose elements have compatible shapes.

For example:

```python
3.0
[1, 2, 3]
((1, 2), (3, 4))
```

Supported scalars are Python `bool`, `int`, `float`, and `complex`, plus NumPy
numeric and boolean scalars. Convert other numeric types, such as `Fraction`
and `Decimal`, explicitly before passing them to OneArray. Datetime and
timedelta values are not supported.

A scalar such as `3.0` is number-like and has shape `()`, but it is not an array
container until it is converted. Likewise, a Python list can describe an array
without belonging to NumPy or PyTorch.

Some values cannot describe a rectangular numeric array. A ragged sequence, for
example, has no single shape:

```python
[[1, 2], [3]]
```

Strings, bytes, non-numeric backend arrays, and sequences that are ragged,
recursive, or contain unsupported values are not array-like.

## Shape inference

OneArray can determine the shape of an array-like value without allocating an
array or choosing a backend. Shape inference follows four rules:

- A scalar has shape `()`.
- An array container contributes its existing shape.
- A sequence of length `n` whose children have shape `S` has shape
  `(n,) + S`.
- An empty sequence has shape `(0,)`.

For example, `[[1, 2], [3, 4]]` has shape `(2, 2)`. Its two children both have
shape `(2,)`, so they form a rectangular two-dimensional value.

An empty sequence is a valid one-dimensional array-like value. Because it has
no elements, its dtype cannot be inferred from its contents. The selected
backend applies its normal default when the value is converted.

## Validation without conversion

Validation utilities can inspect array-like values directly. They do not need
to create a NumPy array or PyTorch tensor first, so validation does not select a
backend or allocate an array container.

Predicates that ask whether **all** elements have a property use the usual
universal interpretation. An empty collection has no element that violates the
property, so it satisfies the predicate.

In particular, empty values satisfy the predicates “all values are real,” “all
values are positive,” and “all values are nonnegative.” This behavior is the
same for empty Python sequences, NumPy arrays, and PyTorch tensors.

## Why numerical operations require containers

Mathematical and Fourier operations accept array containers rather than general
array-like values. A Python scalar or list contains no backend information, so
OneArray cannot know which implementation the caller wants:

```python
# Invalid: a Python list has no numerical backend.
oa.fft([1.0, 0.0, 0.0, 0.0])

# Valid: the array container determines the backend.
x = oa.array([1.0, 0.0, 0.0, 0.0], mode="numpy")
oa.fft(x)
```

Numerical operations never convert their inputs or choose a backend on the
caller's behalf. This keeps allocation and data movement explicit while allowing
the numerical code itself to remain backend-independent.

## Frequency grids

`fftfreq` creates frequency coordinates for a selected backend. With a scalar
window length, it returns one frequency array or tensor. With a tuple shape, it
returns the native backend `meshgrid` result. OneArray preserves that native
container type rather than normalizing it.

```python
numpy_grid = oa.fftfreq((3, 4), mode="numpy")
torch_grid = oa.fftfreq((3, 4), mode="torch")
```

See [Conversion and memory](conversion-and-memory.md) for the rules governing
backend conversion, copying, and shared storage.
