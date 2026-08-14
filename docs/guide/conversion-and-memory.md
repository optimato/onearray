# Conversion and memory

OneArray's numerical operations follow the backend of their input. Conversion
utilities are how Python values become array containers and how existing arrays
move between NumPy and PyTorch.

Conversion is always explicit. It may select a different backend, move data to
another device, or create storage that is independent from the input.

!!! note "Conversion guarantees"
    The guarantees in this guide apply to dense NumPy arrays and dense strided
    PyTorch tensors. Specialized PyTorch layouts and views that NumPy cannot
    expose directly—such as sparse tensors and unresolved conjugate or
    negative-bit views—may raise a backend error during conversion.

## Choosing a backend

The `mode` argument selects the result's backend and device:

| Mode | Result | Device |
| --- | --- | --- |
| `"numpy"` | NumPy `ndarray` | CPU |
| `"torch"` | PyTorch `Tensor` | CPU |
| `"torch_cuda"` | PyTorch `Tensor` | CUDA |

For example:

```python
import onearray as oa

x_np = oa.array([1, 2, 3], mode="numpy")
x_torch = oa.array([1, 2, 3], mode="torch")
x_cuda = oa.array([1, 2, 3], mode="torch_cuda")
```

OneArray does not silently fall back to another backend or device. Requesting a
backend that is not installed is an error, as is requesting CUDA when it is not
available.

## Independent conversion with `array`

`array(value, mode)` converts any valid array-like value to the selected
backend. A scalar becomes a zero-dimensional array container.

The result owns its storage independently from the input. Changes to the result
do not affect the input, and changes to the input do not affect the result:

```python
import numpy as np

source = np.array([1, 2, 3])
result = oa.array(source, mode="torch")

result[0] = 10
assert source[0] == 1
```

## Shared memory

Two arrays share memory when they refer to the same underlying data. A change
made through one array can then be visible through the other:

```python
source = np.array([1, 2, 3])
view = source.view()

view[0] = 10
assert source[0] == 10
```

Sharing avoids an allocation and can be useful for large arrays, but it also
couples the two objects. Code that requires isolation should use `array`.

## Shared conversion with `asarray`

`asarray(value, mode)` accepts the same inputs and modes as `array`. The
difference is that it reuses the input object or shares its storage when that
can be done safely. When sharing is not possible, it creates a copy.

```python
source = np.array([1, 2, 3])
result = oa.asarray(source, mode="torch")

result[0] = 10
assert source[0] == 10
```

Callers must assume that changes can be visible through both the input and the
result. Returning the original object unchanged is also permitted when its
backend and device already match the requested mode.

## When `asarray` shares or copies

The expected behavior depends on both the input and the requested mode:

| Input | `"numpy"` | `"torch"` | `"torch_cuda"` |
| --- | --- | --- | --- |
| Writable NumPy array | Same object | Share if compatible; otherwise copy | Copy |
| CPU PyTorch tensor | Share if compatible; otherwise copy | Same object | Copy |
| CUDA PyTorch tensor | Copy | Copy | Same object |
| Read-only NumPy array | Same object | Copy | Copy |
| Python scalar or sequence | Allocate | Allocate | Allocate |

**Same object** means that `asarray` returns the input itself. **Share** means
that the result is a different kind of container backed by the same memory.
**Copy** and **allocate** produce independent storage.

A device transfer always requires new storage. Converting a Python scalar or
sequence also requires allocation because there is no existing array storage to
share.

Some dtypes and memory layouts cannot be shared between NumPy and PyTorch. In
those cases, `asarray` copies rather than failing solely because sharing is
unavailable. In particular, NumPy arrays with negative strides, such as a
reversed view created by `source[::-1]`, are copied when converted to PyTorch.

## Read-only NumPy arrays

PyTorch cannot reliably preserve NumPy's read-only restriction on shared
storage. `asarray` will therefore copy a read-only NumPy array when targeting
PyTorch. This prevents a writable tensor from modifying storage that NumPy
presents as read-only.

## PyTorch autograd

When a PyTorch tensor is already on the requested device, `asarray` may return
it unchanged. Its autograd state is preserved.

NumPy cannot participate in a PyTorch computation graph. Converting a CPU
tensor to NumPy therefore detaches the resulting array from autograd. The NumPy
array may still share the tensor's storage.

Converting a CUDA tensor to NumPy first requires a CPU transfer, so the result
has separate storage.

## Dtypes

When converting an existing array container, OneArray preserves its dtype if
the target backend supports it. If the target backend cannot represent the
dtype, conversion raises an error. OneArray never changes dtype merely to make
memory sharing possible.

Python scalars and sequences do not have a backend dtype. NumPy or PyTorch
applies its normal dtype inference rules when converting them.

See [Arrays and backends](arrays-and-backends.md) for the distinction between
array-like values and array containers.
