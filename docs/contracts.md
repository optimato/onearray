# OneArray API contracts

Status: draft for design review

This document defines the intended public behavior of OneArray. It is the
reference from which public documentation and tests should be derived. It does
not treat every detail of the current implementation as intentional behavior.

The words **must**, **should**, and **may** identify required behavior,
recommended behavior, and permitted behavior respectively. Items labelled
**Decision required** are not yet part of the contract.

## Sources and precedence

The initial vocabulary comes from `src/onearray/types.py`. Runtime semantics are
cross-checked against `src/onearray/validation.py` and
`src/onearray/shape.py`.

When sources disagree, use this order while drafting the contract:

1. An explicit decision recorded in this document.
2. The public concepts described by `types.py` and `validation.py`.
3. Public function documentation.
4. Current implementation behavior.

`src/onearray/checks.py` contains deprecated compatibility functions. Their
behavior is not authoritative for the new vocabulary, even though they are
currently re-exported by `onearray.__init__`.

## Core vocabulary

### Number-like

A **number-like** value is a scalar numeric or boolean value.

It currently includes:

- Python `bool`, `int`, `float`, and `complex` values;
- other implementations of `numbers.Number`;
- NumPy numeric scalar values;
- NumPy boolean scalar values.

It excludes string, bytes, datetime, timedelta, and other non-numeric scalar
types.

This concept corresponds to the static `NumberLike` alias and the runtime
`validation.is_numberlike` predicate.

### Array container

An **array container** is one of the following:

- a NumPy `ndarray` with a numeric or boolean dtype;
- a PyTorch `Tensor` with a numeric or boolean dtype.

An array container may be zero-dimensional. Python sequences and scalar values
are not array containers.

This concept corresponds to the static `Array` alias and the runtime
`validation.is_array` predicate.

### Array-like

An **array-like** value is a value that has one unambiguous rectangular numeric
or boolean shape. It is one of:

- a number-like scalar;
- an array container;
- a non-string Python sequence whose elements are array-like and all have the
  same shape.

Strings, `bytes`, and `bytearray` are not array-like. Ragged sequences are not
array-like. A self-referential sequence is not array-like.

The shape of a scalar is `()`. The shape of a non-empty sequence is its length
prepended to the common child shape. The current structural policy assigns an
empty sequence the shape `(0,)`.

This concept corresponds to the static `ArrayLike` alias, the runtime
`validation.is_array_like` predicate, and `shape.infer_shape`.

### Backend mode

A **backend mode** selects a target array implementation and device:

| Mode | Backend | Device |
| --- | --- | --- |
| `"numpy"` | NumPy | CPU |
| `"torch"` | PyTorch | CPU |
| `"torch_cuda"` | PyTorch | CUDA |

This concept corresponds to the static `ArrayMode` alias and the runtime
`validation.is_valid_array_mode` predicate.

### Shape and dimension

A **shape** is a tuple of non-negative integer extents. A **dimension** is the
number of entries in that tuple.

- A scalar has shape `()` and dimension `0`.
- An empty sequence has shape `(0,)` and dimension `1` under the current policy.
- A sequence of `n` children with common shape `S` has shape `(n,) + S`.
- An array container contributes its existing backend shape.

## Public API surface

The package root exports the following functions:

- Shape: `dim`.
- Conversion: `array`, `asarray`, `add_axis`, `to_list`, and `zeros_like`.
- Mathematics: `sum`, `exp`, and `abs`.
- Fourier: `fft`, `ifft`, `fftfreq`, `fftshift`, and `ifftshift`.

Validation functions are available from `onearray.validation`. `infer_shape` is
available from `onearray.shape`; neither module's API is re-exported from the
package root.

Deprecated checks may remain temporarily as compatibility shims, but supported
OneArray code must never call them. Internal code must use the new validation
API directly. The version and mechanism by which deprecated checks are removed
remain to be decided.

## Validation contracts

### `is_numberlike(value)`

Returns `True` exactly when `value` satisfies the number-like definition. It
does not raise merely because an optional backend is unavailable.

### `is_array(value)`

Returns `True` exactly when `value` is a supported array container with a
numeric or boolean dtype. It returns `False` for scalars and Python sequences.

This is intentionally different from the deprecated `checks.is_array`, which
also accepts lists without validating rectangularity or dtype.

### `is_array_like(value)`

Returns `True` exactly when `infer_shape(value)` returns a shape. It returns
`False` for invalid inputs rather than raising for ordinary invalid values.

### Real-value predicates

`is_real`, `is_real_positive`, and `is_real_nonnegative` operate on array-like
values and require every contained value to be a finite real number. Boolean and
complex values do not count as real for these predicates.

- `is_real_positive` means strictly greater than zero.
- `is_real_nonnegative` means greater than or equal to zero.

An input that is not array-like raises `TypeError`. Empty array-like values
satisfy these universal predicates by vacuous truth. In particular,
`is_real([])`, `is_real_positive([])`, and `is_real_nonnegative([])` return
`True`, as do the corresponding predicates for empty supported array
containers. Every public predicate's documentation must state its empty-input
behavior explicitly.

## Shape contracts

### `infer_shape(value)`

Performs structural validation without conversion or allocation. It returns the
shape of an array-like value and returns `None` when the value is not array-like.

It must reject ragged, string-like, non-numeric, and self-referential sequences.

### `dim(value)`

Returns the number of dimensions of an array-like value. It raises `TypeError`
when the input is not array-like.

### `shape.len(value)`

Returns the first extent of a non-scalar array-like value. It raises `TypeError`
for scalars and invalid values. It is deliberately not exported from the package
root because it conflicts with Python's built-in `len`.

## Conversion contracts

The conversion guarantees in this section cover dense NumPy arrays and dense
strided PyTorch tensors. Specialized PyTorch layouts and views that cannot be
exposed directly to NumPy, including sparse tensors and unresolved conjugate or
negative-bit views, are outside the current conversion guarantee and may raise
a backend error.

### `array(value, mode="numpy")`

Intended role: normalize an array-like value into an array container for the
selected backend mode.

`array` accepts every valid array-like value, including number-like scalars. A
scalar input produces a zero-dimensional array container on the selected
backend.

The result owns storage independently from the input. Mutating the result must
not mutate an input array container, and mutating an input array container must
not mutate the result.

The following details also require explicit decisions:

- behavior for sequences containing array containers;
- exception type when the requested backend is unavailable;
- whether `torch_cuda` may perform an implicit device transfer.

Invalid backend modes raise `ValueError`. Requesting CUDA when it is unavailable
raises `RuntimeError` in the current implementation.

### `asarray(value, mode="numpy")`

`asarray` accepts the same inputs, modes, and dtype rules as `array`, but reuses
or shares existing storage whenever the target backend and device permit it.

- A same-backend, same-device array container may be returned unchanged.
- A compatible NumPy array and CPU PyTorch tensor share storage when converted
  between those backends.
- A read-only NumPy array is copied when converted to PyTorch so that writable
  tensor access cannot modify storage presented as read-only by NumPy.
- A NumPy array with negative strides is copied when converted to PyTorch,
  because PyTorch cannot consume that layout directly.
- A device transfer necessarily allocates new storage.
- Converting a scalar or Python sequence necessarily allocates array storage.
- When sharing is unsupported for a dtype, layout, or backend combination,
  `asarray` may copy rather than fail solely because sharing is impossible.

Callers must assume that mutating an `asarray` result may mutate the input and
vice versa. A PyTorch tensor is detached before NumPy conversion: shared storage
may remain, but the NumPy result does not participate in autograd.

### Dtype behavior during conversion

Conversion of an existing array container preserves its dtype when the target
backend supports that dtype. If the target backend cannot represent the dtype,
conversion raises an error. Conversion must never change dtype merely to make
memory sharing possible.

Python scalars and sequences do not carry a backend dtype. The selected backend
applies its normal dtype inference rules when converting them.

### `to_list(value)`

Accepts any array-like value. Array containers and nested sequences become
nested Python lists. A zero-dimensional input produces a Python scalar rather
than a one-element list. GPU tensors are detached and transferred to CPU before
conversion.

Detaching a tensor that requires gradients is an intentional part of the
conversion contract. The resulting Python values and NumPy arrays do not
participate in the original PyTorch computation graph.

### `add_axis(array, *axes)`

Currently accepts array containers only and returns a new container on the same
backend. It does not mutate the input.

With no axis positions, it returns an independent copy of the input. Otherwise,
each position must be a unique integer position in the final result. Positions
must be either all nonnegative or all negative; negative positions are
normalized against the final number of dimensions. Mixed-sign, duplicate, and
out-of-range positions raise `ValueError`; non-integer positions raise
`TypeError`.

### `zeros_like(array)`

Accepts an array container and returns a zero-filled container with the same
backend, shape, and dtype. For PyTorch it also preserves the device.

No preservation guarantee is made for gradient flags, layouts, or other
backend-specific metadata.

## Numerical API policy

Mathematical and Fourier functions accept array containers only. They must not
implicitly convert array-like values. Callers use `array` explicitly to choose a
backend and normalize their input before computation.

This separation is part of the public design:

- conversion functions accept `ArrayLike` and produce `Array`;
- numerical functions accept `Array` and preserve its backend;
- numerical functions reject scalars and Python sequences even when those
  values could be converted to arrays;
- numerical functions never select a backend on behalf of the caller.

For example, `fft([1, 2])` is invalid, while
`fft(array([1, 2], mode="numpy"))` is valid.

The detailed numerical contracts must subsequently define:

- accepted dtypes and dtype promotion;
- scalar and empty-array behavior;
- axis and multi-axis normalization;
- backend and device preservation;
- exact versus approximate equality expectations;
- error categories for invalid inputs;
- behavior when an optional backend is unavailable.

`fftfreq(n, d, mode)` is a construction helper and follows the native backend
return type for multidimensional grids. A scalar `n` returns one NumPy array or
PyTorch tensor. A tuple `n` returns the backend's `meshgrid` result. Requesting
`mode="torch_cuda"` without CUDA support raises `RuntimeError`.

## Known inconsistencies

These are observations, not adopted contracts:

1. Deprecated `checks.is_array` considers any list an array, whereas
   `validation.is_array` accepts only backend containers with supported dtypes.
2. Deprecated "positive" checks mean nonnegative; the new validation API
   distinguishes positive from nonnegative.
3. `ArrayError.INVALID_ARRAY_TYPE` says an input must be a list, NumPy array, or
   tensor, which conflicts with the `ArrayLike` definition and tuple support.
4. Some function documentation uses "array-like" where the function is required
   to accept array containers only.

## Turning contracts into tests

Tests should be derived from the settled rules in this document:

- one or two named examples for each central behavior;
- parameterized tables for explicit boundaries and invalid cases;
- property-based tests for rectangular shapes, values, dtypes, and axes;
- backend-parity tests only where the contract promises parity;
- regression tests for confirmed bugs.

An unresolved decision must not become an assertion in a normative contract
test. Existing behavior may be captured temporarily as a characterization test,
but such a test must be labelled accordingly.

## Review order

Continue the review in this order because later contracts depend on earlier
ones:

1. Define the deprecated-check removal schedule.
2. Resolve the remaining conversion decisions: backend availability, gradient
   detachment, axes, sequences containing containers, and preserved metadata.
3. Define individual mathematical and Fourier contracts under the settled
   array-container-only policy.
4. Update every relevant docstring to state empty-input behavior explicitly.
