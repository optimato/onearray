# OneArray

**NumPy syntax across array packages**

Have you ever written some lengthy code in NumPy, only to later realize you needed
some PyTorch functionality and had to rewrite everything from scratch?

*If you did, then OneArray might be the package for you.*

## What it does

OneArray lets you write generic code and choose the *backend* later.
The same function works with NumPy arrays and PyTorch tensors, selecting 
the appropriate implementation from the input at runtime.

The central rule is simple:

> Numerical operations preserve and follow the backend of their input. 
> Conversion utilities are available when data needs to move from one backend to another.

Everything stays NumPy or PyTorch. There are no new classes to learn. The
API has a familiar NumPy feel, leading to readable code and intuitive usage:

```python
import onearray as oa

x_np = oa.array([1.0, 2.0, 3.0], mode="numpy")
x_torch = oa.array([1.0, 2.0, 3.0], mode="torch")

y_np = oa.exp(x_np)  # numpy array
y_torch = oa.exp(x_torch)  # torch tensor
```

OneArray currently supports NumPy arrays and PyTorch tensors, including CUDA
tensors where an operation and the local PyTorch installation support them.

Python scalars and rectangular nested sequences are *array-like*. They become
NumPy arrays or PyTorch tensors only when passed to an explicit conversion
function, but they can be validated without any allocation by OneArray's validation 
utilities. Numerical functions do not perform any conversion. They select the 
appropriate NumPy or PyTorch implementation from the input at runtime.


## Learn more

- [Arrays and backends](guide/arrays-and-backends.md)
- [Conversion and memory](guide/conversion-and-memory.md)