# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0-alpha.1] - 2026-10-07

This is the first public alpha release of OneArray. Its API is intentionally
unstable while the project gathers feedback before `1.0.0`.

### Added

- Backend-agnostic array creation and conversion for NumPy and PyTorch,
  including explicit CPU and CUDA modes.
- Backend-preserving mathematical helpers: `sum`, `exp`, and `abs`.
- Fourier helpers: `fft`, `ifft`, `fftfreq`, `fftshift`, and `ifftshift`.
- Shape inference, dimensionality, length, validation, list-conversion,
  axis-insertion, and zero-allocation helpers.
- Explicit contracts for supported scalar types, rectangular sequences,
  dtype preservation, memory sharing, backend conversion, and FFT behavior.
- Test coverage across supported Python, NumPy, and PyTorch versions, including
  tests against the installed wheel outside the source checkout.
- Online documentation and an automatically generated public API reference.

### Changed

- Restricted number-like scalars to Python numeric and boolean scalars plus
  NumPy numeric and boolean scalars. Other numeric implementations require
  explicit conversion.
- Limited the package-root namespace to the supported primary API. Validation,
  shape, and type helpers remain available from their documented submodules.

### Fixed

- Preserved shapes when converting empty nested sequences to PyTorch tensors.
- Rejected scalar inputs consistently from the array-length helper.
- Treated empty numeric arrays and tensors as vacuously real for real-value
  predicates.
- Validated FFT axes consistently and checked CUDA availability before creating
  CUDA frequency arrays.
- Avoided unnecessary copies in `asarray` while rejecting unsafe shared-memory
  conversions.

[Unreleased]: https://github.com/optimato/onearray/compare/v0.1.0-alpha.1...HEAD
[0.1.0-alpha.1]: https://github.com/optimato/onearray/releases/tag/v0.1.0-alpha.1
