# Versioning and public API

OneArray follows [Semantic Versioning 2.0.0](https://semver.org/spec/v2.0.0.html).
The rules on this page define the public API whose compatibility is communicated
by OneArray's version numbers.

## Version identifiers

Upstream releases use SemVer identifiers. Git tags add a conventional `v`
prefix; for example, the first alpha is tagged `v0.1.0-alpha.1`.

Python distribution metadata follows the Python packaging version standard.
PyPI therefore normalizes the same alpha to `0.1.0a1`. These two strings name
the same release:

| Context | Version |
| --- | --- |
| SemVer, Git tag, and GitHub release | `0.1.0-alpha.1` (`v0.1.0-alpha.1` as a tag) |
| Python package metadata and PyPI | `0.1.0a1` |

Release tags do not use SemVer build metadata (`+...`) because public Python
package indexes reserve local version labels for downstream builds. The release
workflow rejects tags that are not both valid SemVer and suitable for PyPI, and
it verifies that the built package version equals the Python-normalized tag.

Versions below `1.0.0` are for initial development. Their public API is not yet
stable and may change between releases. Pre-release identifiers additionally
mean that a release may not satisfy the compatibility guarantees of its
associated normal version.

Published releases are immutable. Corrections are made in a new release; files
belonging to an existing version are never replaced.

## Public API

The public API consists only of:

- names listed in `onearray.__all__`, imported directly from `onearray`;
- names listed in `onearray.validation.__all__`, imported from
  `onearray.validation`;
- names listed in `onearray.shape.__all__`, imported from `onearray.shape`; and
- names listed in `onearray.types.__all__`, imported from `onearray.types`.

The generated [API Reference](../reference/onearray/) documents those namespaces
and their members. The behavioral guarantees for those objects are defined by the
[API contracts](contracts.md).

Everything else is private implementation detail. In particular, underscore-
prefixed names, backend helpers, implementation modules, and the deprecated
`onearray.checks` compatibility module are not public API. Their presence in an
installed distribution does not imply a compatibility guarantee, and they are
intentionally omitted from the API reference.

## Release policy

Until `1.0.0`, OneArray increments the minor version for a new development
series and uses patch and pre-release identifiers for successive releases in
that series. Every release records user-visible changes in the
[changelog](https://github.com/optimato/onearray/blob/main/CHANGELOG.md).

Starting with `1.0.0`:

- incompatible public API changes increment the major version;
- backward-compatible public API additions and deprecations increment the
  minor version; and
- backward-compatible bug fixes increment the patch version.

Deprecations are documented before removal. Removing a public API after
`1.0.0` requires a new major version.
