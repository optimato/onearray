import onearray as oa


def test_package_root_exports_supported_primary_api():
    expected = {
        "dim",
        "array",
        "asarray",
        "add_axis",
        "to_list",
        "zeros_like",
        "sum",
        "exp",
        "abs",
        "fft",
        "ifft",
        "fftfreq",
        "fftshift",
        "ifftshift",
    }

    assert set(oa.__all__) == expected
    assert all(hasattr(oa, name) for name in expected)


def test_validation_and_shape_helpers_remain_submodule_apis():
    assert not hasattr(oa, "is_array")
    assert not hasattr(oa, "infer_shape")

    from onearray.shape import infer_shape
    from onearray.validation import is_array

    assert infer_shape([1, 2]) == (2,)
    assert not is_array([1, 2])
