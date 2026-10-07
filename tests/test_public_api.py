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

    from onearray.shape import infer_shape, len as array_len
    from onearray.validation import is_array

    assert infer_shape([1, 2]) == (2,)
    assert array_len([1, 2]) == 2
    assert not is_array([1, 2])


def test_public_submodule_exports_are_explicit():
    from onearray import shape, types, validation

    assert set(shape.__all__) == {"dim", "infer_shape", "len"}
    assert set(types.__all__) == {"NumberLike", "Array", "ArrayLike", "ArrayMode"}
    assert set(validation.__all__) == {
        "is_numberlike",
        "is_array",
        "is_array_like",
        "all_real_and",
        "is_real",
        "is_real_positive",
        "is_real_nonnegative",
        "is_valid_array_mode",
    }
