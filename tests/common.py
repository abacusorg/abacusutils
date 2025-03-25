"""
Common resources for the tests.
"""

import numbers

import numpy.testing as npt


def assert_close(arr1, arr2):
    """Checks exact equality for int arrays, and np.isclose for floats"""
    if arr1.dtype.names is not None:
        notinboth = set(arr1.dtype.names) ^ set(arr2.dtype.names)
        assert not notinboth, f"Field names don't match: {notinboth=}"
        # if the arrays are structured, check each field
        for name in arr1.dtype.names:
            try:
                assert_close(arr1[name], arr2[name])
            except AssertionError as e:
                raise AssertionError(f'Field "{name}" does not match') from e
        return

    if issubclass(arr1.dtype.type, numbers.Integral):
        assert issubclass(arr2.dtype.type, numbers.Integral)
        npt.assert_array_equal(arr1, arr2)
    else:
        npt.assert_allclose(arr1, arr2)
