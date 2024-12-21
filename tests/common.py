"""
Common resources for the tests.
"""

import numbers

import numpy.testing as npt


def assert_close(arr1, arr2):
    """Checks exact equality for int arrays, and np.isclose for floats"""
    if issubclass(arr1.dtype.type, numbers.Integral):
        assert issubclass(arr2.dtype.type, numbers.Integral)
        npt.assert_array_equal(arr1, arr2)
    else:
        npt.assert_allclose(arr1, arr2)
