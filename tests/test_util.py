import numpy as np
from numpy.testing import assert_array_equal
import pytest
from abacusnbody.util import cumsum


def test_cumsum_final():
    arr = np.array([1, 2, 3, 4])
    out = np.empty(4, dtype=arr.dtype)
    total = cumsum(arr, out, initial=False, final=True)
    assert_array_equal(out, np.array([1, 3, 6, 10]))
    assert total == 10

    # this version is equivalent to numpy.cumsum()
    np_res = np.cumsum(arr)
    assert_array_equal(out, np_res)


def test_cumsum_initial():
    arr = np.array([1, 2, 3, 4])
    out = np.empty(4, dtype=arr.dtype)
    total = cumsum(arr, out, initial=True, final=False)
    assert_array_equal(out, np.array([0, 1, 3, 6]))
    assert total == 10


def test_cumsum_no_initial_final():
    arr = np.array([1, 2, 3, 4])
    out = np.empty(3, dtype=arr.dtype)
    total = cumsum(arr, out, initial=False, final=False)
    assert_array_equal(out, np.array([1, 3, 6]))
    assert total == 10


def test_cumsum_initial_and_final():
    arr = np.array([1, 2, 3, 4])
    out = np.empty(5, dtype=arr.dtype)
    total = cumsum(arr, out, initial=True, final=True)
    assert_array_equal(out, np.array([0, 1, 3, 6, 10]))
    assert total == 10


def test_cumsum_with_offset():
    arr = np.array([1, 2, 3, 4])
    out = np.empty(4, dtype=arr.dtype)
    total = cumsum(arr, out, offset=5)
    assert_array_equal(out, np.array([6, 8, 11, 15]))
    assert total == 15


def test_cumsum_output_length_error():
    arr = np.array([1, 2, 3, 4])
    out = np.empty(3, dtype=arr.dtype)
    with pytest.raises(ValueError):
        cumsum(arr, out)
