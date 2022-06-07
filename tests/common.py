'''
Common resources for the tests.
'''

import numbers
import numpy as np

def check_close(arr1, arr2):
    '''Checks exact equality for int arrays, and np.isclose for floats
    '''
    if issubclass(arr1.dtype.type, numbers.Integral):
        assert issubclass(arr2.dtype.type, numbers.Integral)
        return np.all(arr1 == arr2)
    else:
        return np.allclose(arr1, arr2)
