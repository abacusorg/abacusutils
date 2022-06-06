import numpy as np
from numba import jit

@jit(nopython = True)
def histogram_hp(rho,heal):
    """ Faster histograming """
    for i in range(len(heal)):
        ind = heal[i]
        rho[ind] += 1

    return rho
