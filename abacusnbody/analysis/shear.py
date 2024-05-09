import time
import gc

import numpy as np
import numpy.linalg as la
import numba
from scipy.fft import irfftn, rfftn
from scipy.ndimage import gaussian_filter

"""
Code still under construction. Originally written by Boryana Hadzhiyska for the ancient: https://arxiv.org/abs/1512.03402.
"""


def smooth_density(D, R, N_dim, Lbox):
    # cell size
    cell = Lbox / N_dim
    # smoothing scale
    R /= cell
    D_smooth = gaussian_filter(D, R)
    return D_smooth


# tophat
@numba.njit
def Wth(ksq, r):
    k = np.sqrt(ksq)
    w = 3 * (np.sin(k * r) - k * r * np.cos(k * r)) / (k * r) ** 3
    return w


# gaussian
@numba.njit
def Wg(k, r):
    return np.exp(-k * r * r / 2.0)


@numba.njit(parallel=False, fastmath=True)  # parallel=True gives seg fault
def get_tidal(dfour, karr, N_dim, R, dtype=np.float32):
    # initialize array
    tfour = np.zeros((N_dim, N_dim, N_dim // 2 + 1, 6), dtype=np.complex64)

    # computing tidal tensor
    for a in range(N_dim):
        for b in range(N_dim):
            for c in numba.prange(N_dim // 2 + 1):
                if a * b * c == 0:
                    continue

                ksq = dtype(karr[a] ** 2 + karr[b] ** 2 + karr[c] ** 2)
                dok2 = dfour[a, b, c] / ksq

                # smoothed density Gauss fourier
                # dksmo[a, b, c] = Wg(ksq)*dfour[a, b, c]
                # smoothed density TH fourier
                # dkth[a, b, c] = Wth(ksq)*dfour[a, b, c]
                # 0,0 is 0; 0,1 is 1; 0,2 is 2; 1,1 is 3; 1,2 is 4; 2,2 is 5
                tfour[a, b, c, 0] = karr[a] * karr[a] * dok2
                tfour[a, b, c, 3] = karr[b] * karr[b] * dok2
                tfour[a, b, c, 5] = karr[c] * karr[c] * dok2
                tfour[a, b, c, 1] = karr[a] * karr[b] * dok2
                tfour[a, b, c, 2] = karr[a] * karr[c] * dok2
                tfour[a, b, c, 4] = karr[b] * karr[c] * dok2
                if R is not None:
                    tfour[a, b, c, :] *= Wth(ksq, R)
    return tfour


@numba.njit(parallel=False, fastmath=True)
def get_shear_nb(tidr, N_dim):
    shear = np.zeros(shape=(N_dim, N_dim, N_dim), dtype=np.float32)
    tensor = np.zeros((3, 3), dtype=np.float32)
    for a in range(N_dim):
        for b in range(N_dim):
            for c in range(N_dim):
                t = tidr[a, b, c, :]
                tensor[0, 0] = t[0]
                tensor[0, 1] = t[1]
                tensor[0, 2] = t[2]
                tensor[1, 0] = t[1]
                tensor[1, 1] = t[3]
                tensor[1, 2] = t[4]
                tensor[2, 0] = t[2]
                tensor[2, 1] = t[4]
                tensor[2, 2] = t[5]
                evals = la.eigvals(tensor)
                l1 = evals[0]
                l2 = evals[1]
                l3 = evals[2]
                shear[a, b, c] = np.sqrt(
                    0.5 * ((l2 - l1) ** 2 + (l3 - l1) ** 2 + (l3 - l2) ** 2)
                )
    return shear


def get_shear(dsmo, N_dim, Lbox, R=None, dtype=np.float32):
    # user can also pass string
    if isinstance(dsmo, str):
        dsmo = np.load(dsmo)

    # fourier transform the density field
    dsmo = dsmo.astype(dtype)
    dfour = rfftn(dsmo, overwrite_x=True, workers=-1)
    del dsmo
    gc.collect()

    # k values
    karr = np.fft.fftfreq(N_dim, d=Lbox / (2 * np.pi * N_dim)).astype(dtype)

    # compute fourier tidal
    start = time.time()
    tfour = get_tidal(dfour, karr, N_dim, R)
    del dfour
    gc.collect()
    print('finished fourier tidal, took time', time.time() - start)

    # compute real tidal
    start = time.time()
    tidr = irfftn(tfour, axes=(0, 1, 2), workers=-1).real
    del tfour
    gc.collect()
    print('finished tidal, took time', time.time() - start)

    # compute shear
    start = time.time()
    shear = get_shear_nb(tidr, N_dim)
    del tidr
    gc.collect()
    print('finished shear, took time', time.time() - start)

    return shear
