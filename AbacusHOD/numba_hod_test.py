#!/usr/bin/env python3

'''Test the speed of a Numba-based HOD'''

import math
import timeit

import numpy as np
import numba
from numba import njit

# configuration
dtype = np.float32  # float dtype
H = int(2.4e7)  # num halos
numba.set_num_threads(16)  # num threads


@njit(fastmath=True)
def n_cen(M_in, M_cut, sigma, m_cutoff = 1e12): 
    return 0.5*math.erfc((np.log(M_cut) - np.log(M_in))/(2**.5*sigma))

@njit(fastmath=True)
def n_sat(M_in, M_cut, M1, sigma, alpha, kappa, m_cutoff = 1e12): 
    return ((M_in - kappa*M_cut)/M1)**alpha*0.5*math.erfc((np.log(M_cut) - np.log(M_in))/(2**.5*sigma))


@njit(fastmath=True)
def wrap(x, L):
    '''Fast scalar mod implementation'''
    L2 = L/2
    if x >= L2:
        return x - L
    elif x < -L2:
        return x + L
    return x


@njit(parallel=True,fastmath=True)
def gen_cent(pos, vel, mass, ids, randoms, design_array, ic, rsd, inv_velz2kms, lbox):
    # parse out the hod parameters 
    M_cut, M1, sigma, alpha, kappa = \
    design_array[0], design_array[1], design_array[2], design_array[3], design_array[4]

    H = len(mass)

    Nthread = numba.get_num_threads()
    Nout = np.zeros((Nthread, 8), dtype = np.int64)
    hstart = np.rint(np.linspace(0, H, Nthread + 1)) # starting index of each thread

    keep = np.empty(H, dtype = np.int8) # mask array tracking which halos to keep

    # figuring out the number of halos kept for each thread
    for tid in numba.prange(Nthread):
        for i in range(hstart[tid], hstart[tid + 1]):
            if n_cen(mass[i], M_cut, sigma) * ic > randoms[i]:
                Nout[tid, 0] += 1 # counting
                keep[i] = 1
            else:
                keep[i] = 0

    # compose galaxy array, first create array of galaxy starting indices for the threads
    gstart = np.empty(Nthread + 1, dtype = np.int64)
    gstart[0] = 0
    gstart[1:] = Nout[:, 0].cumsum()

    # galaxy arrays
    gpos = np.empty((gstart[-1], 3), dtype = pos.dtype)
    gvel = np.empty((gstart[-1], 3), dtype = vel.dtype)
    gmass = np.empty(gstart[-1], dtype = mass.dtype)
    gid = np.empty(gstart[-1], dtype = ids.dtype)

    # fill in the galaxy arrays
    for tid in numba.prange(Nthread):
        j = gstart[tid]
        for i in range(hstart[tid], hstart[tid + 1]):
            if keep[i]:
                for k in range(3):
                    if rsd:
                        gpos[j,k] = wrap(pos[i,k] + vel[i,k] * inv_velz2kms, lbox)
                    else:
                        gpos[j,k] = pos[i,k]
                    gvel[j,k] = vel[i,k] # need to extend to include vel bias 
                gmass[j] = mass[i]
                gid[j] = ids[i]
                j += 1
        # assert j == gstart[tid + 1]

    return gpos, gvel, gmass, gid 

# make fake data
rng = np.random.default_rng()
pos = rng.random((H,3), dtype=dtype)-0.5
vel = rng.random((H,3), dtype=dtype)
mass = rng.random(H, dtype=dtype)
ids = rng.integers(H, size=H)
randoms = rng.random(H, dtype=dtype)
design_array = np.array([0.5,0.5,0.5,0.5,0.5], dtype=dtype)
ic = dtype(1.)
rsd = True
velz2kms = dtype(1.)
lbox = dtype(1.)

# time the centrals
res = gen_cent(pos, vel, mass, ids, randoms, design_array, ic, rsd, velz2kms, lbox)  # force jit compilation
timer = timeit.Timer('gen_cent(pos, vel, mass, ids, randoms, design_array, ic, rsd, velz2kms, lbox)', globals=globals())
nrep, tot_time = timer.autorange()
print(f'Generated {len(res[0]):.3g} centrals from {H:.3g} halos in {tot_time/nrep*1e3:.1f} ms with {numba.get_num_threads()} threads')