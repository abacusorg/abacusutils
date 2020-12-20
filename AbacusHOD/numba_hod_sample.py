import numba
import math
import numpy as np
import scipy.special

@numba.njit
def n_cent(M_in, sigma=0.5, M_cut=0.5):
    return 0.5*math.erfc(np.log(M_cut/M_in)/(2**.5*sigma))


@numba.njit(parallel=True)
def do_halos_twopassflag(pos, vel, M, r):
    H = len(pos)
    
    Nthread = numba.get_num_threads()
    Nout = np.zeros((Nthread,8), dtype=np.int64)  # note the cache line padding
    hstart = np.rint(np.linspace(0, H, Nthread+1)).astype(np.int64)
    
    # if the results are sparse, could keep per-thread index arrays instead of bool flags
    keep = np.empty(H, dtype=np.int8)
    
    for tid in numba.prange(Nthread):
        for i in range(hstart[tid], hstart[tid+1]):
            if n_cent(M[i]) > r[i]:
                Nout[tid,0] += 1
                keep[i] = 1
            else:
                keep[i] = 0
                
    gstart = np.empty(Nthread+1, dtype=np.int64)
    gstart[0] = 0
    gstart[1:] = Nout[:,0].cumsum()
    
    gpos = np.empty((gstart[-1],3), dtype=pos.dtype)
    gvel = np.empty((gstart[-1],3), dtype=pos.dtype)
    for tid in numba.prange(Nthread):
        j = gstart[tid]
        for i in range(hstart[tid], hstart[tid+1]):
            if keep[i]:
                gpos[j] = pos[i]
                gvel[j] = vel[i]
                j += 1
        assert j == gstart[tid+1]
    
    return gpos, gvel

  
H = int(1e7)
M = np.random.rand(H)
pos = np.random.rand(H,3)
vel = np.random.rand(H,3)
r = np.random.rand(H)

numba.set_num_threads(8)
%timeit do_halos_twopassflag(pos, vel, M, r)