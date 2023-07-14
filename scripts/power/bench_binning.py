#!/usr/bin/env python3

import os
import timeit

import click
import numba
import numpy as np

NTHREAD = len(os.sched_getaffinity(0))

@numba.njit(parallel=True, fastmath=True)
def bin_kmu(n1d, L, kedges, Nmu, dtype=np.float32):
    '''
    Count modes in (k,mu) bins for a 3D rfft mesh of shape (n1d, n1d, n1d//2+1).

    The k and mu values are constructed on the fly.  We use the monotonicity of
    k and mu with respect to kz to accelerate the bin search.

    The main opportunity we're not taking advantage of here is an eightfold
    symmetry in k-space. One just has to take care to single count the boundary
    modes and double count the others, as one does with the rfft symmetry.
    '''

    kzlen = n1d//2 + 1
    Nk = len(kedges) - 1

    kedges2 = ((kedges/(2*np.pi/L))**2).astype(dtype)
    muedges2 = (np.linspace(0., 1., Nmu+1)**2).astype(dtype)

    nthread = numba.get_num_threads()
    counts = np.zeros((nthread, Nk, Nmu), dtype=np.int64)

    # Loop over all k vectors
    for i in numba.prange(n1d):
        tid = numba.get_thread_id()
        i2 = i**2 if i <= n1d//2 else (n1d - i)**2
        for j in range(n1d):
            bk,bmu = 0,0
            j2 = j**2 if j <= n1d//2 else (n1d - j)**2
            for k in range(kzlen):
                kmag2 = dtype(i2 + j2 + k**2)
                if kmag2 > 0:
                    invkmag2 = kmag2**-1
                    mu2 = dtype(k**2) * invkmag2
                else:
                    mu2 = dtype(1.)

                if kmag2 < kedges2[0]:
                    continue

                if kmag2 >= kedges2[-1]:
                    break

                while kmag2 > kedges2[bk+1]:
                    bk += 1

                while mu2 > muedges2[bmu+1]:
                    bmu += 1

                counts[tid, bk, bmu] += 1 if k == 0 else 2

    counts = counts.sum(axis=0)
    return counts


def bin_kmu_weighted():
    pass

@click.command
@click.argument('n1d', default=256)
@click.argument('dtype', default='f4')
@click.option('-t', '--nthreads', 'nworker', default=NTHREAD)
def main(n1d: int = 256,
         dtype: str = 'f4',
         nworker: int = NTHREAD,
         ):
    config = dict(n1d=n1d, dtype=dtype, nworker=nworker)
    rng = np.random.default_rng(300)
    numba.set_num_threads(nworker)

    # pk3d = np.empty((n1d,n1d,n1d//2+1), dtype=dtype)
    # rng.random((n1d,n1d,n1d), dtype=dtype, out=pk3d)

    Nk = 40
    Nmu = 50
    kedges = np.linspace(0, np.pi*n1d, Nk)
    res = bin_kmu(n1d, 1, kedges, Nmu)
    print(res)
    print(res.sum())
    with open('types.txt', 'w') as fp:
        bin_kmu.inspect_types(fp)
    # breakpoint()
    cmd = 'bin_kmu(n1d, 1, kedges, Nmu)'

    number = max(3,int(12*nworker**0.5*(64/n1d)))
    t = timeit.repeat(
        cmd,
        repeat=5,
        number=number,
        globals=globals() | locals(),
    )
    t = np.array(t) / number
    print(f'Time: {t.min() * 1e3:.2f} ms')
    print(f'Config: {config}')

if __name__ == '__main__':
    main()
