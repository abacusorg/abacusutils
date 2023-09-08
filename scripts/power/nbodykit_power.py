#!/usr/bin/env python3

import timeit

import click
import numpy as np
import nbodykit
from nbodykit.source.catalog import UniformCatalog
from nbodykit.algorithms.fftpower import FFTPower


@click.command()
@click.argument('ngrid', default=256)
@click.option('-r', '--nrep', default=3)
def main(**kwargs):
    ngrid = kwargs['ngrid']
    nrep = kwargs['nrep']

    comm = nbodykit.CurrentMPIComm.get()

    N = 10**7

    cat = UniformCatalog(N, BoxSize=1., dtype='f4')

    mesh = cat.to_mesh(Nmesh=ngrid, resampler='tsc', compensated=False,
        interlaced=False, dtype='f4',
    ).compute()
    FFTPower(mesh, mode='1d', dk=np.pi*ngrid/100, kmin=0., kmax=np.pi*ngrid)

    t = -timeit.default_timer()
    for _ in range(nrep):
        mesh = cat.to_mesh(Nmesh=ngrid, resampler='tsc', compensated=False,
            interlaced=False, dtype='f4',
        ).compute()
        FFTPower(mesh, mode='1d', dk=np.pi*ngrid/100, kmin=0., kmax=np.pi*ngrid)
    t += timeit.default_timer()

    if comm.rank == 0:
        print(f'Time: {t/nrep:.4g} sec x {nrep} rep')
        print(f'Rate: {N/(t/nrep)/1e6:.4g} M part/sec')


if __name__ == '__main__':
    main()
