#!/usr/bin/env python3

import timeit
import click
import nbodykit
from nbodykit.source.catalog import UniformCatalog


@click.command()
@click.option('-r', '--nrep', default=3)
def main(**kwargs):
    nrep = kwargs['nrep']

    comm = nbodykit.CurrentMPIComm.get()

    N = 10**7
    ngrid = 256

    cat = UniformCatalog(N, BoxSize=1., dtype='f4')

    t = -timeit.default_timer()
    for _ in range(nrep):
        cat.to_mesh(Nmesh=ngrid, resampler='tsc', compensated=False,
            interlaced=False, dtype='f4',
        ).compute()
    t += timeit.default_timer()

    # print(np.sum(mesh))

    if comm.rank == 0:
        print(f'Time: {t/nrep:.4g} sec x {nrep} rep')
        print(f'Rate: {N/(t/nrep)/1e6:.4g} M part/sec')


if __name__ == '__main__':
    main()
