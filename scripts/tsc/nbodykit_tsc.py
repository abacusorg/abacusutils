#!/usr/bin/env python3

import timeit
import click
import nbodykit
from nbodykit.source.catalog import UniformCatalog


@click.command()
def main():
    comm = nbodykit.CurrentMPIComm.get()

    N = 10**7
    ngrid = 256

    cat = UniformCatalog(N, BoxSize=1., dtype='f4')

    t = -timeit.default_timer()
    cat.to_mesh(Nmesh=ngrid, resampler='tsc', compensated=False,
        interlaced=False, dtype='f4',
    ).compute()
    t += timeit.default_timer()

    # print(np.sum(mesh))

    if comm.rank == 0:
        print(f'Time: {t:.4g} sec')
        print(f'Rate: {N/1e6/t:.4g}M part/sec')


if __name__ == '__main__':
    main()
