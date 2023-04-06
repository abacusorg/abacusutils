#!/usr/bin/env python3

import timeit

import click
import numba
import numpy as np

import abacusnbody.analysis.power_spectrum
# from tsc_gather import tsc_gather
from abacusnbody.analysis import tsc


@click.command()
@click.argument('n', default=10**6)
@click.argument('ngrid', default=256)
@click.option('-t', '--nthread', default=1)
@click.option('-r', '--nrep', default=3)
@click.option('-p', '--npartition', default=None, type=int)
@click.option('-s', '--sort', is_flag=True)
@click.option('-w', '--weights', is_flag=True)
def main(**kwargs):
    n = kwargs['n']
    ngrid = kwargs['ngrid']
    nthread = kwargs['nthread']
    nrep = kwargs['nrep']
    npartition = kwargs['npartition']
    sort = kwargs['sort']
    weights = kwargs['weights']

    if nthread < 0:
        nthread = numba.config.NUMBA_NUM_THREADS

    max_partition = ngrid // 2  # algorithmic max
    if not npartition:
        if nthread != 1:
            npartition = min(max_partition, 2*nthread)
            npartition = 2 * (npartition // 2)
        else:
            npartition = 1

    print(kwargs)

    box = 123.
    rng = np.random.default_rng(0xDEADBEEF)
    pos = rng.random((n,3), dtype=np.float32)*box
    if weights:
        weights = rng.random(n, dtype=np.float32)
    else:
        weights = None

    dens = np.zeros((ngrid,ngrid,ngrid), dtype=np.float32)

    # burn-in
    # psort, offsets = particle_grid.particle_grid(pos, ngrid, box, nthread=nthread, ngp=False)
    # tsc_gather(psort, offsets, dens, box, nthread=nthread)
    # tsc_gather.inspect_types()
    # abacusnbody.analysis.power_spectrum.numba_tsc_3D(pos, dens, box)
    # particle_grid.numba_tsc_3D(pos, dens, box)
    # tsc_scatter.tsc_scatter(pos, dens, box)
    tsc.tsc_parallel(pos, dens, box, nthread=nthread,
        npartition=npartition, sort=sort, weights=weights,
        )
    # tsc_scatter.inspect_types()
    ppart, starts, wpart = tsc.partition_parallel(pos, npartition, box,
        nthread=nthread, sort=sort, weights=weights,
        )

    print("Threading layer chosen: %s" % numba.threading_layer())
    # breakpoint()
    # tsc_scatter.partition_parallel.inspect_types(pretty=False)
    # print((a:=tsc_scatter.partition_parallel.inspect_asm())[list(a)[0]])
    # return
    # tottime = -timeit.default_timer()
    # psort, offsets = particle_grid.particle_grid(pos, ngrid, box, nthread=nthread, ngp=False)
    # tottime += timeit.default_timer()
    # print('Particle grid')
    # print('=============')
    # print(f'Time: {tottime/1:.4g} sec x {1} rep')
    # print(f'Rate: {n/(tottime/1)/1e6:.4g} Mp/s')

    tottime = -timeit.default_timer()
    for _ in range(nrep):
        tsc.partition_parallel(pos, npartition, box, nthread=nthread,
            sort=sort, weights=weights,
            )
    tottime += timeit.default_timer()

    print('Particle partition')
    print('==================')
    print(f'Time: {tottime/nrep:.4g} sec x {nrep} rep')
    print(f'Rate: {n/(tottime/nrep)/1e6:.4g} Mp/s')

    # tottime = -timeit.default_timer()
    # pos = pos[pos[:,0].argsort()]
    # tottime += timeit.default_timer()

    # print('Particle partition')
    # print('==================')
    # print(f'Time: {tottime/1:.4g} sec x {1} rep')
    # print(f'Rate: {n/(tottime/1)/1e6:.4g} Mp/s')

    # tottime = -timeit.default_timer()
    # for _ in range(nrep):
    #     tsc_gather(psort, offsets, dens, box, nthread=nthread)
    # tottime += timeit.default_timer()

    # print()
    # print('TSC Gather')
    # print('==========')
    # print(f'Time: {tottime/nrep:.4g} sec x {nrep} rep')
    # print(f'Rate: {n/(tottime/nrep)/1e6:.4g} Mp/s')

    tottime = -timeit.default_timer()
    for _ in range(nrep):
        tsc.tsc_parallel(pos, dens, box, nthread=nthread,
            npartition=npartition, sort=sort, weights=weights,
            )
    tottime += timeit.default_timer()

    print()
    print('TSC Parallel')
    print('============')
    print(f'Time: {tottime/nrep:.4g} sec x {nrep} rep')
    print(f'Rate: {n/(tottime/nrep)/1e6:.4g} Mp/s')

    tottime = -timeit.default_timer()
    for _ in range(nrep):
        tsc._tsc_scatter(pos, dens, box)
    tottime += timeit.default_timer()

    print()
    print('TSC Scatter')
    print('===========')
    print(f'Time: {tottime/nrep:.4g} sec x {nrep} rep')
    print(f'Rate: {n/(tottime/nrep)/1e6:.4g} Mp/s')

    return

    tottime = -timeit.default_timer()
    for _ in range(nrep):
        abacusnbody.analysis.power_spectrum.numba_tsc_3D(pos, dens, box)
    tottime += timeit.default_timer()

    print()
    print('abacusutils TSC Scatter')
    print('=======================')
    print(f'Time: {tottime/nrep:.4g} sec x {nrep} rep')
    print(f'Rate: {n/(tottime/nrep)/1e6:.4g} Mp/s')


    dens1 = np.zeros((ngrid,ngrid,ngrid), dtype=np.float32)
    abacusnbody.analysis.power_spectrum.numba_tsc_3D(pos, dens1, box)
    dens2 = np.zeros((ngrid,ngrid,ngrid), dtype=np.float32)
    tsc.tsc_parallel(pos, dens2, box, nthread=nthread, npartition=npartition)

    print()
    print('Match')
    print('=====')
    print(f'Frac: {np.isclose(dens1, dens2, rtol=1e-3, atol=1e-5).mean()*100:.4g}%')

    # diff = np.absolute(dens1 - dens2)
    # tol = 1e-8 + 1e-3*np.absolute(dens2)
    # breakpoint()

if __name__ == '__main__':
    main()
