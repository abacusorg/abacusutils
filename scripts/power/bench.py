#!/usr/bin/env python3

import timeit

import click
import numba
import numpy as np

from abacusnbody.analysis.power_spectrum import (
    # calc_pk_from_deltak,
    calc_power,
    # get_field_fft,
    # get_k_mu_edges,
    # normalize_field,
    # get_field
)

@click.command()
@click.argument('nmesh', default=256)
@click.option('-r', '--nrep', default=3)
@click.option('-t', '--nthread', default=1)
def main(**kwargs):
    nmesh = kwargs['nmesh']
    nrep = kwargs['nrep']
    nthread = kwargs['nthread']
    print(kwargs)

    rng = np.random.default_rng(300)

    # specifications of the power spectrum computation
    paste = "TSC"
    compensated = interlaced = False
    nbins_mu = 1
    logk = False
    kmax = np.pi*nmesh
    nbins_k = 100

    # create synthetic particles
    Lbox = 1.
    N = 10**7
    pos = rng.random((N,3), dtype='f4')

    numba.set_num_threads(nthread)
    # compute power
    calc_power(pos, Lbox, nbins_k, nbins_mu, kmax, logk,
            paste, nmesh, compensated, interlaced, #poles=[0,2],
            nthread=nthread
            )

    t = -timeit.default_timer()
    for _ in range(nrep):
        # field_fft = get_field_fft(pos, Lbox, nmesh, paste, None, None, compensated, interlaced, nthread=nthread)
        # k_bin_edges, mu_bin_edges = get_k_mu_edges(Lbox, kmax, nbins_k, nbins_mu, logk)
        # p3d, N3d, binned_poles, Npoles, k_avg = calc_pk_from_deltak(field_fft, Lbox, k_bin_edges, mu_bin_edges, None, None, nthread=nthread)
        calc_power(pos, Lbox, nbins_k, nbins_mu, kmax, logk,
                paste, nmesh, compensated, interlaced,
                nthread=nthread
                )
    t += timeit.default_timer()
    print(f'Time: {t/nrep:.4g} sec x {nrep} rep')
    print(f'Rate: {N/(t/nrep)/1e6:.4g} M part/sec')

if __name__ == '__main__':
    main()
