# allsims testhod rsd
import numpy as np
import os, sys, time
import Corrfunc
# from Corrfunc.mocks.DDrppi_mocks import DDrppi_mocks
# from Corrfunc.utils import convert_3d_counts_to_cf, convert_rp_pi_counts_to_wp
# from Corrfunc.theory.DDrppi import 
from Corrfunc.theory import wp, xi, DDsmu, DDrppi

from halotools.mock_observables import tpcf_multipole, tpcf


def calc_xirppi_fast(x1, y1, z1, rpbins, pimax, 
    pi_bin_size, lbox, Nthread, num_cells = 20, x2 = None, y2 = None, z2 = None):  # all r assumed to be in h-1 mpc units. 
    start = time.time()
    if not isinstance(pimax, int):
        raise ValueError("pimax needs to be an integer")
    if not isinstance(pi_bin_size, int):
        raise ValueError("pi_bin_size needs to be an integer")
    if not pimax % pi_bin_size == 0:
        raise ValueError("pi_bin_size needs to be an integer divisor of pimax, current values are ", pi_bin_size, pimax)

    ND1 = float(len(x1))
    if x2 is not None:
        ND2 = len(x2)
        autocorr = 0
    else:
        autocorr = 1
        ND2 = ND1
    
    # single precision mode
    # to do: make this native 
    cf_start = time.time()
    rpbins = rpbins.astype(np.float32)
    pimax = np.float32(pimax)
    x1 = x1.astype(np.float32)
    y1 = y1.astype(np.float32)
    z1 = z1.astype(np.float32)
    lbox = np.float32(lbox)

    if autocorr == 1:    
        results = DDrppi(autocorr, Nthread, pimax, rpbins, x1, y1, z1,
            boxsize = lbox, periodic = True, max_cells_per_dim = num_cells)
        DD_counts = results['npairs']
    else:
        results = DDrppi(autocorr, Nthread, pimax, rpbins, x1, y1, z1, X2 = x2, Y2 = y2, Z2 = z2, 
            boxsize = lbox, periodic = True, max_cells_per_dim = num_cells)
        DD_counts = results['npairs']
    print("corrfunc took time ", time.time() - cf_start)

    DD_counts_new = np.array([np.sum(DD_counts[i:i+pi_bin_size]) for i in range(0, len(DD_counts), pi_bin_size)])
    DD_counts_new = DD_counts_new.reshape((len(rpbins) - 1, int(pimax/pi_bin_size)))

    # RR_counts_new = np.zeros((len(rpbins) - 1, int(pimax/pi_bin_size)))
    RR_counts_new = np.pi*(rpbins[1:]**2 - rpbins[:-1]**2)*pi_bin_size / lbox**3 * ND1 * ND2 * 2
    xirppi = DD_counts_new / RR_counts_new[:, None] - 1
    print("corrfunc took ", time.time() - start, "ngal ", len(x1))
    return xirppi


def calc_multipole_fast(x1, y1, z1, rpbins, 
    lbox, Nthread, num_cells = 20, x2 = None, y2 = None, z2 = None, orders = [0, 2, 4]):  # all r assumed to be in h-1 mpc units. 

    ND1 = float(len(x1))
    if x2 is not None:
        ND2 = len(x2)
        autocorr = 0
    else:
        autocorr = 1
        ND2 = ND1
    
    # single precision mode
    # to do: make this native 
    cf_start = time.time()
    rpbins = rpbins.astype(np.float32)
    x1 = x1.astype(np.float32)
    y1 = y1.astype(np.float32)
    z1 = z1.astype(np.float32)
    pos1 = np.array([x1, y1, z1]).T % lbox
    lbox = np.float32(lbox)

    # mu_bins = np.linspace(0, 1, 20)

    # if autocorr == 1: 
    #     xi_s_mu = s_mu_tpcf(pos1, rpbins, mu_bins, period = lbox, num_threads = Nthread)
    #     print("halotools ", xi_s_mu)
    # else:
    #     xi_s_mu = s_mu_tpcf(pos1, rpbins, mu_bins, sample2 = np.array([x2, y2, z2]).T % lbox, period = lbox, num_threads = Nthread)

    nbins_mu = 20
    if autocorr == 1: 
        results = DDsmu(autocorr, Nthread, rpbins, 1, nbins_mu, x1, y1, z1, periodic = True, boxsize = lbox, max_cells_per_dim = num_cells)
        DD_counts = results['npairs']
    else:
        results = DDsmu(autocorr, Nthread, rpbins, 1, nbins_mu, x1, y1, z1, X2 = x2, Y2 = y2, Z2 = z2, 
            periodic = True, boxsize = lbox, max_cells_per_dim = num_cells)
        DD_counts = results['npairs']
    DD_counts = DD_counts.reshape((len(rpbins) - 1, nbins_mu))

    mu_bins = np.linspace(0, 1, nbins_mu+1)
    RR_counts = 2*np.pi/3*(rpbins[1:, None]**3 - rpbins[:-1, None]**3)*(mu_bins[None, 1:] - mu_bins[None, :-1]) / lbox**3 * ND1 * ND2 * 2

    xi_s_mu = DD_counts / RR_counts - 1

    xi_array = []
    for neworder in orders:
        # print(neworder, rpbins, tpcf_multipole(xi_s_mu, mu_bins, order = neworder))
        xi_array += [tpcf_multipole(xi_s_mu, mu_bins, order=neworder)]
    xi_array = np.concatenate(xi_array)

    return xi_array


def calc_wp_fast(x1, y1, z1, rpbins, pimax, 
    lbox, Nthread, num_cells = 30, x2 = None, y2 = None, z2 = None):  # all r assumed to be in h-1 mpc units. 
    if not isinstance(pimax, int):
        raise ValueError("pimax needs to be an integer")

    ND1 = float(len(x1))
    if x2 is not None:
        ND2 = len(x2)
        autocorr = 0
    else:
        autocorr = 1
        ND2 = ND1

    # single precision mode
    # to do: make this native 
    cf_start = time.time()
    rpbins = rpbins.astype(np.float32)
    pimax = np.float32(pimax)
    x1 = x1.astype(np.float32)
    y1 = y1.astype(np.float32)
    z1 = z1.astype(np.float32)
    lbox = np.float32(lbox)

    if autocorr == 1:    
        print("sample size", len(x1))
        results = DDrppi(autocorr, Nthread, pimax, rpbins, x1, y1, z1,
            boxsize = lbox, periodic = True, max_cells_per_dim = num_cells)
        DD_counts = results['npairs']
    else:
        print("sample size", len(x1), len(x2))
        x2 = x2.astype(np.float32)
        y2 = y2.astype(np.float32)
        z2 = z2.astype(np.float32)
        results = DDrppi(autocorr, Nthread, pimax, rpbins, x1, y1, z1, X2 = x2, Y2 = y2, Z2 = z2, 
            boxsize = lbox, periodic = True, max_cells_per_dim = num_cells)
        DD_counts = results['npairs']
    print("corrfunc took time ", time.time() - cf_start)
    DD_counts = DD_counts.reshape((len(rpbins) - 1, int(pimax)))

    # RR_counts = np.zeros((len(rpbins) - 1, int(pimax)))
    # for i in range(len(rpbins) - 1):
    RR_counts = np.pi*(rpbins[1:]**2 - rpbins[:-1]**2) / lbox**3 * ND1 * ND2 * 2
    xirppi = DD_counts / RR_counts[:, None] - 1

    return 2*np.sum(xirppi, axis = 1)


