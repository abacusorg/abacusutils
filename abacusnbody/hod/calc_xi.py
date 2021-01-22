# allsims testhod rsd
import numpy as np
import os,sys
import Corrfunc
# from Corrfunc.mocks.DDrppi_mocks import DDrppi_mocks
# from Corrfunc.utils import convert_3d_counts_to_cf, convert_rp_pi_counts_to_wp
from Corrfunc.theory.DDrppi import DDrppi
from Corrfunc.theory import wp, xi


def calc_xirppi_fast(x, y, z, rpbins, pimax, pi_bin_size, lbox, Nthread, num_cells = 20):  # all r assumed to be in h-1 mpc units. 
    ND = float(len(x))

    DD_counts = DDrppi(1, Nthread, pimax, rpbins, x, y, z, 
        boxsize = lbox, periodic = True, max_cells_per_dim = 20)['npairs']
    DD_counts_new = np.array([np.sum(DD_counts[i:i+pi_bin_size]) for i in range(0, len(DD_counts), pi_bin_size)])
    DD_counts_new = DD_counts_new.reshape((len(rpbins) - 1, int(pimax/pi_bin_size)))

    RR_counts_new = np.zeros((len(rpbins) - 1, int(pimax/pi_bin_size)))
    for i in range(len(rpbins) - 1):
        RR_counts_new[i] = np.pi*(rpbins[i+1]**2 - rpbins[i]**2)*pi_bin_size / lbox**3 * ND**2 * 2
    xirppi = DD_counts_new / RR_counts_new - 1

    return xirppi

def calc_wp_fast(x, y, z, rpbins, pimax, lbox, Nthread, num_cells = 20):  # all r assumed to be in h-1 mpc units. 
    ND = float(len(x))

    DD_counts = DDrppi(1, Nthread, pimax, rpbins, x, y, z, 
        boxsize = lbox, periodic = True, max_cells_per_dim = 20)['npairs']
    DD_counts = DD_counts.reshape((len(rpbins) - 1, int(pimax)))

    RR_counts = np.zeros((len(rpbins) - 1, int(pimax)))
    for i in range(len(rpbins) - 1):
        RR_counts[i] = np.pi*(rpbins[i+1]**2 - rpbins[i]**2) / lbox**3 * ND**2 * 2
    xirppi = DD_counts / RR_counts - 1

    return 2*np.sum(xirppi, axis = 1)


