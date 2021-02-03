# allsims testhod rsd
import numpy as np
import os,sys
import Corrfunc
# from Corrfunc.mocks.DDrppi_mocks import DDrppi_mocks
# from Corrfunc.utils import convert_3d_counts_to_cf, convert_rp_pi_counts_to_wp
from Corrfunc.theory.DDrppi import DDrppi
from Corrfunc.theory import wp, xi


def calc_xirppi_fast(x1, y1, z1, rpbins, pimax, 
    pi_bin_size, lbox, Nthread, num_cells = 20, x2 = None, y2 = None, z2 = None):  # all r assumed to be in h-1 mpc units. 
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
    
    DD_counts = DDrppi(autocorr, Nthread, pimax, rpbins, x1, y1, z1, X2 = x2, Y2 = y2, Z2 = z2, 
        boxsize = lbox, periodic = True, max_cells_per_dim = 20)['npairs']
    DD_counts_new = np.array([np.sum(DD_counts[i:i+pi_bin_size]) for i in range(0, len(DD_counts), pi_bin_size)])
    DD_counts_new = DD_counts_new.reshape((len(rpbins) - 1, int(pimax/pi_bin_size)))

    # RR_counts_new = np.zeros((len(rpbins) - 1, int(pimax/pi_bin_size)))
    RR_counts_new = np.pi*(rpbins[1:]**2 - rpbins[:-1]**2)*pi_bin_size / lbox**3 * ND1 * ND2 * 2
    xirppi = DD_counts_new / RR_counts_new[:, None] - 1

    return xirppi

def calc_wp_fast(x1, y1, z1, rpbins, pimax, 
    lbox, Nthread, num_cells = 20, x2 = None, y2 = None, z2 = None):  # all r assumed to be in h-1 mpc units. 
    if not isinstance(pimax, int):
        raise ValueError("pimax needs to be an integer")

    ND1 = float(len(x1))
    if x2 is not None:
        ND2 = len(x2)
        autocorr = 0
    else:
        autocorr = 1
        ND2 = ND1

    DD_counts = DDrppi(autocorr, Nthread, pimax, rpbins, x1, y1, z1, X2 = x2, Y2 = y2, Z2 = z2, 
        boxsize = lbox, periodic = True, max_cells_per_dim = 20)['npairs']
    DD_counts = DD_counts.reshape((len(rpbins) - 1, int(pimax)))

    # RR_counts = np.zeros((len(rpbins) - 1, int(pimax)))
    # for i in range(len(rpbins) - 1):
    RR_counts = np.pi*(rpbins[1:]**2 - rpbins[:-1]**2) / lbox**3 * ND1 * ND2 * 2
    xirppi = DD_counts / RR_counts[:, None] - 1

    return 2*np.sum(xirppi, axis = 1)


