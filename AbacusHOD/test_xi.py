import numpy as np
import time
from astropy.table import Table
from Corrfunc.theory.DDrppi import DDrppi

rpbins = np.logspace(-1, 1.5, 9)
pimax = 30
pi_bin_size = 5
lbox = 2000

Nthread = 32

def calc_xirppi_fast(x, y, z, rpbins, pimax, pi_bin_size, lbox):  # all r assumed to be in h-1 mpc units. 
    ND = float(len(x))

    DD_counts = DDrppi(1, Nthread, pimax, rpbins, x, y, z, 
        boxsize = lbox, periodic = True, max_cells_per_dim = 20, verbose = True)['npairs']
    DD_counts_new = np.array([np.sum(DD_counts[i:i+pi_bin_size]) for i in range(0, len(DD_counts), pi_bin_size)])
    DD_counts_new = DD_counts_new.reshape((len(rpbins) - 1, int(pimax/pi_bin_size)))

    RR_counts_new = np.zeros((len(rpbins) - 1, int(pimax/pi_bin_size)))
    for i in range(len(rpbins) - 1):
        RR_counts_new[i] = np.pi*(rpbins[i+1]**2 - rpbins[i]**2)*pi_bin_size / lbox**3 * ND**2 * 2
    xirppi = DD_counts_new / RR_counts_new - 1

    return xirppi

path = "/mnt/marvin1/syuan/scratch/data_mocks_summit_new/galaxies_13.3_14.4_0.8_1.0_0.4_decor_0_1_0_0_0_0_0_0_0_0"
cents = Table.read(path+"/gals_cent.dat", format = 'ascii')
sats = Table.read(path+"/gals_sat.dat", format = 'ascii')

xfull = np.concatenate((cents['x_gal'], sats['x_gal']), axis = 0) # .astype(np.float32)
yfull = np.concatenate((cents['y_gal'], sats['y_gal']), axis = 0) # .astype(np.float32)
zfull = np.concatenate((cents['z_gal'], sats['z_gal']), axis = 0) # .astype(np.float32)
# posfull = np.stack((xfull, yfull, zfull)).T

for i in range(10):
    start = time.time()
    calc_xirppi_fast(xfull, yfull, zfull, rpbins, pimax, pi_bin_size, lbox)
    print(i, time.time() - start)