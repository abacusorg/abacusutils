
import numpy as np
from astropy.table import Table
from math import erfc
import h5py
from scipy import special

import copy
import os, sys
import shutil
import random
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as pl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from matplotlib import rc, rcParams
rcParams.update({'font.size': 10})

from abacus_halo_catalog import AbacusHaloCatalog

simname = "/AbacusSummit_base_c000_ph006"
savedir = "/mnt/marvin1/syuan/scratch/data_summit"+simname


# the subsampling curve for halos
def subsample_halos(m):
    x = np.log10(m)
    return 1.0/(1.0 + 0.1*np.exp(-(x - 13.3)*4))

def subsample_particles(m):
    x = np.log10(m)
    # return 1.0/(1.0 + np.exp(-(x - 13.5)*3))
    return 4/(200.0 + np.exp(-(x - 13.7)*6))


def n_cen(M_in, design_array, m_cutoff = 1e12): 

    M_cut, M1, sigma, alpha, kappa = \
    design_array[0], design_array[1], design_array[2], design_array[3], design_array[4]

    return 0.5*special.erfc(np.log(M_cut/M_in)/(2**.5*sigma))

def n_sat(M_in, design_array, m_cutoff = 1e12): 

    M_cut, M1, sigma, alpha, kappa = \
    design_array[0], design_array[1], design_array[2], design_array[3], design_array[4]


    return ((M_in - kappa*M_cut)/M1)**alpha*0.5*special.erfc(np.log(M_cut/M_in)/(2**.5*sigma))

def nM():
    Mbins = np.logspace(12, 16, 101)
    nhalos = 0
    for i in range(34):
        # load the halo catalog chunk
        print("loading catalog", i)
        cat = AbacusHaloCatalog(
            '/mnt/store2/bigsims/AbacusSummit'+simname+'/halos/z0.500/halo_info/halo_info_'\
            +str(i).zfill(3)+'.asdf')
        halos = cat.halos
        header = cat.header
        Mpart = header['ParticleMassHMsun'] # msun / h 
        H0 = header['H0']
        h = H0/100.0

        halos_N = halos['N']
        halos_M = halos['N']*Mpart / h

        nhalos_new, medges = np.histogram(halos_M, Mbins)
        nhalos += nhalos_new
    np.savez("./data_NM_halos", NM = nhalos, Mbins = Mbins)


def Ng_exp(design_array):
    nm_file = np.load("./data_NM_halos.npz")
    mbins = nm_file['Mbins']
    mmids = 0.5*(mbins[1:] + mbins[:-1])
    nm = nm_file['NM']

    mmask = mmids > 2e12

    ncens = n_cen(mmids[mmask], design_array)
    nsats = n_sat(mmids[mmask], design_array)

    print("number of centrals and satellites per chunk", np.sum(nm[mmask] * ncens)/34, np.sum(nm[mmask] * nsats)/34)


if __name__ == "__main__":

    design_array = [10**13.3, 10**14.4, 0.8, 1.0, 0.4]
    print(design_array)

    Ms = np.logspace(12, 16, 100)

    halos_subsampling = subsample_halos(Ms)
    particle_subsampling = subsample_particles(Ms)

    fig = pl.figure(figsize = (5, 4))
    pl.xlabel('$\log M$')
    pl.xscale('log')
    pl.ylabel('$N_g$')
    pl.yscale('log')
    pl.ylim(1e-4, 1e6)

    pl.plot(Ms, n_cen(Ms, design_array), label = 'cent')
    pl.plot(Ms, n_sat(Ms, design_array), label = 'sat')
    pl.plot(Ms, Ms / 3131059264.330557 * halos_subsampling * particle_subsampling, 'k-', label = 'Np eff')
    pl.plot(Ms, halos_subsampling, 'k--', label = 'halo subsampling')

    pl.legend(loc = 'best')
    pl.tight_layout()
    fig.savefig("./plots/plot_subsample.pdf", dpi = 300)

    # compute mass function
    # nM()

    # expected number of galaxies per chunk
    Ng_exp(design_array)
