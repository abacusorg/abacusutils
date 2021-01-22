
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
import tracers.tracer_fun as MT

simname = "/AbacusSummit_base_c000_ph006"
savedir = "/mnt/marvin1/syuan/scratch/data_summit"+simname


# the subsampling curve for halos
def subsample_halos(m):
    x = np.log10(m)
    return 1.0/(1.0 + 10*np.exp(-(x - 11.2)*25)) # MT

def subsample_particles(m):
    x = np.log10(m)
    # return 1.0/(1.0 + np.exp(-(x - 13.5)*3))
    return 4/(200.0 + np.exp(-(x - 13.2)*6)) # MT


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

    # lrg
    LRG_design = [10**13.3, 10**14.4, 0.8, 1.0, 0.4]

    # elg
    p_max = 0.33;
    Q = 100.;
    M_cut = 10.**11.75;
    kappa = 1.;
    sigma = 0.58;
    M_1 = 10.**13.53;
    alpha = 1.;
    gamma = 4.12;
    A_s = 1.
    ELG_design = {
        'p_max': p_max,
        'Q': Q,
        'M_cut': M_cut,
        'kappa': kappa,
        'sigma': sigma,
        'M_1': M_1,
        'alpha': alpha,
        'gamma': gamma,
        'A_s': A_s
    }

    Ms = np.logspace(11, 16, 100)

    halos_subsampling = subsample_halos(Ms)
    particle_subsampling = subsample_particles(Ms)

    fig = pl.figure(figsize = (5, 4))
    pl.xlabel('$\log M$')
    pl.xscale('log')
    pl.ylabel('$N_g$')
    pl.yscale('log')
    pl.ylim(1e-4, 1e6)

    pl.plot(Ms, n_cen(Ms, LRG_design), 'r-', label = 'LRG cent')
    pl.plot(Ms, n_sat(Ms, LRG_design), 'r--', label = 'LRG sat')
    pl.plot(Ms, MT.N_cen_ELG_v1(Ms, **ELG_design), 'b-', label = 'ELG cent')
    pl.plot(Ms, MT.N_sat(Ms, **ELG_design), 'b--', label = 'ELG sat')
    pl.plot(Ms, halos_subsampling, 'k-', label = 'halo subsampling')
    pl.plot(Ms, Ms / 3131059264.330557 * halos_subsampling * particle_subsampling, 'k--', label = 'Np eff')

    pl.legend(loc = 'best')
    pl.tight_layout()
    fig.savefig("./plots/plot_subsample.pdf", dpi = 300)

    # compute mass function
    # nM()

    # # expected number of galaxies per chunk
    # Ng_exp(design_array)
