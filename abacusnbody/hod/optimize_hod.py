#!/usr/bin/env python3
'''
This is a script for generating `vanilla' HOD mock catalogs.

Usage
-----
$ ./run_hod.py --help
'''


import os
import glob
import time
import timeit
from pathlib import Path

import numpy as np
from astropy.table import Table, vstack
import h5py
import asdf
import argparse
import multiprocessing
from multiprocessing import Pool
from itertools import repeat
from stochopy import MonteCarlo, Evolutionary

from GRAND_HOD import gen_gal_catalog_rockstar_modified_subsampled_numba as galcat
from calc_xi import calc_xirppi_fast, calc_wp_fast

DEFAULTS = {}
DEFAULTS['sim_name'] = "AbacusSummit_base_c000_ph006"
DEFAULTS['z_mock'] = 0.5
DEFAULTS['scratch_dir'] = "/mnt/marvin1/syuan/scratch"
DEFAULTS['subsample_dir'] = "/mnt/marvin1/syuan/scratch/data_summit/"
DEFAULTS['sim_dir'] = "/mnt/gosling2/bigsims/"
DEFAULTS['want_rsd'] = True
DEFAULTS['want_ranks'] = False
DEFAULTS['want_LRG'] = True
DEFAULTS['want_ELG'] = False
DEFAULTS['want_QSO'] = False

# LRG HOD
LRG_HOD = {}
LRG_HOD['logM_cut'] = 13.38
LRG_HOD['logM1'] = 14.5
LRG_HOD['sigma'] = 0.25
LRG_HOD['alpha'] = 1.11
LRG_HOD['kappa'] = 0.03
# velocity bias
LRG_HOD['alpha_c'] = 0
LRG_HOD['alpha_s'] = 1
# satellite extensions, assembly bias, and incompleteness
LRG_HOD['s'] = 0
LRG_HOD['s_v'] = 0
LRG_HOD['s_p'] = 0
LRG_HOD['s_r'] = 0
LRG_HOD['Acent'] = 0
LRG_HOD['Asat'] = 0
LRG_HOD['Bcent'] = 0
LRG_HOD['Bsat'] = 0
LRG_HOD['ic'] = 0.97

# ELG HOD
ELG_HOD = {}
ELG_HOD['p_max'] = 0.33
ELG_HOD['Q'] = 100.
ELG_HOD['logM_cut'] = 11.75
ELG_HOD['kappa'] = 1.
ELG_HOD['sigma'] = 0.58
ELG_HOD['logM1'] = 13.53
ELG_HOD['alpha'] = 1.
ELG_HOD['gamma'] = 4.12
ELG_HOD['A_s'] = 1.

# QSO HOD
QSO_HOD = {}
QSO_HOD['p_max'] = 0.33
QSO_HOD['logM_cut'] = 12.21
QSO_HOD['kappa'] = 1.0
QSO_HOD['sigma'] = 0.56
QSO_HOD['logM1'] = 13.94
QSO_HOD['alpha'] = 0.4
QSO_HOD['A_s'] = 1.


def staging(sim_name, z_mock, scratch_dir, subsample_dir, sim_dir, want_rsd=False, want_ranks=False, 
    want_LRG = True, want_ELG = False, want_QSO = False):
    # flag for redshift space distortions
    if want_rsd:
        rsd_string = "_rsd"
    else:
        rsd_string = ""

    # folder name where data are saved
    scratch_name = "data_mocks_summit_new"
    scratch_name += rsd_string

    # all paths relevant for mock generation
    scratch_dir = Path(scratch_dir)
    mock_dir = scratch_dir / scratch_name / sim_name / ('z%4.3f'%z_mock)
    # create mock_dir if not created
    if not mock_dir.exists():
        mock_dir.mkdir(parents = True)
    subsample_dir = Path(subsample_dir) / sim_name / ('z%4.3f'%z_mock)
    sim_dir = Path(sim_dir)

    # load header to read parameters
    halo_info_fns = list((sim_dir / sim_name / 'halos' / ('z%4.3f'%z_mock) / 'halo_info').glob('*.asdf'))
    f = asdf.open(halo_info_fns[0], lazy_load=True, copy_arrays=False)
    header = f['header']

    # constants
    params = {}
    params['z'] = z_mock
    params['h'] = header['H0']/100.
    params['Lbox'] = header['BoxSize'] # Mpc / h, box size
    params['Mpart'] = header['ParticleMassHMsun']  # Msun / h, mass of each particle
    params['velz2kms'] = header['VelZSpace_to_kms']/params['Lbox']
    params['numchunks'] = len(halo_info_fns)
    params['rsd'] = want_rsd

    # list holding individual chunks
    # halo_data = Table()
    # particle_data = Table()
    hpos = np.empty((1, 3))
    hvel = np.empty((1, 3))
    hmass = np.array([])
    hid = np.array([])
    hmultis = np.array([])
    hrandoms = np.array([])
    hveldev = np.array([])
    hdeltac = np.array([])
    hfenv = np.array([])

    ppos = np.empty((1, 3))
    pvel = np.empty((1, 3))
    phvel = np.empty((1, 3))
    phmass = np.array([])
    phid = np.array([])
    pNp = np.array([])
    psubsampling = np.array([])
    prandoms = np.array([])
    pdeltac = np.array([])
    pfenv = np.array([])

    # ranks
    if want_ranks:
        p_ranks = np.array([])
        p_ranksv = np.array([])
        p_ranksp = np.array([])
        p_ranksr = np.array([])

    # B.H. make into ASDF
    # load all the halo and particle data we need
    for echunk in range(params['numchunks']):
        print(echunk)
        if (not want_ELG) & (not want_QSO):
            halofilename = subsample_dir / ('halos_xcom_%d_seed600_abacushod.h5'%echunk)
            particlefilename = subsample_dir / ('particles_xcom_%d_seed600_abacushod.h5'%echunk)
        else:
            halofilename = subsample_dir / ('halos_xcom_%d_seed600_abacushodMT.h5'%echunk)
            particlefilename = subsample_dir / ('particles_xcom_%d_seed600_abacushodMT.h5'%echunk)            

        newfile = h5py.File(halofilename, 'r')
        allhalos = newfile['halos']
        mask = np.array(allhalos['mask_subsample'], dtype = bool)
        maskedhalos = allhalos[mask]

        # extracting the halo properties that we need
        halo_ids = np.array(maskedhalos["id"], dtype = int) # halo IDs
        halo_pos = maskedhalos["x_com"] # halo positions, Mpc / h
        halo_vels = maskedhalos['v_com'] # halo velocities, km/s
        halo_vel_dev = maskedhalos["randoms_gaus_vrms"] # halo velocity dispersions, km/s
        halo_mass = maskedhalos['N']*params['Mpart'] # halo mass, Msun / h, 200b
        halo_deltac = maskedhalos['deltac_rank'] # halo concentration
        halo_fenv = maskedhalos['fenv_rank'] # halo velocities, km/s
        halo_pstart = np.array(maskedhalos['npstartA'], dtype = int) # starting index of particles
        halo_pnum = np.array(maskedhalos['npoutA'], dtype = int) # number of particles 
        halo_multi = maskedhalos['multi_halos']
        halo_submask = np.array(maskedhalos['mask_subsample'], dtype = bool)
        halo_randoms = maskedhalos['randoms']
        # new_halo_table = Table([halo_ids, halo_pos[:, 0], halo_pos[:, 1], halo_pos[:, 2], 
        #     halo_vels[:, 0], halo_vels[:, 1], halo_vels[:, 2], halo_vrms, halo_mass, halo_deltac, 
        #     halo_fenv, halo_pstart, halo_pnum, halo_multi, halo_submask, halo_randoms], 
        #     names=('id', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'vrms', 'mass', 'deltac', 'fenv', 
        #     'pstart', 'pnum', 'multi', 'submask', 'randoms'))
        # halo_data = vstack([halo_data, new_halo_table])
        hpos = np.concatenate((hpos, halo_pos))
        hvel = np.concatenate((hvel, halo_vels))
        hmass = np.concatenate((hmass, halo_mass))
        hid = np.concatenate((hid, halo_ids))
        hmultis = np.concatenate((hmultis, halo_multi))
        hrandoms = np.concatenate((hrandoms, halo_randoms))
        hveldev = np.concatenate((hveldev, halo_vel_dev))
        hdeltac = np.concatenate((hdeltac, halo_deltac))
        hfenv = np.concatenate((hfenv, halo_fenv))

        # extract particle data that we need
        newpart = h5py.File(particlefilename, 'r')
        subsample = newpart['particles']
        part_pos = subsample['pos']
        part_vel = subsample['vel']
        part_hvel = subsample['halo_vel']
        part_halomass = subsample['halo_mass'] # msun / h
        part_haloid = np.array(subsample['halo_id'], dtype = int)
        part_Np = subsample['Np'] # number of particles that end up in the halo
        part_subsample = subsample['downsample_halo']
        part_randoms = subsample['randoms']
        part_deltac = subsample['halo_deltac']
        part_fenv = subsample['halo_fenv']
        # new_part_table = Table([part_pos[:, 0], part_pos[:, 1], part_pos[:, 2], 
        #     part_vel[:, 0], part_vel[:, 1], part_vel[:, 2], part_halomass, 
        #     part_haloid, part_Np, part_subsample, part_randoms],
        #     names = ('x', 'y', 'z', 'vx', 'vy', 'vz', 'hmass', 'hid', 'Np', 'subsample', 'randoms'))

        if want_ranks:
            part_ranks = subsample['ranks']
            part_ranksv = subsample['ranksv']
            part_ranksp = subsample['ranksp']
            part_ranksr = subsample['ranksr']
            p_ranks = np.concatenate((p_ranks, part_ranks))
            p_ranksv = np.concatenate((p_ranksv, part_ranksv))
            p_ranksp = np.concatenate((p_ranksp, part_ranksp))
            p_ranksr = np.concatenate((p_ranksr, part_ranksr))

        # #     part_data_chunk += [part_ranks, part_ranksv, part_ranksp, part_ranksr]
        # particle_data = vstack([particle_data, new_part_table])
        ppos = np.concatenate((ppos, part_pos))
        pvel = np.concatenate((pvel, part_vel))
        phvel = np.concatenate((phvel, part_hvel))
        phmass = np.concatenate((phmass, part_halomass))
        phid = np.concatenate((phid, part_haloid))
        pNp = np.concatenate((pNp, part_Np))
        psubsampling = np.concatenate((psubsampling, part_subsample))
        prandoms = np.concatenate((prandoms, part_randoms))
        pdeltac = np.concatenate((pdeltac, part_deltac))
        pfenv = np.concatenate((pfenv, part_fenv))

    hpos = hpos[1:]
    hvel = hvel[1:]
    ppos = ppos[1:]
    pvel = pvel[1:]

    halo_data = {"hpos": hpos, 
                 "hvel": hvel, 
                 "hmass": hmass, 
                 "hid": hid, 
                 "hmultis": hmultis, 
                 "hrandoms": hrandoms, 
                 "hveldev": hveldev, 
                 "hdeltac": hdeltac, 
                 "hfenv": hfenv}
    pweights = 1/pNp/psubsampling
    particle_data = {"ppos": ppos, 
                     "pvel": pvel, 
                     "phvel": phvel, 
                     "phmass": phmass, 
                     "phid": phid, 
                     "pweights": pweights, 
                     "prandoms": prandoms, 
                     "pdeltac": pdeltac, 
                     "pfenv": pfenv}
    if want_ranks:
        particle_data['pranks'] = p_ranks
        particle_data['pranksv'] = p_ranksv
        particle_data['pranksp'] = p_ranksp
        particle_data['pranksr'] = p_ranksr
    else:
        particle_data['pranks'] = np.ones(len(phmass))
        particle_data['pranksv'] =  np.ones(len(phmass))
        particle_data['pranksp'] =  np.ones(len(phmass))
        particle_data['pranksr'] =  np.ones(len(phmass))
        
    return halo_data, particle_data, params, mock_dir


# # B.H. is this function necessary?
# def run_onebox(i):
#     gen_gal_onesim_onehod(i, halo_data[i], particle_data[i], newdesign, newdecor, save_dir, newseed, params)
#     return


class ArgParseFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    pass

if __name__ == "__main__":


    # parsing arguments
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=ArgParseFormatter)
    parser.add_argument('--sim_name', help='Simulation name', default=DEFAULTS['sim_name'])
    parser.add_argument('--z_mock', help='Redshift of the mock', type=float, default=DEFAULTS['z_mock'])
    parser.add_argument('--scratch_dir', help='Scratch directory', default=DEFAULTS['scratch_dir'])
    parser.add_argument('--subsample_dir', help='Particle subsample directory', default=DEFAULTS['subsample_dir'])
    parser.add_argument('--sim_dir', help='Simulation directory', default=DEFAULTS['sim_dir'])
    parser.add_argument('--want_rsd', help='Want redshift space distortions', default=DEFAULTS['want_rsd'])
    parser.add_argument('--want_ranks', help='Want extended satellite parameters', default=DEFAULTS['want_ranks'])
    parser.add_argument('--want_LRG', help='Want LRGs', default=DEFAULTS['want_LRG'])
    parser.add_argument('--want_ELG', help='Want ELGs', default=DEFAULTS['want_ELG'])
    parser.add_argument('--want_QSO', help='Want QSOs', default=DEFAULTS['want_QSO'])
    args = vars(parser.parse_args())

    # preload the simulation
    print("preloading simulation")
    halo_data, particle_data, params, mock_dir = staging(**args)
    print("finished loading the data into memory")

    # load hong wp data, the cosmology is a little wrong
    hong_wp_data = np.loadtxt("../../s3PCF_fenv/hong_data_final/wp_cmass_final_finebins_z0.46-0.60")
    rp_hong = hong_wp_data[:, 0] # h-1 mpc
    wp_hong = hong_wp_data[:, 1]
    rwp_hong = rp_hong * wp_hong
    rp_hong_log = np.log10(rp_hong)
    delta_rp_hong = 0.125 # h-1 mpc
    # courser bins 
    nbins_trans = len(rp_hong)
    rp_bins_log = np.linspace(np.min(rp_hong_log) - delta_rp_hong/2, 
                              np.max(rp_hong_log) + delta_rp_hong/2, nbins_trans + 1)
    rpbins = 10**rp_bins_log # h-1 mpc   
     
    # rpbins = np.logspace(-1, 1.5, 9)
    pimax = 30
    pi_bin_size = 5

    # load covariance matrix
    hong_wp_covmat = np.loadtxt("../../s3PCF_fenv/hong_data_final/wpcov_cmass_final_finebins_z0.46-0.60")
    hong_rwp_covmat = np.zeros(np.shape(hong_wp_covmat))
    for i in range(np.shape(hong_wp_covmat)[0]):
        for j in range(np.shape(hong_wp_covmat)[1]):
            hong_rwp_covmat[i, j] = hong_wp_covmat[i, j]*rp_hong[i]*rp_hong[j]
    hong_rwp_covmat_inv = np.linalg.inv(hong_rwp_covmat)
    hong_rwp_covmat_inv_short = np.linalg.inv(hong_rwp_covmat)[2:, 2:]

    # run the fit 10 times for timing 
    def mylikelihood(hod_array):
        LRG_HOD['logM_cut'] = hod_array[0]
        LRG_HOD['logM1'] = hod_array[1]
        LRG_HOD['sigma'] = hod_array[2]
        LRG_HOD['alpha'] = hod_array[3]
        LRG_HOD['kappa'] = hod_array[4]

        start = time.time()
        cent_pos, cent_vel, cent_mass, cent_id, cent_type, sat_pos, sat_vel, sat_mass, sat_id, sat_type = \
        galcat.gen_gal_cat(halo_data, particle_data, LRG_HOD, ELG_HOD, QSO_HOD, 
            params, enable_ranks = args['want_ranks'], rsd = args['want_rsd'], 
            want_LRG = args['want_LRG'], want_ELG = args['want_ELG'], want_QSO = args['want_QSO'],
            write_to_disk = False)
        print("Done mock took time ", time.time() - start)
        
        start = time.time()
        pos_full = np.concatenate((cent_pos, sat_pos), axis = 0)
        print("concatenate took time ", time.time() - start)
        start = time.time()
        wp = calc_wp_fast(pos_full[:, 0], pos_full[:, 1], pos_full[:, 2], rpbins, pimax, params['Lbox'], 32)
        print("wp took time ", time.time() - start)

        delta_wp = rp_hong*wp - rwp_hong

        logl_xi = np.dot(delta_wp[2:], np.dot(hong_rwp_covmat_inv_short, delta_wp[2:]))
        print(hod_array, logl_xi)
        return logl_xi

    init_params = np.array([13.38, 14.5, 0.25, 1.11, 0.03]) 
    n_dim = len(init_params)
    lower = np.array([12.5, 13.5, 0.01,   0.8,  0.0])
    upper = np.array([14,   14.8,    1,   1.4,  1.0])
    popsize = 4 + np.floor(20.*np.log(n_dim))
    max_iter = 2000
    ea = Evolutionary(mylikelihood, lower = lower, upper = upper, popsize = popsize, max_iter = max_iter)
    xopt, gfit = ea.optimize(solver = "cmaes")
    print(xopt)
    print(gfit)

    print("Done optimizing")


    plot_wp(pos_full[:, 0], pos_full[:, 1], pos_full[:, 2], rpbins, pimax, params['Lbox'])

    # multiprocess
    # p = multiprocessing.Pool(17)
    # p.map(run_onebox, range(params['numchunks']))
    #p.map(gen_gal_onesim_onehod, zip((i, halo_data[i], particle_data[i], newdesign, newdecor, save_dir, newseed, params) for i in range(params['numchunks'])))
    # p.starmap(gen_gal_onesim_onehod, zip(range(params['numchunks']), repeat(halo_data), repeat(particle_data), repeat(newdesign), repeat(newdecor), repeat(save_dir), repeat(newseed), repeat(params)))
    #p.close()
    #p.join()
