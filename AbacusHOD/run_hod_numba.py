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

from GRAND_HOD import gen_gal_catalog_rockstar_modified_subsampled_numba as galcat
from calc_xi import calc_xirppi

DEFAULTS = {}
DEFAULTS['sim_name'] = "AbacusSummit_base_c000_ph006"
DEFAULTS['z_mock'] = 0.5
DEFAULTS['scratch_dir'] = "/mnt/marvin1/syuan/scratch"
DEFAULTS['subsample_dir'] = "/mnt/marvin1/syuan/scratch/data_summit/"
DEFAULTS['sim_dir'] = "/mnt/gosling2/bigsims/"

DEFAULTS['logM_cut'] = 13.3
DEFAULTS['logM1'] = 14.4
DEFAULTS['sigma'] = 0.8
DEFAULTS['alpha'] = 1.0
DEFAULTS['kappa'] = 0.4

DEFAULTS['alpha_c'] = 0
DEFAULTS['alpha_s'] = 1

DEFAULTS['s'] = 0
DEFAULTS['s_v'] = 0
DEFAULTS['s_p'] = 0
DEFAULTS['s_r'] = 0
DEFAULTS['Acent'] = 0
DEFAULTS['Asat'] = 0
DEFAULTS['Bcent'] = 0
DEFAULTS['Bsat'] = 0
DEFAULTS['ic'] = 0.97


def staging(sim_name, z_mock, scratch_dir, subsample_dir, sim_dir, want_rsd=False, want_ranks=False):
    
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
    mock_dir = scratch_dir / scratch_name
    subsample_dir = Path(subsample_dir) / sim_name
    sim_dir = Path(sim_dir)

    # load header to read parameters
    halo_info_fns = list((sim_dir / sim_name / 'halos' / ('z%4.3f'%z_mock) / 'halo_info').glob('*.asdf'))
    f = asdf.open(halo_info_fns[0], lazy_load=True, copy_arrays=False)
    header = f['header']

    # constants
    params = {}
    params['z'] = z_mock
    params['h'] = header['H0']/100.
    params['Lbox'] = header['BoxSize']/params['h'] # Mpc, box size
    params['Mpart'] = header['ParticleMassHMsun']/params['h']  # Msun, mass of each particle
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

        newfile = h5py.File(subsample_dir / ('halos_xcom_%d_seed600_abacushod.h5'%echunk), 'r')
        allhalos = newfile['halos']
        mask = np.array(allhalos['mask_subsample'], dtype = bool)
        maskedhalos = allhalos[mask]

        # extracting the halo properties that we need
        halo_ids = np.array(maskedhalos["id"], dtype = int) # halo IDs
        halo_pos = maskedhalos["x_com"]/params['h'] # halo positions, Mpc
        halo_vels = maskedhalos['v_com'] # halo velocities, km/s
        halo_vel_dev = maskedhalos["randoms_gaus_vrms"] # halo velocity dispersions, km/s
        halo_mass = maskedhalos['N']*params['Mpart'] # halo mass, Msun, 200b
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
        newpart = h5py.File(subsample_dir / ('particles_xcom_%d_seed600_abacushod.h5'%echunk), 'r')
        subsample = newpart['particles']
        part_pos = subsample['pos'] / params['h']
        part_vel = subsample['vel']
        part_hvel = subsample['halo_vel']
        part_halomass = subsample['halo_mass'] / params['h'] # msun
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

    halo_data = [hpos, hvel, hmass, hid, hmultis, hrandoms, hveldev, hdeltac, hfenv]
    particle_data = [ppos, pvel, phvel, phmass, phid, pNp, psubsampling, prandoms, pdeltac, pfenv]
    if want_ranks:
        particle_data += [p_ranks, p_ranksv, p_ranksp, p_ranksr]

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
    parser.add_argument('--want_rsd', help='Want redshift space distortions', action='store_true')
    parser.add_argument('--want_ranks', help='Want extended satellite parameters', action='store_true')
    args = vars(parser.parse_args())

    # preload the simulation
    print("preloading simulation")
    halo_data, particle_data, params, mock_dir = staging(**args)
    print("finished loading the data into memory")

    # B.H. I think the HOD parameters could be read from a yaml file or something
    parser.add_argument('--logM_cut', help='base HOD parameter (Zheng+2007)', type=float, default=DEFAULTS['logM_cut'])
    parser.add_argument('--logM1', help='base HOD parameter (Zheng+2007)', type=float, default=DEFAULTS['logM1'])
    parser.add_argument('--sigma', help='base HOD parameter (Zheng+2007)', type=float, default=DEFAULTS['sigma'])
    parser.add_argument('--alpha', help='base HOD parameter (Zheng+2007)', type=float, default=DEFAULTS['alpha'])
    parser.add_argument('--kappa', help='base HOD parameter (Zheng+2007)', type=float, default=DEFAULTS['kappa'])
    parser.add_argument('--alpha_c', help='central velocity bias parameter', type=float, default=DEFAULTS['alpha_c'])
    parser.add_argument('--alpha_s', help='satellite velocity bias parameter', type=float, default=DEFAULTS['alpha_s'])
    parser.add_argument('--s', help='satellite distribution parameter', type=float, default=DEFAULTS['s'])
    parser.add_argument('--s_v', help='satellite distribution parameter', type=float, default=DEFAULTS['s_v'])
    parser.add_argument('--s_p', help='satellite distribution parameter', type=float, default=DEFAULTS['s_p'])
    parser.add_argument('--s_r', help='satellite distribution parameter', type=float, default=DEFAULTS['s_r'])
    parser.add_argument('--Acent', help='Assembly bias parameter for centrals', type=float, default=DEFAULTS['Acent'])
    parser.add_argument('--Asat', help='Assembly bias parameter for satellites', type=float, default=DEFAULTS['Asat'])
    parser.add_argument('--Bcent', help='Environmental bias parameter for centrals', type=float, default=DEFAULTS['Bcent'])
    parser.add_argument('--Bsat', help='Environmental bias parameter for satellites', type=float, default=DEFAULTS['Bsat'])
    parser.add_argument('--ic', help='incompleteness factor', type=float, default=DEFAULTS['ic'])
    args = vars(parser.parse_args())
    
    newdesign = {'M_cut': 10.**args['logM_cut'],
                 'M1': 10.**args['logM1'],
                 'sigma': args['sigma'],
                 'alpha': args['alpha'],
                 'kappa': args['kappa']}
    newdecor = {'alpha_c': args['alpha_c'],
                'alpha_s': args['alpha_s'],
                's': args['s'],
                's_v': args['s_v'],
                's_p': args['s_p'],
                's_r': args['s_r'],
                'Acent': args['Acent'],
                'Asat': args['Asat'],
                'Bcent': args['Bcent'],
                'Bsat': args['Bsat'],
                'ic': args['ic']}


    # throw away run for jit to compile
    cent_pos, cent_vel, cent_mass, cent_id, sat_pos, sat_vel, sat_mass, sat_id = \
    galcat.gen_gal_cat(halo_data, particle_data, newdesign, newdecor, params, enable_ranks = args['want_ranks'], 
        rsd = args['want_rsd'], write_to_disk = True, savedir = mock_dir)

    # run the fit 10 times for timing 
    for i in range(10):
        start = time.time()
        galcat.gen_gal_cat(halo_data, particle_data, newdesign, newdecor, params, enable_ranks = args['want_ranks'], 
            rsd = args['want_rsd'], write_to_disk = False)
        print("Done iteration ", i, "took time ", time.time() - start)
    # multiprocess
    # p = multiprocessing.Pool(17)
    # p.map(run_onebox, range(params['numchunks']))
    #p.map(gen_gal_onesim_onehod, zip((i, halo_data[i], particle_data[i], newdesign, newdecor, save_dir, newseed, params) for i in range(params['numchunks'])))
    # p.starmap(gen_gal_onesim_onehod, zip(range(params['numchunks']), repeat(halo_data), repeat(particle_data), repeat(newdesign), repeat(newdecor), repeat(save_dir), repeat(newseed), repeat(params)))
    #p.close()
    #p.join()
