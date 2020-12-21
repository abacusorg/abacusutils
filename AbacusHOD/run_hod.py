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
from pathlib import Path

import numpy as np
from astropy.table import Table
import h5py
import asdf
import argparse
import multiprocessing
from multiprocessing import Pool
from itertools import repeat

from GRAND_HOD import gen_gal_catalog_rockstar_modified_subsampled as galcat

DEFAULTS = {}
DEFAULTS['sim_name'] = "AbacusSummit_base_c000_ph006"
DEFAULTS['z_mock'] = 0.5
DEFAULTS['scratch_dir'] = "/mnt/marvin1/syuan/scratch"
DEFAULTS['subsample_dir'] = "/mnt/marvin1/syuan/scratch/data_summit/"
DEFAULTS['sim_dir'] = "/mnt/gosling2/bigsims/"

DEFAULTS['newseed'] = 0

DEFAULTS['logM_cut'] = 13.3
DEFAULTS['logM1'] = 14.4
DEFAULTS['sigma'] = 0.8
DEFAULTS['alpha'] = 1.0
DEFAULTS['kappa'] = 0.4

DEFAULTS['s'] = 0
DEFAULTS['s_v'] = 0
DEFAULTS['alpha_c'] = 0
DEFAULTS['s_p'] = 0
DEFAULTS['s_r'] = 0
DEFAULTS['A'] = 0
DEFAULTS['Ae'] = 0
DEFAULTS['ic'] = 0.97


def main(sim_name, z_mock, scratch_dir, subsample_dir, sim_dir, want_rsd=False, want_ranks=False):
    
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
    halo_data = []
    particle_data = []

    # B.H. make into ASDF
    # load all the halo and particle data we need
    for echunk in range(params['numchunks']):

        newfile = h5py.File(subsample_dir / ('halos_xcom_%d_seed600_abacushod.h5'%echunk), 'r')
        maskedhalos = newfile['halos']

        # extracting the halo properties that we need
        halo_ids = np.array(maskedhalos["id"], dtype = int) # halo IDs
        halo_pos = maskedhalos["x_com"]/params['h'] # halo positions, Mpc
        halo_vels = maskedhalos['v_com'] # halo velocities, km/s
        halo_vrms = maskedhalos["sigmav3d_com"] # halo velocity dispersions, km/s
        halo_mass = maskedhalos['N']*params['Mpart'] # halo mass, Msun, 200b
        halo_deltac = maskedhalos['deltac'] # halo concentration
        halo_fenv = maskedhalos['fenv_binnorm'] # halo velocities, km/s
        halo_pstart = np.array(maskedhalos['npstartA'], dtype = int) # starting index of particles
        halo_pnum = np.array(maskedhalos['npoutA'], dtype = int) # number of particles 
        halo_multi = maskedhalos['multi_halos']
        halo_submask = np.array(maskedhalos['mask_subsample'], dtype = bool)
        halo_randoms = maskedhalos['randoms']
        halo_table = Table([halo_ids, halo_pos[:, 0], halo_pos[:, 1], halo_pos[:, 2], 
            halo_vels[:, 0], halo_vels[:, 1], halo_vels[:, 2], halo_vrms, halo_mass, halo_deltac, 
            halo_fenv, halo_pstart, halo_pnum, halo_multi, halo_submask, halo_randoms], 
            names=('id', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'vrms', 'mass', 'deltac', 'fenv', 
            'pstart', 'pnum', 'multi', 'submask', 'randoms'))
        halo_data += [halo_table]

        # extract particle data that we need
        newpart = h5py.File(subsample_dir / ('particles_xcom_%d_seed600_abacushod.h5'%echunk), 'r')
        subsample = newpart['particles']
        part_pos = subsample['pos'] / params['h']
        part_vel = subsample['vel']
        part_halomass = subsample['halo_mass'] / params['h'] # msun
        part_haloid = np.array(subsample['halo_id'], dtype = int)
        part_Np = subsample['Np'] # number of particles that end up in the halo
        part_subsample = subsample['downsample_halo']
        part_randoms = subsample['randoms']
        part_table = Table([part_pos[:, 0], part_pos[:, 1], part_pos[:, 2], 
            part_vel[:, 0], part_vel[:, 1], part_vel[:, 2], part_halomass, 
            part_haloid, part_Np, part_subsample, part_randoms],
            names = ('x', 'y', 'z', 'vx', 'vy', 'vz', 'hmass', 'hid', 'Np', 'subsample', 'randoms'))

        # if want_ranks:
        #     part_ranks = subsample['ranks']
        #     part_ranksv = subsample['ranksv']
        #     part_ranksp = subsample['ranksp']
        #     part_ranksr = subsample['ranksr']
        #     part_data_chunk += [part_ranks, part_ranksv, part_ranksp, part_ranksr]

        particle_data += [part_table]

    return halo_data, particle_data, params, mock_dir

# method for running one hod on one sim with one reseeding
def gen_gal_onesim_onehod(whichchunk, halo_data, particle_data, design, decorations, savedir, eseed, params):

    galcat.gen_gal_cat(whichchunk, halo_data, particle_data, design, decorations, params, savedir, whatseed=eseed, rsd=params['rsd'])


# B.H. is this function necessary?
def run_onebox(i):
    gen_gal_onesim_onehod(i, halo_data[i], particle_data[i], newdesign, newdecor, save_dir, newseed, params)
    return


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
    parser.add_argument('--want_ranks', help='Want velocity bias parameters', action='store_true')
    args = vars(parser.parse_args())

    # preload the simulation
    print("preloading simulation")
    halo_data, particle_data, params, mock_dir = main(**args)
    print("finished loading the data into memory")

    # B.H. I think the HOD parameters could be read from a yaml file or something
    parser.add_argument('--newseed', help='Random seed for generating the mock', type=int, default=DEFAULTS['newseed'])
    parser.add_argument('--logM_cut', help='HOD parameter', type=float, default=DEFAULTS['logM_cut'])
    parser.add_argument('--logM1', help='HOD parameter', type=float, default=DEFAULTS['logM1'])
    parser.add_argument('--sigma', help='HOD parameter', type=float, default=DEFAULTS['sigma'])
    parser.add_argument('--alpha', help='HOD parameter', type=float, default=DEFAULTS['alpha'])
    parser.add_argument('--kappa', help='HOD parameter', type=float, default=DEFAULTS['kappa'])
    parser.add_argument('--s', help='Extra HOD parameter', type=float, default=DEFAULTS['s'])
    parser.add_argument('--s_v', help='Extra HOD parameter', type=float, default=DEFAULTS['s_v'])
    parser.add_argument('--s_p', help='Extra HOD parameter', type=float, default=DEFAULTS['s_p'])
    parser.add_argument('--s_r', help='Extra HOD parameter', type=float, default=DEFAULTS['s_r'])
    parser.add_argument('--alpha_c', help='Extra HOD parameter', type=float, default=DEFAULTS['alpha_c'])
    parser.add_argument('--A', help='Extra HOD parameter', type=float, default=DEFAULTS['A'])
    parser.add_argument('--Ae', help='Extra HOD parameter', type=float, default=DEFAULTS['Ae'])
    parser.add_argument('--ic', help='Extra HOD parameter', type=float, default=DEFAULTS['ic'])
    args = vars(parser.parse_args())
    
    newdesign = {'M_cut': 10.**args['logM_cut'],
                 'M1': 10.**args['logM1'],
                 'sigma': args['sigma'],
                 'alpha': args['alpha'],
                 'kappa': args['kappa']}
    newdecor = {'s': args['s'],
                's_v': args['s_v'],
                's_p': args['s_p'],
                's_r': args['s_r'],
                'alpha_c': args['alpha_c'],
                'A': args['A'],
                'Ae': args['Ae'],
                'ic': args['ic']}

    # seeding for this run
    newseed = args['newseed']
    if newseed != 0:
        seed_string = "_%d"%newseed
    else:
        seed_string = ""

    if params['rsd']:
        rsd_string = "_rsd"
    else:
        rsd_string = ""

    # B.H. I think we don't need this (just put into header and have model_number)
    # only do this if this has not been done before (B.H. not sure what this means)
    M_cutn, M1n, sigman, alphan, kappan = map(newdesign.get, ('M_cut', 'M1', 'sigma', 'alpha', 'kappa'))
    sn, s_vn, alpha_cn, s_pn, s_rn, An, Aen = map(newdecor.get, ('s', 's_v', 'alpha_c', 's_p', 's_r', 'A', 'Ae'))    

    # B.H. does the naming need to be so convoluted? Can't we just put into a header?
    save_dir = mock_dir / ("rockstar_"+str(np.log10(M_cutn))[0:10]+
    "_"+str(np.log10(M1n))[0:10]+"_"+str(sigman)[0:6]+"_"+
    str(alphan)[0:6]+"_"+str(kappan)[0:6]+"_decor_"+str(sn)+
    "_"+str(s_vn)+"_"+str(alpha_cn)+"_"+str(s_pn)+"_"+str(s_rn)+
    "_"+str(An)+"_"+str(Aen)+rsd_string+seed_string)

    # create directories if not existing
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    start = time.time()
    # run_onebox(0)
    #gen_gal_onesim_onehod(0, halo_data, particle_data, newdesign, newdecor, save_dir, newseed, params)
    # multiprocess
    p = multiprocessing.Pool(17)
    p.map(run_onebox, range(params['numchunks']))
    #p.map(gen_gal_onesim_onehod, zip((i, halo_data[i], particle_data[i], newdesign, newdecor, save_dir, newseed, params) for i in range(params['numchunks'])))
    # p.starmap(gen_gal_onesim_onehod, zip(range(params['numchunks']), repeat(halo_data), repeat(particle_data), repeat(newdesign), repeat(newdecor), repeat(save_dir), repeat(newseed), repeat(params)))
    #p.close()
    #p.join()
    print("Done ", time.time() - start)
