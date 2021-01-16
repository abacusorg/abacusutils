#!/usr/bin/env python3
'''
This is a script for generating HOD mock catalogs.

Usage
-----
$ ./run_hod_numba.py --help
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
from calc_xi import calc_xirppi_fast



class AbacusHOD:
    def __init__(self, sim_name, simdir, scratchdir, subsampledir, z, 
        ranks = False, lrg = True, elg = False, qso = False):

        self.DEFAULTS = {}
        self.DEFAULTS['sim_name'] = sim_name # "AbacusSummit_base_c000_ph006"
        self.DEFAULTS['z_mock'] = z
        self.DEFAULTS['scratch_dir'] = scratchdir # "/mnt/marvin1/syuan/scratch/data_mocks_summit_new"
        self.DEFAULTS['subsample_dir'] = subsampledir # "/mnt/marvin1/syuan/scratch/data_summit/"
        self.DEFAULTS['sim_dir'] = simdir # "/mnt/gosling2/bigsims/"
        self.DEFAULTS['want_ranks'] = ranks
        self.DEFAULTS['want_LRG'] = lrg
        self.DEFAULTS['want_ELG'] = elg
        self.DEFAULTS['want_QSO'] = qso

        self.halo_data, self.particle_data, self.params, self.mock_dir = self.staging()


    def staging(self):

        # all paths relevant for mock generation
        scratch_dir = Path(self.DEFAULTS['scratch_dir'])
        simname = Path(self.DEFAULTS['sim_name'])
        sim_dir = Path(self.DEFAULTS['sim_dir'])
        mock_dir = scratch_dir / simname / ('z%4.3f'%self.DEFAULTS['z_mock'])
        # create mock_dir if not created
        if not mock_dir.exists():
            mock_dir.mkdir(parents = True)
        subsample_dir = \
        Path(self.DEFAULTS['subsample_dir']) / simname / ('z%4.3f'%self.DEFAULTS['z_mock'])

        # load header to read parameters
        halo_info_fns = \
        list((sim_dir / simname / 'halos' / ('z%4.3f'%self.DEFAULTS['z_mock']) / 'halo_info').glob('*.asdf'))
        f = asdf.open(halo_info_fns[0], lazy_load=True, copy_arrays=False)
        header = f['header']

        # constants
        params = {}
        params['z'] = self.DEFAULTS['z_mock']
        params['h'] = header['H0']/100.
        params['Lbox'] = header['BoxSize'] # Mpc / h, box size
        params['Mpart'] = header['ParticleMassHMsun']  # Msun / h, mass of each particle
        params['velz2kms'] = header['VelZSpace_to_kms']/params['Lbox']
        params['numchunks'] = len(halo_info_fns)

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
        if self.DEFAULTS['want_ranks']:
            p_ranks = np.array([])
            p_ranksv = np.array([])
            p_ranksp = np.array([])
            p_ranksr = np.array([])

        # B.H. make into ASDF
        # load all the halo and particle data we need
        for echunk in range(params['numchunks']):
            print("Loading simulation by chunk, ", echunk)
            if (not self.DEFAULTS['want_ELG']) & (not self.DEFAULTS['want_QSO']):
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

            if self.DEFAULTS['want_ranks']:
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
        if self.DEFAULTS['want_ranks']:
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

    def run_hod(self, LRG_HOD, ELG_HOD, QSO_HOD, rsd = True, write_to_disk = False):

        LRG_dict, ELG_dict, QSO_dict = \
        galcat.gen_gal_cat(self.halo_data, self.particle_data, LRG_HOD, ELG_HOD, QSO_HOD, 
        self.params, enable_ranks = self.DEFAULTS['want_ranks'], rsd = rsd, 
        want_LRG = self.DEFAULTS['want_LRG'], want_ELG = self.DEFAULTS['want_ELG'], 
        want_QSO = self.DEFAULTS['want_QSO'],
        write_to_disk = write_to_disk, savedir = self.mock_dir)

        return LRG_dict, ELG_dict, QSO_dict

# # B.H. is this function necessary?
# def run_onebox(i):
#     gen_gal_onesim_onehod(i, halo_data[i], particle_data[i], newdesign, newdecor, save_dir, newseed, params)
#     return


class ArgParseFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    pass

if __name__ == "__main__":


    # # parsing arguments
    # parser = argparse.ArgumentParser(description=__doc__, formatter_class=ArgParseFormatter)
    # parser.add_argument('--sim_name', help='Simulation name', default=DEFAULTS['sim_name'])
    # parser.add_argument('--z_mock', help='Redshift of the mock', type=float, default=DEFAULTS['z_mock'])
    # parser.add_argument('--scratch_dir', help='Scratch directory', default=DEFAULTS['scratch_dir'])
    # parser.add_argument('--subsample_dir', help='Particle subsample directory', default=DEFAULTS['subsample_dir'])
    # parser.add_argument('--sim_dir', help='Simulation directory', default=DEFAULTS['sim_dir'])
    # parser.add_argument('--want_rsd', help='Want redshift space distortions', default=DEFAULTS['want_rsd'])
    # parser.add_argument('--want_ranks', help='Want extended satellite parameters', default=DEFAULTS['want_ranks'])
    # parser.add_argument('--want_LRG', help='Want LRGs', default=DEFAULTS['want_LRG'])
    # parser.add_argument('--want_ELG', help='Want ELGs', default=DEFAULTS['want_ELG'])
    # parser.add_argument('--want_QSO', help='Want QSOs', default=DEFAULTS['want_QSO'])
    # args = vars(parser.parse_args())

    # # preload the simulation
    # print("preloading simulation")
    # halo_data, particle_data, params, mock_dir = staging(**args)
    # print("finished loading the data into memory")

    # # B.H. I think the HOD parameters could be read from a yaml file or something
    # parser.add_argument('--logM_cut', help='base HOD parameter (Zheng+2007)', type=float, default=DEFAULTS['logM_cut'])
    # parser.add_argument('--logM1', help='base HOD parameter (Zheng+2007)', type=float, default=DEFAULTS['logM1'])
    # parser.add_argument('--sigma', help='base HOD parameter (Zheng+2007)', type=float, default=DEFAULTS['sigma'])
    # parser.add_argument('--alpha', help='base HOD parameter (Zheng+2007)', type=float, default=DEFAULTS['alpha'])
    # parser.add_argument('--kappa', help='base HOD parameter (Zheng+2007)', type=float, default=DEFAULTS['kappa'])
    # parser.add_argument('--alpha_c', help='central velocity bias parameter', type=float, default=DEFAULTS['alpha_c'])
    # parser.add_argument('--alpha_s', help='satellite velocity bias parameter', type=float, default=DEFAULTS['alpha_s'])
    # parser.add_argument('--s', help='satellite distribution parameter', type=float, default=DEFAULTS['s'])
    # parser.add_argument('--s_v', help='satellite distribution parameter', type=float, default=DEFAULTS['s_v'])
    # parser.add_argument('--s_p', help='satellite distribution parameter', type=float, default=DEFAULTS['s_p'])
    # parser.add_argument('--s_r', help='satellite distribution parameter', type=float, default=DEFAULTS['s_r'])
    # parser.add_argument('--Acent', help='Assembly bias parameter for centrals', type=float, default=DEFAULTS['Acent'])
    # parser.add_argument('--Asat', help='Assembly bias parameter for satellites', type=float, default=DEFAULTS['Asat'])
    # parser.add_argument('--Bcent', help='Environmental bias parameter for centrals', type=float, default=DEFAULTS['Bcent'])
    # parser.add_argument('--Bsat', help='Environmental bias parameter for satellites', type=float, default=DEFAULTS['Bsat'])
    # parser.add_argument('--ic', help='incompleteness factor', type=float, default=DEFAULTS['ic'])
    # args = vars(parser.parse_args())
    
    # # setting up the hod dictionary
    # LRGdesign = {'M_cut': 10.**LRG_HOD['logM_cut'],
    #              'M1': 10.**LRG_HOD['logM1'],
    #              'sigma': LRG_HOD['sigma'],
    #              'alpha': LRG_HOD['alpha'],
    #              'kappa': LRG_HOD['kappa']}
    # LRGdecor = {'alpha_c': LRG_HOD['alpha_c'],
    #             'alpha_s': LRG_HOD['alpha_s'],
    #             's': LRG_HOD['s'],
    #             's_v': LRG_HOD['s_v'],
    #             's_p': LRG_HOD['s_p'],
    #             's_r': LRG_HOD['s_r'],
    #             'Acent': LRG_HOD['Acent'],
    #             'Asat': LRG_HOD['Asat'],
    #             'Bcent': LRG_HOD['Bcent'],
    #             'Bsat': LRG_HOD['Bsat'],
    #             'ic': LRG_HOD['ic']}


    # create a new abacushod object
    newBall = AbacusHOD(sim_name = "AbacusSummit_base_c000_ph006",
                        simdir = "/mnt/gosling2/bigsims/", 
                        scratchdir = "/mnt/marvin1/syuan/scratch/data_mocks_summit_new", 
                        subsampledir = "/mnt/marvin1/syuan/scratch/data_summit/", 
                        z = 0.5, 
                        ranks = False, lrg = True, elg = False, qso = False)


    # LRG HOD
    LRG_HOD = {}
    LRG_HOD['logM_cut'] = 13.3
    LRG_HOD['logM1'] = 14.3
    LRG_HOD['sigma'] = 0.3
    LRG_HOD['alpha'] = 1.0
    LRG_HOD['kappa'] = 0.4
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
    # velocity bias
    ELG_HOD['alpha_c'] = 0
    ELG_HOD['alpha_s'] = 1
    # satellite extensions, assembly bias, and incompleteness
    ELG_HOD['s'] = 0
    ELG_HOD['s_v'] = 0
    ELG_HOD['s_p'] = 0
    ELG_HOD['s_r'] = 0
    ELG_HOD['Acent'] = 0
    ELG_HOD['Asat'] = 0
    ELG_HOD['Bcent'] = 0
    ELG_HOD['Bsat'] = 0

    # QSO HOD
    QSO_HOD = {}
    QSO_HOD['p_max'] = 0.33
    QSO_HOD['logM_cut'] = 12.21
    QSO_HOD['kappa'] = 1.0
    QSO_HOD['sigma'] = 0.56
    QSO_HOD['logM1'] = 13.94
    QSO_HOD['alpha'] = 0.4
    QSO_HOD['A_s'] = 1.
    # velocity bias
    QSO_HOD['alpha_c'] = 0
    QSO_HOD['alpha_s'] = 1
    # satellite extensions, assembly bias, and incompleteness
    QSO_HOD['s'] = 0
    QSO_HOD['s_v'] = 0
    QSO_HOD['s_p'] = 0
    QSO_HOD['s_r'] = 0
    QSO_HOD['Acent'] = 0
    QSO_HOD['Asat'] = 0
    QSO_HOD['Bcent'] = 0
    QSO_HOD['Bsat'] = 0


    # throw away run for jit to compile, write to disk
    LRG_dict, ELG_dict, QSO_dict = \
    newBall.run_hod(LRG_HOD, ELG_HOD, QSO_HOD, rsd = True, write_to_disk = True)
    # rpbins and pi bins for benchmarking xirppi code
    rpbins = np.logspace(-1, 1.5, 9)
    pimax = 30
    pi_bin_size = 5
    # run the fit 10 times for timing 
    for i in range(10):
        start = time.time()
        LRG_dict, ELG_dict, QSO_dict = \
        newBall.run_hod(LRG_HOD, ELG_HOD, QSO_HOD, rsd = True, write_to_disk = False)
        print("Done iteration ", i, "took time ", time.time() - start)
        
        # start = time.time()
        # pos_full = np.concatenate((cent_pos, sat_pos), axis = 0)
        # xi = calc_xirppi_fast(pos_full[:, 0], pos_full[:, 1], pos_full[:, 2], rpbins, pimax, pi_bin_size, params['Lbox'], 64)
        # print("xi took time ", time.time() - start)

    # multiprocess
    # p = multiprocessing.Pool(17)
    # p.map(run_onebox, range(params['numchunks']))
    #p.map(gen_gal_onesim_onehod, zip((i, halo_data[i], particle_data[i], newdesign, newdecor, save_dir, newseed, params) for i in range(params['numchunks'])))
    # p.starmap(gen_gal_onesim_onehod, zip(range(params['numchunks']), repeat(halo_data), repeat(particle_data), repeat(newdesign), repeat(newdecor), repeat(save_dir), repeat(newseed), repeat(params)))
    #p.close()
    #p.join()
