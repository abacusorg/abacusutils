'''
This is a script for creating an AbacusHOD.
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
# TODO B.H.: add power spectrum calculation
# TODO B.H.: staging can be shorter and prettier; perhaps asdf for h5 and ecsv?
# TODO B.H.: hook to emcee: need data
# TODO B.H.: returns dictionary with keys ELG, LRG, QSO

class AbacusHOD:
    def __init__(self, sim_params, HOD_params, power_params):

        # simulation details
        self.sim_name = sim_params['sim_name']
        self.sim_dir = sim_params['sim_dir']
        self.subsample_dir = sim_params['subsample_dir']
        self.z_mock = sim_params['z_mock']
        self.scratch_dir = sim_params['scratch_dir']

        # power spectrum parameters
        self.bin_params = power_params['bin_params']

        # tracers
        tracer_flags = HOD_params['tracer_flags']
        tracers = {}
        for key in tracer_flags.keys():
            if tracer_flags[key]:
                tracers[key] = HOD_params[key+'_params']
        self.tracers = tracers

        # HOD parameter choices
        self.want_ranks = HOD_params['want_ranks']
        self.want_rsd = HOD_params['want_rsd']
        self.write_to_disk = HOD_params['write_to_disk']
        
        self.halo_data, self.particle_data, self.params, self.mock_dir = self.staging()


    def staging(self):

        # all paths relevant for mock generation
        scratch_dir = Path(self.scratch_dir)
        simname = Path(self.sim_name)
        sim_dir = Path(self.sim_dir)
        mock_dir = scratch_dir / simname / ('z%4.3f'%self.z_mock)
        # create mock_dir if not created
        if not mock_dir.exists():
            mock_dir.mkdir(parents = True)
        subsample_dir = \
        Path(self.subsample_dir) / simname / ('z%4.3f'%self.z_mock)

        # load header to read parameters
        halo_info_fns = \
        list((sim_dir / simname / 'halos' / ('z%4.3f'%self.z_mock) / 'halo_info').glob('*.asdf'))
        f = asdf.open(halo_info_fns[0], lazy_load=True, copy_arrays=False)
        header = f['header']

        # constants
        params = {}
        params['z'] = self.z_mock
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
        if self.want_ranks:
            p_ranks = np.array([])
            p_ranksv = np.array([])
            p_ranksp = np.array([])
            p_ranksr = np.array([])

        # B.H. make into ASDF
        # load all the halo and particle data we need
        for echunk in range(params['numchunks']):
            print("Loading simulation by chunk, ", echunk)
            
            if 'ELG' not in self.tracers.keys() and 'QSO' not in self.tracers.keys():
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

            if self.want_ranks:
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
        if self.want_ranks:
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

    
    def run_hod(self, tracers):
        
        HOD_dict = galcat.gen_gal_cat(self.halo_data, self.particle_data, self.tracers, self.params, enable_ranks = self.want_ranks, rsd = self.want_rsd, write_to_disk = self.write_to_disk, savedir = self.mock_dir)

        return HOD_dict
