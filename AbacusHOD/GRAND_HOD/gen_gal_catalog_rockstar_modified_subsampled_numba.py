#!/usr/bin/env python

"""
Module implementation of a generalized and differentiable Halo Occupation 
Distribution (HOD)for N-body cosmological simulations. 

Add to .bashrc:
export PYTHONPATH="/path/to/GRAND-HOD:$PYTHONPATH"

"""

import os
import sys
import time
from pathlib import Path
import math

import numpy as np
import random
from astropy.table import Table
from astropy.io import ascii
from math import erfc
import h5py
from scipy import special

import numba
from numba import njit

@njit
def n_cen(M_in, M_cut, sigma, m_cutoff = 1e12): 
    """
    Computes the expected number of central galaxies given a halo mass and 
    the HOD design. 

    Parameters
    ----------

    M_in : float
        Halo mass in solar mass.

    design : dict
        Dictionary containing the five HOD parameters. 
        
    m_cutoff: float, optional
        Ignore halos small than this mass.

    Returns
    -------

    n_cen : float
        Number of centrals expected for the halo within the range (0, 1).
        This number should be interpreted as a probability.

    """
    # if M_in < m_cutoff: # this cutoff ignores halos with less than 100 particles
    #     return 0
    # M_cut, M1, sigma, alpha, kappa = \
    # design_array[0], design_array[1], design_array[2], design_array[3], design_array[4]

    return 0.5*math.erfc(np.log(M_cut/M_in)/(2**.5*sigma))

@njit
def n_sat(M_in, M_cut, M1, sigma, alpha, kappa, m_cutoff = 1e12): 
    """
    Computes the expected number of satellite galaxies given a halo mass and 
    the HOD design. 

    Parameters
    ----------

    M_in : float
        Halo mass in solar mass.

    design : dict
        Dictionary containing the five HOD parameters. 
    
    Returns
    -------

    n_sat : float
        Expected number of satellite galaxies for the said halo.

    """

    # if M_in < m_cutoff: # this cutoff ignores halos with less than 100 particles
    #     return 0

    # M_cut, M1, sigma, alpha, kappa = \
    # design_array[0], design_array[1], design_array[2], design_array[3], design_array[4]

    # if M_in < kappa*M_cut:
    #     return 0

    return ((M_in - kappa*M_cut)/M1)**alpha*0.5*math.erfc(np.log(M_cut/M_in)/(2**.5*sigma))

@njit(parallel = True)
def gen_cent(pos, vel, mass, ids, randoms, design_array, ic, rsd, velz2kms, lbox):
    """
    Function that generates central galaxies and its position and velocity 
    given a halo catalog and HOD designs and decorations. The generated 
    galaxies are output to file fcent. 

    Parameters
    ----------

    halo_ids : numpy.array
        Array of halo IDs.

    halo_pos : numpy.array
        Array of halo positions of shape (N, 3) in box units.

    halo_vels : numpy.array
        Array of halo velocities of shape (N, 3) in km/s.

    halo_vrms: numpy.array
        Array of halo particle velocity dispersion in km/s.

    halo_mass : numpy.array
        Array of halo mass in solar mass.

    design : dict
        Dictionary of the five baseline HOD parameters. 

    decorations : dict
        Dictionary of generalized HOD parameters. 

    fcent : file pointer
        Pointer to the central galaxies output file location. 

    rsd : boolean
        Flag of whether to implement RSD. 

    params : dict
        Dictionary of various simulation parameters. 

    Outputs
    -------

    For each halo, if there exists a central, the function outputs the 
    3D position (Mpc), halo ID, and halo mass (Msun) to file.

    """

    # parse out the hod parameters 
    M_cut, M1, sigma, alpha, kappa = \
    design_array[0], design_array[1], design_array[2], design_array[3], design_array[4]

    H = len(mass)

    Nthread = numba.get_num_threads()
    Nout = np.zeros((Nthread, 8), dtype = np.int64)
    hstart = np.rint(np.linspace(0, H, Nthread + 1)) # starting index of each thread

    keep = np.empty(H, dtype = np.int8) # mask array tracking which halos to keep

    # figuring out the number of halos kept for each thread
    for tid in numba.prange(Nthread):
        for i in range(hstart[tid], hstart[tid + 1]):
            if n_cen(mass[i], M_cut, sigma) * ic > randoms[i]:
                Nout[tid, 0] += 1 # counting
                keep[i] = 1
            else:
                keep[i] = 0

    # compose galaxy array, first create array of galaxy starting indices for the threads
    gstart = np.empty(Nthread + 1, dtype = np.int64)
    gstart[0] = 0
    gstart[1:] = Nout[:, 0].cumsum()

    # galaxy arrays
    gpos = np.empty((gstart[-1], 3), dtype = pos.dtype)
    gvel = np.empty((gstart[-1], 3), dtype = vel.dtype)
    gmass = np.empty(gstart[-1], dtype = mass.dtype)
    gid = np.empty(gstart[-1], dtype = ids.dtype)

    # fill in the galaxy arrays
    for tid in numba.prange(Nthread):
        j = gstart[tid]
        for i in range(hstart[tid], hstart[tid + 1]):
            if keep[i]:
                if rsd:
                    gpos[j] = (pos[i] + vel[i] / velz2kms) % lbox
                else:
                    gpos[j] = pos[i]
                gvel[j] = vel[i] # need to extend to include vel bias 
                gmass[j] = mass[i]
                gid[j] = ids[i]
                j += 1
        assert j == gstart[tid + 1]

    return gpos, gvel, gmass, gid 

    # # form the probability array
    # ps = n_cen(mass, design_array) * ic
    # # ps = 0.5*special.erfc(np.log(M_cut/halo_mass)/(2**.5*sigma)) * ic

    # # generate a bunch of numbers for central occupation
    # # do we have centrals?
    # mask_cents = randoms < ps 

    # # generate central los velocity
    # extra_vlos = np.random.normal(loc = 0, scale = abs(alpha_c)*vrms/1.7320508076)

    # # compile the centrals
    # x_cents = maskedhalos['x'][mask_cents]
    # y_cents = maskedhalos['y'][mask_cents]
    # z_cents = maskedhalos['z'][mask_cents]
    # vx_cents = maskedhalos['vx'][mask_cents]
    # vy_cents = maskedhalos['vy'][mask_cents]
    # vz_cents = maskedhalos['vz'][mask_cents]
    # vz_cents += extra_vlos[mask_cents] # add on velocity bias
    # mass_cents = maskedhalos['mass'][mask_cents]
    # ids_cents = maskedhalos['id'][mask_cents]

    # # rsd
    # if rsd:
    #     z_cents = (z_cents + vz_cents/velz2kms) % lbox

    # # output to file
    # newtable = Table([x_cents, y_cents, z_cents, vx_cents, vy_cents, vz_cents, 
    #     ids_cents, mass_cents], names = ('x', 'y', 'z', 'vx', 'vy', 'vz', 'halo_id', 'halo_mass'))
    # print("number of centrals ", len(newtable))
    # return newtable
    # # newarray.tofile(fcent)
    # # for i in range(len(pos_cents)):
    # #     if i % 1000 == 0:
    # #         print(i)
    # #     newline = np.array([pos_cents[i, 0], pos_cents[i, 1], pos_cents[i, 2], 
    # #                         vel_cents[i, 0], vel_cents[i, 1], vel_cents[i, 2], 
    # #                         ids_cents[i], mass_cents[i]])
    # #     newline.tofile(fcent)


@njit(parallel = True)
def gen_sats(ppos, pvel, hmass, hid, Np, subsampling, randoms, design_array, decorations_array, rsd, velz2kms, lbox, Mpart):

    """
    Function that generates satellite galaxies and their positions and 
    velocities given a halo catalog and HOD designs and decorations. 

    The decorations are implemented using a particle re-ranking procedure
    that preserves the random number thrown for each particle so that 
    the resulting statistic has no induced shot-noise and is thus 
    differentiable.

    The generated galaxies are output to binary file fsats. 

    Parameters
    ----------

    halo_ids : numpy.array
        Array of halo IDs.

    halo_pos : numpy.array
        Array of halo positions of shape (N, 3) in box units.

    halo_vels : numpy.array
        Array of halo velocities of shape (N, 3) in km/s.

    newpart : h5py file pointer
        The pointer to the particle file.

    halo_mass : numpy.array
        Array of halo mass in solar mass.

    halo_pstart : numpy.array
        Array of particle start indices for each halo.

    halo_pnum : numpy.array
        Array of number of particles for halos. 

    design : dict
        Dictionary of the five baseline HOD parameters. 

    decorations : dict
        Dictionary of generalized HOD parameters. 

    fsats : file pointer
        Pointer to the satellite galaxies output file location. 

    rsd : boolean
        Flag of whether to implement RSD. 

    params : dict
        Dictionary of various simulation parameters. 

    Outputs
    -------

    For each halo, the function outputs the satellite galaxies, specifically
    the 3D position (Mpc), halo ID, and halo mass (Msun) to file.


    """

    # standard hod design
    M_cut, M1, sigma, alpha, kappa = \
    design_array[0], design_array[1], design_array[2], design_array[3], design_array[4]
    s, s_v, alpha_c, s_p, s_r, A, Ae, ic = decorations_array[0], \
    decorations_array[1], decorations_array[2], decorations_array[3], decorations_array[4], \
    decorations_array[5], decorations_array[6], decorations_array[7]

    H = len(hmass) # num of particles

    Nthread = numba.get_num_threads()
    Nout = np.zeros((Nthread, 8), dtype = np.int64)
    hstart = np.rint(np.linspace(0, H, Nthread + 1)) # starting index of each thread

    keep = np.empty(H, dtype = np.int8) # mask array tracking which halos to keep

    # figuring out the number of particles kept for each thread
    for tid in numba.prange(Nthread):
        for i in range(hstart[tid], hstart[tid + 1]):
            if n_sat(hmass[i], M_cut, M1, sigma, alpha, kappa) / Np[i] / subsampling[i] * ic > randoms[i]:
                Nout[tid, 0] += 1 # counting
                keep[i] = 1
            else:
                keep[i] = 0

    # compose galaxy array, first create array of galaxy starting indices for the threads
    gstart = np.empty(Nthread + 1, dtype = np.int64)
    gstart[0] = 0
    gstart[1:] = Nout[:, 0].cumsum()

    # galaxy arrays
    gpos = np.empty((gstart[-1], 3), dtype = ppos.dtype)
    gvel = np.empty((gstart[-1], 3), dtype = pvel.dtype)
    gmass = np.empty(gstart[-1], dtype = hmass.dtype)
    gid = np.empty(gstart[-1], dtype = hid.dtype)

    # fill in the galaxy arrays
    for tid in numba.prange(Nthread):
        j = gstart[tid]
        for i in range(hstart[tid], hstart[tid + 1]):
            if keep[i]:
                if rsd:
                    gpos[j] = (ppos[i] + pvel[i] / velz2kms) % lbox
                else:
                    gpos[j] = ppos[i]
                gvel[j] = pvel[i] # need to extend to include vel bias 
                gmass[j] = hmass[i]
                gid[j] = hid[i]
                j += 1
        assert j == gstart[tid + 1]

    return gpos, gvel, gmass, gid 

    # # expected number of galaxies for each particle 
    # Nsat_exp = n_sat(hmass, design_array) / Np / subsampling * ic

    # # random_list = np.random.random(len(Nsat_exp))
    # satmask = subsample['randoms'] < Nsat_exp

    # return satmask



def gen_gals(halos_array, subsample, design, decorations, rsd, params):
    """
    parse hod parameters, pass them on to central and satellite generators 
    and then format the results 

    Parameters
    ----------

    halos_array : list of arrays 
        a list of halo properties (pos, vel, mass, id, randoms)

    subsample : list of arrays
        a list of particle propoerties (pos, vel, hmass, hid, Np, subsampling, randoms)

    design : dict
        Dictionary of the five baseline HOD parameters. 

    decorations : dict
        Dictionary of generalized HOD parameters. 

    rsd : boolean
        Flag of whether to implement RSD. 

    params : dict
        Dictionary of various simulation parameters. 


    """

    M_cut, M1, sigma, alpha, kappa = map(design.get, ('M_cut', 
                                                      'M1', 
                                                      'sigma', 
                                                      'alpha', 
                                                      'kappa'))
    design_array = np.array([M_cut, M1, sigma, alpha, kappa])

    s, s_v, alpha_c, s_p, s_r, A, Ae, ic = map(decorations.get, ('s', 
                                                    's_v', 
                                                    'alpha_c', 
                                                    's_p', 
                                                    's_r',
                                                    'A',
                                                    'Ae',
                                                    'ic'))
    decorations_array = np.array([s, s_v, alpha_c, s_p, s_r, A, Ae, ic])

    start = time.time()

    velz2kms = params['velz2kms']
    lbox = params['Lbox']
    # for each halo, generate central galaxies and output to file
    cent_pos, cent_vel, cent_mass, cent_id = \
    gen_cent(halos_array[0], halos_array[1], halos_array[2], halos_array[3],
     halos_array[4], design_array, ic, rsd, velz2kms, lbox)

    print("generating centrals took ", time.time() - start)
    # open particle file
    # part_pos = subsample[0]
    # part_vel = subsample[1]
    # part_halomass = subsample[2]
    # part_haloid = subsample[3]
    # part_Np = subsample[4]
    # part_subsample = subsample[5]
    # part_randoms = subsample[6]
    # part_ranks = subsample[:, 6]
    # part_ranksv = subsample[:, 7]
    # part_ranksp = subsample[:, 8]
    # part_ranksr = subsample[:, 9]

    start = time.time()
    sat_pos, sat_vel, sat_mass, sat_id = \
    gen_sats(subsample[0], subsample[1], subsample[2], subsample[3], 
        subsample[4],  subsample[5],  subsample[6], 
        design_array, decorations_array, rsd, velz2kms, lbox, params['Mpart'])
    print("generating satellites took ", time.time() - start)

    # start = time.time()
    # sat_table = Table([subsample['x'][satmask], subsample['y'][satmask], subsample['z'][satmask], 
    #     subsample['vx'][satmask], subsample['vy'][satmask], subsample['vz'][satmask],
    #     subsample['hid'][satmask], subsample['hmass'][satmask]], 
    #     names = ('x', 'y', 'z', 'vx', 'vy', 'vz', 'halo_id', 'halo_mass'))
    # # satellite rsd
    # if rsd:
    #     sat_table['z'] = (sat_table['z'] + sat_table['vz']/velz2kms) % lbox

    # subsample[satmask, 0:8].tofile(fsats)
    # newarray.tofile(fsats)
    # print("outputting satellites took ", time.time() - start, "number of satellites", len(sat_table))

    return cent_pos, cent_vel, cent_mass, cent_id, sat_pos, sat_vel, sat_mass, sat_id


def gen_gal_cat(halo_data, particle_data, design, decorations, params, 
                rsd = True, write_to_disk = False, savedir = "./"):
    """
    takes in data, do some checks, call the gen_gal functions, and then take care of outputs

    Parameters
    ----------

    whichsim : int
        Simulation number. Ranges between [0, 15] for current Planck 1100 sims.

    design : dict
        Dictionary of the five baseline HOD parameters. 

    decorations : dict
        Dictionary of generalized HOD parameters. 

    params : dict
        Dictionary of various simulation parameters. 

    whatseed : integer, optional
        The initial seed to the random number generator. 

    rsd : boolean, optional
        Flag of whether to implement RSD. 

    product_dir : string, optional
        A string indicating the location of the simulation data. 
        You should not need to change this if you are on Eisenstein group clusters.

    simname : string, optional
        The name of the simulation boxes. Defaulted to 1100 planck boxes.

    """

    # checking for errors
    # if not type(whichchunk) is int or whichchunk < 0:
    #     print("Error: whichchunk has to be a non-negative integer.")

    if not type(rsd) is bool:
        print("Error: rsd has to be a boolean.")


    # # find the halos, populate them with galaxies and write them to files
    cent_pos, cent_vel, cent_mass, cent_id, sat_pos, sat_vel, sat_mass, sat_id \
     = gen_gals(halo_data, particle_data, design, decorations, rsd, params)

    print(np.shape(cent_pos), np.shape(sat_pos))
    # # close the files in the end
    # if write_to_disk:
    #     start = time.time()
    #     ascii.write(cent_table, output = savedir / ("halos_gal_cent_full.dat"), overwrite = True)
    #     ascii.write(sat_table, output = savedir / ("halos_gal_sat_full.dat"), overwrite = True)
    #     print("Done writing to disk, time ", time.time() - start)
    # return cent_table, sat_table

    # fcent = open(savedir / ("halos_gal_cent_full"),'wb')
    # fsats = open(savedir / ("halos_gal_sats_full"),'wb')

    # fcent.close()
    # fsats.close()
    # print("Galaxy Catalogs Done. chunk: ", whichchunk)



