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

import numpy as np
import random
from astropy.table import Table
from math import erfc
import h5py
from scipy import special

from numba import jit

def n_cen(M_in, design_array, m_cutoff = 1e12): 
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
    M_cut, M1, sigma, alpha, kappa = \
    design_array[0], design_array[1], design_array[2], design_array[3], design_array[4]

    return 0.5*special.erfc(np.log(M_cut/M_in)/(2**.5*sigma))

# @jit(nopython = True)
def n_sat(M_in, design_array, m_cutoff = 1e12): 
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

    M_cut, M1, sigma, alpha, kappa = \
    design_array[0], design_array[1], design_array[2], design_array[3], design_array[4]

    # if M_in < kappa*M_cut:
    #     return 0

    return ((M_in - kappa*M_cut)/M1)**alpha*0.5*special.erfc(np.log(M_cut/M_in)/(2**.5*sigma))

# generate central galaxy given a halo
# @jit(nopython = True)
def gen_cent(maskedhalos, design_array, alpha_c, ic, rsd, fcent, velz2kms, lbox, whatseed = 0):
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

    np.random.seed(whatseed)

    # parse out the hod parameters 
    M_cut, M1, sigma, alpha, kappa = \
    design_array[0], design_array[1], design_array[2], design_array[3], design_array[4]

    # # there is a cutoff mass 
    # m_cutoff = 1e12

    # form the probability array
    ps = n_cen(maskedhalos['mass'], design_array) * ic
    # ps = 0.5*special.erfc(np.log(M_cut/halo_mass)/(2**.5*sigma)) * ic

    # generate a bunch of numbers for central occupation
    # do we have centrals?
    mask_cents = maskedhalos['randoms'] < ps 

    # generate central los velocity
    extra_vlos = np.random.normal(loc = 0, scale = abs(alpha_c)*maskedhalos['vrms']/1.7320508076)

    # compile the centrals
    x_cents = maskedhalos['x'][mask_cents]
    # print(x_cents)
    y_cents = maskedhalos['y'][mask_cents]
    z_cents = maskedhalos['z'][mask_cents]
    vx_cents = maskedhalos['vx'][mask_cents]
    vy_cents = maskedhalos['vy'][mask_cents]
    vz_cents = maskedhalos['vz'][mask_cents]
    vz_cents += extra_vlos[mask_cents] # add on velocity bias
    mass_cents = maskedhalos['mass'][mask_cents]
    ids_cents = maskedhalos['id'][mask_cents]

    # rsd
    if rsd:
        z_cents = (z_cents + vz_cents/velz2kms) % lbox

    # output to file
    newarray = np.concatenate((x_cents[:, None], y_cents[:, None], z_cents[:, None], 
        vx_cents[:, None], vy_cents[:, None], vz_cents[:, None], 
        ids_cents[:, None], mass_cents[:, None]), axis = 1)
    print("number of centrals ", len(newarray))
    newarray.tofile(fcent)
    # for i in range(len(pos_cents)):
    #     if i % 1000 == 0:
    #         print(i)
    #     newline = np.array([pos_cents[i, 0], pos_cents[i, 1], pos_cents[i, 2], 
    #                         vel_cents[i, 0], vel_cents[i, 1], vel_cents[i, 2], 
    #                         ids_cents[i], mass_cents[i]])
    #     newline.tofile(fcent)


# @jit(nopython = True)
def gen_sats(subsample, design_array, decorations_array, rsd, velz2kms, lbox, Mpart, whatseed = 0):

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
    np.random.seed(whatseed + 14838492)

    # standard hod design
    M_cut, M1, sigma, alpha, kappa = \
    design_array[0], design_array[1], design_array[2], design_array[3], design_array[4]
    s, s_v, alpha_c, s_p, s_r, A, Ae, ic = decorations_array[0], \
    decorations_array[1], decorations_array[2], decorations_array[3], decorations_array[4], \
    decorations_array[5], decorations_array[6], decorations_array[7]

    # expected number of galaxies for each particle 
    Nsat_exp = n_sat(subsample['hmass'], design_array) / subsample['Np'] / subsample['subsample'] * ic

    # random_list = np.random.random(len(Nsat_exp))
    satmask = subsample['randoms'] < Nsat_exp

    return satmask


def gen_gals(whichchunk, maskedhalos, subsample, design, decorations, 
    fcent, fsats, rsd, params, whatseed = 0):
    """
    Function that compiles halo catalog from directory and implements 
    assembly bias decoration by halo re-ranking. 

    The halo catalogs are then passed on to functions that generate central
    and satellite galaxies. 

    Parameters
    ----------

    directory : string 
        Directory of the halo and particle files. 

    design : dict
        Dictionary of the five baseline HOD parameters. 

    decorations : dict
        Dictionary of generalized HOD parameters. 

    fcent : file pointer
        Pointer to the central galaxies output file location. 

    fsats : file pointer
        Pointer to the satellite galaxies output file location. 

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
    gen_cent(maskedhalos, design_array, alpha_c, ic, rsd, fcent, velz2kms, lbox, whatseed = whatseed + whichchunk)

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
    # for each halo, generate satellites and output to file
    # print(len(halo_ids), sum(halo_submask))
    satmask = gen_sats(subsample, design_array, decorations_array, rsd, velz2kms, lbox, 
        params['Mpart'], whatseed = whatseed + whichchunk)
    print("generating mask took ", time.time() - start)
    start = time.time()
    newarray = np.concatenate((subsample['x'][satmask, None], subsample['y'][satmask, None], subsample['z'][satmask, None], 
        subsample['vx'][satmask, None], subsample['vy'][satmask, None], subsample['vz'][satmask, None], 
        subsample['hmass'][satmask, None], subsample['hid'][satmask, None]), axis = 1)
    # satellite rsd
    if rsd:
        newarray[:,2] = (newarray[:,2] + newarray[:,5]/velz2kms) % lbox

    # subsample[satmask, 0:8].tofile(fsats)
    newarray.tofile(fsats)
    print("outputting satellites took ", time.time() - start, "number of satellites", len(newarray))


def gen_gal_cat(whichchunk, halo_data, particle_data, design, decorations, params, savedir, 
                whatseed = 0, rsd = True):
    """
    Main interface that takes in the simulation number, HOD design and 
    decorations and outputs the resulting central and satellite galaxy
    catalogs to file in binary format. 

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
    if not type(whichchunk) is int or whichchunk < 0:
        print("Error: whichchunk has to be a non-negative integer.")

    if not type(rsd) is bool:
        print("Error: rsd has to be a boolean.")


    whichchunk = int(whichchunk)
    # print("Generating galaxy catalog. chunk: ", whichchunk)

    # # directory of the halo and particle files
    # directory_part = product_dir \
    # +simname + "-"+str(whichsim)+"_products/"\
    # +simname + "-"+str(whichsim)\
    # +"_rockstar_halos/z{:.3f}".format(params['z'])

    # directory = "/mnt/marvin1/syuan/scratch/data_summit"

    # # if this directory does not exist, make it
    # if not os.path.exists(savedir):
    #     os.makedirs(savedir)

    # # binary output galaxy catalog
    # print("Building galaxy catalog (binary output)")
    fcent = open(savedir / ("halos_gal_cent_%d"%whichchunk),'wb')
    fsats = open(savedir / ("halos_gal_sats_%d"%whichchunk),'wb')

    # # find the halos, populate them with galaxies and write them to files
    gen_gals(whichchunk, halo_data, particle_data, design, decorations, 
        fcent, fsats, rsd, params, whatseed = whatseed)

    # # close the files in the end
    fcent.close()
    fsats.close()
    # print("Galaxy Catalogs Done. chunk: ", whichchunk)



