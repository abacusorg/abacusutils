#!/usr/bin/env python

"""
Module implementation of a generalized and differentiable Halo Occupation 
Distribution (HOD)for N-body cosmological simulations. 

Add to .bashrc:
export PYTHONPATH="/path/to/GRAND-HOD:$PYTHONPATH"

"""

import numpy as np
import os
import sys
import random
import time
from astropy.table import Table
import astropy.io.fits as pf
from math import erfc
import h5py
from scipy import special
from glob import glob

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
def gen_cent(halo_ids, halo_pos, halo_vels, halo_vrms, halo_mass, 
             design_array, alpha_c, ic, rsd, fcent, velz2kms, lbox, whatseed = 0):
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
    ps = n_cen(halo_mass, design_array) * ic
    # ps = 0.5*special.erfc(np.log(M_cut/halo_mass)/(2**.5*sigma)) * ic

    # generate a bunch of numbers for central occupation
    r_cents = np.random.random(len(ps))
    # do we have centrals?
    mask_cents = r_cents < ps 

    # generate central los velocity
    vrms_los = halo_vrms/1.7320508076 # km/s
    extra_vlos = np.random.normal(loc = 0, scale = abs(alpha_c)*vrms_los)

    # compile the centrals
    pos_cents = halo_pos[mask_cents]
    vel_cents = halo_vels[mask_cents]
    vel_cents[:, 2] += extra_vlos[mask_cents] # add on velocity bias
    mass_cents = halo_mass[mask_cents]
    ids_cents = halo_ids[mask_cents]

    # rsd
    if rsd:
        pos_cents[:, 2] = (pos_cents[:, 2] + vel_cents[:, 2]/velz2kms) % lbox

    # output to file
    newarray = np.concatenate((pos_cents, vel_cents, ids_cents[:, None], mass_cents[:, None]), axis = 1)
    newarray.tofile(fcent)
    # for i in range(len(pos_cents)):
    #     if i % 1000 == 0:
    #         print(i)
    #     newline = np.array([pos_cents[i, 0], pos_cents[i, 1], pos_cents[i, 2], 
    #                         vel_cents[i, 0], vel_cents[i, 1], vel_cents[i, 2], 
    #                         ids_cents[i], mass_cents[i]])
    #     newline.tofile(fcent)


# @jit(nopython = True)
def gen_sats(part_pos, part_vel, part_halomass, part_haloid, part_Np, part_subsample, 
    design_array, decorations_array, rsd, velz2kms, lbox, Mpart, whatseed = 0):

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
    Nsat_exp = n_sat(part_halomass, design_array) / part_Np / part_subsample * ic

    random_list = np.random.random(len(Nsat_exp))
    satmask = random_list < Nsat_exp

    return satmask

    # # loop through the halos to populate satellites
    # # print("generating satellites")
    # data_sats = np.zeros((1, 8))
    # for i in np.arange(len(halo_ids)):
    #     thishalo_id = halo_ids[i]
    #     thishalo_mass = halo_mass[i]
    #     # load the 10% subsample belonging to the halo
    #     start_ind = int(halo_pstart[i])
    #     numparts = int(halo_pnum[i])
    #     # if there are no particles in the particle subsample, move on
    #     if numparts == 0:
    #         continue
    #     # extract the particle positions and vels
    #     ss_pos = part_pos[start_ind: start_ind + numparts] # Mpc
    #     ss_vels = part_vel[start_ind: start_ind + numparts] # km/s
    #     ss_ranks = part_ranks[start_ind: start_ind + numparts] # km/s
    #     ss_ranksv = part_ranksv[start_ind: start_ind + numparts] # km/s
    #     ss_ranksp = part_ranksp[start_ind: start_ind + numparts] # km/s
    #     ss_ranksr = part_ranksr[start_ind: start_ind + numparts] # km/s

    #     # generate a list of random numbers that will track each particle
    #     random_list = np.random.random(numparts)

    #     # compute the undecorated expected number of satellites
    #     N_sat = n_sat(halo_mass[i], design_array) * halo_multi[i]

    #     if N_sat == 0:
    #         continue

    #     # the undecorated probability of each particle hosting a satellite
    #     eachprob = float(N_sat)/numparts * ic
    #     eachprob_array = np.ones(numparts)*eachprob

    #     temp_indices = np.arange(numparts)
    #     temp_range = numparts - 1

    #     # if there is one particle, then we dont need to do any reranking
    #     if numparts > 1:
    #         modifier = (1 - s*(1 - ss_ranks/(temp_range/2.0))) * (1 - s_v*(1 - ss_ranksv/(temp_range/2.0))) \
    #         *(1 - s_p*(1 - ss_ranksp/(temp_range/2.0))) *(1 - s_r*(1 - ss_ranksr/(temp_range/2.0)))
    #         eachprob_array = eachprob_array*modifier / np.mean(modifier)
            
    #     # we have finished the reranking routines
    #     # decide which particle bears galaxy
    #     newmask = random_list < eachprob_array
    #     # generate the position and velocity of the galaxies
    #     sat_pos = ss_pos[newmask]
    #     sat_vels = ss_vels[newmask]

    #     # so a lot of the sat_pos are empty, in that case, just pass
    #     if len(sat_pos) == 0:
    #         continue

    #     # rsd, modify the satellite positions by their velocities
    #     if rsd:
    #         sat_pos[:,2] = (sat_pos[:,2] + sat_vels[:,2]/velz2kms) % lbox

    #     # output
    #     for j in range(len(sat_pos)):
    #         newline_sat = np.array([[sat_pos[j, 0],
    #                                 sat_pos[j, 1],
    #                                 sat_pos[j, 2],
    #                                 sat_vels[j, 0],
    #                                 sat_vels[j, 1],
    #                                 sat_vels[j, 2],                                    
    #                                 thishalo_id, 
    #                                 thishalo_mass]])
    #         data_sats = np.vstack((data_sats, newline_sat))

    # return data_sats[1:]


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
    # extracting the halo properties that we need
    halo_ids = np.array(maskedhalos[:, 0], dtype = int) # halo IDs
    halo_pos = maskedhalos[:, 1:4] # halo positions, Mpc
    halo_vels = maskedhalos[:, 4:7] # halo velocities, km/s
    halo_vrms = maskedhalos[:, 7] # halo velocity dispersions, km/s
    halo_mass = maskedhalos[:, 8] # halo mass, Msun, 200b
    halo_deltac = maskedhalos[:, 9] # halo concentration
    halo_fenv = maskedhalos[:, 10] # halo velocities, km/s
    halo_pstart = np.array(maskedhalos[:, 11], dtype = int) # starting index of particles
    halo_pnum = np.array(maskedhalos[:, 12], dtype = int) # number of particles 
    halo_multi = maskedhalos[:, 13]
    halo_submask = np.array(maskedhalos[:, 14], dtype = bool)
    print("the halo loading stuff took ", time.time() - start)
    start = time.time()

    # # if assembly bias parameter is not zero, then we do halo reranking
    # if not ((A == 0) & (Ae == 0)):

    #     # define a ranking parameter
    #     halo_pseudomass = halo_mass*np.exp(A*halo_deltac + Ae*halo_fenv)

    #     # create a list that indicates the original order 
    #     halo_order = np.arange(len(halo_ids))

    #     # first we sort everything by mass, original mass
    #     msorted_indices = halo_mass.argsort()[::-1] # descending order

    #     halo_mass = halo_mass[msorted_indices] 
    #     halo_pseudomass = halo_pseudomass[msorted_indices]
    #     halo_order = halo_order[msorted_indices]

    #     # now we resort using halo_pseudomass and get the indices
    #     new_indices = halo_pseudomass.argsort()[::-1] # descending order
    #     halo_order = halo_order[new_indices]
    #     # we dont touch halo mass so it is still sorted by mass

    #     # revert to the original order 
    #     original_indices = halo_order.argsort() # ascending order
    #     halo_mass = halo_mass[original_indices]

    # print("the assembly bias stuff took ", time.time() - start)
    # start = time.time()

    velz2kms = params['velz2kms']
    lbox = params['Lbox']
    # for each halo, generate central galaxies and output to file
    gen_cent(halo_ids, halo_pos, halo_vels, halo_vrms, halo_mass, 
             design_array, alpha_c, ic, rsd, fcent, velz2kms, lbox, whatseed = whatseed)

    print("generating centrals took ", time.time() - start)
    # open particle file
    part_pos = subsample[:, 0:3]
    part_vel = subsample[:, 3:6]
    part_halomass = subsample[:, 6]
    part_haloid = subsample[:, 7]
    part_Np = subsample[:, 8]
    part_subsample = subsample[:, 9]
    # part_ranks = subsample[:, 6]
    # part_ranksv = subsample[:, 7]
    # part_ranksp = subsample[:, 8]
    # part_ranksr = subsample[:, 9]

    start = time.time()
    # for each halo, generate satellites and output to file
    # print(len(halo_ids), sum(halo_submask))
    satmask = gen_sats(part_pos, part_vel, part_halomass, part_haloid, part_Np, part_subsample, 
        design_array, decorations_array, rsd, velz2kms, lbox, params['Mpart'], whatseed = whatseed)
    # satellite rsd
    if rsd:
        subsample[:,2] = (subsample[:,2] + subsample[:,5]/velz2kms) % lbox

    subsample[satmask, 0:8].tofile(fsats)
    print("outputting satellites took ", time.time() - start)


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
    fcent = open(savedir+"/halos_gal_cent_"+str(whichchunk),'wb')
    fsats = open(savedir+"/halos_gal_sats_"+str(whichchunk),'wb')

    # # find the halos, populate them with galaxies and write them to files
    gen_gals(whichchunk, halo_data, particle_data, design, decorations, 
        fcent, fsats, rsd, params, whatseed = whatseed)

    # # close the files in the end
    fcent.close()
    fsats.close()
    # print("Galaxy Catalogs Done. chunk: ", whichchunk)



