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
numba.set_num_threads(64)

@njit(fastmath=True)
def n_cen(M_in, logM_cut, sigma): 
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

    #return 0.5*math.erfc(np.log(M_cut/M_in)/(2**.5*sigma))
    # log seems faster than division
    return 0.5*math.erfc((logM_cut - np.log(M_in))/(2**.5*sigma))

@njit(fastmath=True)
def n_sat(M_in, logM_cut, M_cut, M1, sigma, alpha, kappa): 
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

    return ((M_in - kappa*M_cut)/M1)**alpha*0.5*math.erfc((logM_cut - np.log(M_in))/(2**.5*sigma))


@njit(fastmath=True)
def wrap(x, L):
    '''Fast scalar mod implementation'''
    L2 = L/2
    if x >= L2:
        return x - L
    elif x < -L2:
        return x + L
    return x


@njit(parallel=True, fastmath=True)
def gen_cent(pos, vel, mass, ids, multis, randoms, vdev, deltac, fenv, 
    design_array, ic, alpha_c, Ac, Bc, rsd, inv_velz2kms, lbox):
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
    logM_cut, M1, sigma, alpha, kappa = \
    design_array[0], design_array[1], design_array[2], design_array[3], design_array[4]

    H = len(mass)

    Nthread = numba.get_num_threads()
    Nout = np.zeros((Nthread, 8), dtype = np.int64)
    hstart = np.rint(np.linspace(0, H, Nthread + 1)) # starting index of each thread

    keep = np.empty(H, dtype = np.int8) # mask array tracking which halos to keep

    # figuring out the number of halos kept for each thread
    for tid in numba.prange(Nthread):
        for i in range(hstart[tid], hstart[tid + 1]):
            # do assembly bias and secondary bias
            logM_cut_temp = logM_cut + Ac * deltac[i] + Bc * fenv[i]
            if n_cen(mass[i], logM_cut_temp, sigma) * ic * multis[i] > randoms[i]:
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
                # loop thru three directions to assign galaxy velocities and positions
                for k in range(3):
                    gpos[j,k] = pos[i,k]
                    gvel[j,k] = vel[i,k] + alpha_c * vdev[i] # velocity bias
                # rsd only applies to the z direction
                if rsd:
                    gpos[j,2] = wrap(pos[i,2] + gvel[j,2] * inv_velz2kms, lbox)
                gmass[j] = mass[i]
                gid[j] = ids[i]
                j += 1
        # assert j == gstart[tid + 1]

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


@njit(parallel = True, fastmath = True)
def gen_sats(ppos, pvel, hvel, hmass, hid, inv_Np, 
    inv_subsampling, randoms, hdeltac, hfenv, 
    enable_ranks, ranks, ranksv, ranksp, ranksr, 
    design_array, decorations_array, rsd, inv_velz2kms, lbox, Mpart):

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
    logM_cut, M1, sigma, alpha, kappa = \
    design_array[0], design_array[1], design_array[2], design_array[3], design_array[4]
    M_cut = np.exp(logM_cut)
    logM1 = np.log(M1)
    alpha_s, s, s_v, s_p, s_r, Ac, As, Bc, Bs, ic = \
    decorations_array[1], decorations_array[2], decorations_array[3], decorations_array[4], \
    decorations_array[5], decorations_array[6], decorations_array[7], decorations_array[8], \
    decorations_array[9], decorations_array[10]

    H = len(hmass) # num of particles

    Nthread = numba.get_num_threads()
    Nout = np.zeros((Nthread, 8), dtype = np.int64)
    hstart = np.rint(np.linspace(0, H, Nthread + 1)) # starting index of each thread

    keep = np.empty(H, dtype = np.int8) # mask array tracking which halos to keep

    # figuring out the number of particles kept for each thread
    for tid in numba.prange(Nthread): #numba.prange(Nthread):
        for i in range(hstart[tid], hstart[tid + 1]):
            # print(logM1, As, hdeltac[i], Bs, hfenv[i])
            M1_temp = np.exp(logM1 + As * hdeltac[i] + Bs * hfenv[i])
            logM_cut_temp = logM_cut + Ac * hdeltac[i] + Bc * hfenv[i]
            base_p = n_sat(hmass[i], logM_cut_temp, np.exp(logM_cut_temp), M1_temp, sigma, alpha, kappa)\
             * inv_Np[i] * inv_subsampling[i] * ic
            if enable_ranks:
                decorator = 1 + s * ranks[i] + s_v * ranksv[i] + s_p * ranksp[i] + s_r * ranksr[i]
                exp_sat = base_p * decorator
            else:
                exp_sat = base_p
            if exp_sat > randoms[i]:
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
                for k in range(3):
                    gpos[j, k] = ppos[i, k]
                    gvel[j, k] = hvel[i, k] + alpha_s * (pvel[i, k] - hvel[i, k]) # velocity bias
                if rsd:
                    gpos[j, 2] = wrap(gpos[i, 2] + gvel[i, 2] * inv_velz2kms, lbox)
                gmass[j] = hmass[i]
                gid[j] = hid[i]
                j += 1
        # assert j == gstart[tid + 1]

    return gpos, gvel, gmass, gid 




def gen_gals(halos_array, subsample, design, decorations, rsd, params, enable_ranks):
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
    design_array = np.array([np.log(M_cut), M1, sigma, alpha, kappa])

    alpha_c, alpha_s, s, s_v, s_p, s_r, Ac, As, Bc, Bs, ic = map(decorations.get, ('alpha_c', 
                                                    'alpha_c',  
                                                    's', 
                                                    's_v', 
                                                    's_p', 
                                                    's_r',
                                                    'Acent',
                                                    'Asat',
                                                    'Bcent',
                                                    'Bsat',
                                                    'ic'))
    decorations_array = np.array([alpha_c, alpha_s, s, s_v, s_p, s_r, Ac, As, Bc, Bs, ic])

    start = time.time()

    velz2kms = params['velz2kms']
    lbox = params['Lbox']
    # for each halo, generate central galaxies and output to file
    cent_pos, cent_vel, cent_mass, cent_id = \
    gen_cent(halos_array[0], halos_array[1], halos_array[2], halos_array[3], halos_array[4], 
     halos_array[5], halos_array[6], halos_array[7], halos_array[8], 
     design_array, ic, alpha_c, Ac, Bc, rsd, 1/velz2kms, lbox)
    print("generating centrals took ", time.time() - start, "number of centrals ", len(cent_mass))

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

    inv_Np = 1/subsample[5]
    inv_subsampling = 1/subsample[6]
    inv_velz2kms = 1/velz2kms
    numparts = len(subsample[5])
    if enable_ranks:
        ranks = subsample[10]
        ranksv = subsample[11]
        ranksp = subsample[12]
        ranksr = subsample[13]
    else:
        ranks = np.zeros(numparts)
        ranksv = np.zeros(numparts)
        ranksp = np.zeros(numparts)
        ranksr = np.zeros(numparts)
    start = time.time()
    sat_pos, sat_vel, sat_mass, sat_id = \
    gen_sats(subsample[0], subsample[1], subsample[2], subsample[3], subsample[4], inv_Np,  
        inv_subsampling, subsample[7], subsample[8], subsample[9], 
        enable_ranks, ranks, ranksv, ranksp, ranksr,
        design_array, decorations_array, rsd, inv_velz2kms, lbox, params['Mpart'])

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


def gen_gal_cat(halo_data, particle_data, design, decorations, params, enable_ranks = False,
                rsd = True, write_to_disk = False, savedir = ("./")):
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
     = gen_gals(halo_data, particle_data, design, decorations, rsd, params, enable_ranks)

    print("generated ", np.shape(cent_pos), " centrals and ", np.shape(sat_pos), " satellites.")
    if write_to_disk:
        print("outputting galaxies to disk")

        M_cutn, M1n, sigman, alphan, kappan \
        = map(design.get, ('M_cut', 'M1', 'sigma', 'alpha', 'kappa'))
        alpha_cn, alpha_sn, sn, s_vn, s_pn, s_rn, Acn, Asn, Bcn, Bsn \
        = map(decorations.get, 
        ('alpha_c', 'alpha_s', 's', 's_v', 's_p', 's_r', 'Acent', 'Asat', 'Bcent', 'Bsat'))    
    
        if params['rsd']:
            rsd_string = "_rsd"
        else:
            rsd_string = ""

        outdir = savedir / ("rockstar_"+str(np.log10(M_cutn))[0:10]+\
        "_"+str(np.log10(M1n))[0:10]+"_"+str(sigman)[0:6]+"_"+\
        str(alphan)[0:6]+"_"+str(kappan)[0:6]+"_decor_"+str(alpha_cn)+"_"+str(alpha_sn)\
        +"_"+str(sn)+"_"+str(s_vn)+"_"+str(s_pn)+"_"+str(s_rn)+\
        "_"+str(Acn)+"_"+str(Asn)+"_"+str(Bcn)+"_"+str(Bsn)+rsd_string)

        # create directories if not existing
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        # save to file 
        ascii.write([cent_pos[:, 0], cent_pos[:, 1], cent_pos[:, 2], 
            cent_vel[:, 0], cent_vel[:, 1], cent_vel[:, 2], cent_mass, cent_id], 
            outdir / ("gals_cent.dat"), names = ['x_gal', 'y_gal', 'z_gal', 
            'vx_gal', 'vy_gal', 'vz_gal', 'mass_halo', 'id_halo'], overwrite = True)
        ascii.write([sat_pos[:, 0], sat_pos[:, 1], sat_pos[:, 2], 
            sat_vel[:, 0], sat_vel[:, 1], sat_vel[:, 2], sat_mass, sat_id], 
            outdir / ("gals_sat.dat"), names = ['x_gal', 'y_gal', 'z_gal', 
            'vx_gal', 'vy_gal', 'vz_gal', 'mass_halo', 'id_halo'], overwrite = True)
    return cent_pos, cent_vel, cent_mass, cent_id, sat_pos, sat_vel, sat_mass, sat_id

