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
def n_sat_LRG_modified(M_h, logM_cut, M_cut, M_1, sigma, alpha, kappa): 
    """
    Standard Zheng et al. (2005) satellite HOD parametrization for LRGs, modified with n_cent_LRG
    """
    return ((M_h - kappa*M_cut)/M_1)**alpha*0.5*math.erfc((logM_cut - np.log10(M_h))/(1.41421356*sigma))


@njit(fastmath=True)
def n_cen_LRG(M_h, logM_cut, sigma): 
    """
    Standard Zheng et al. (2005) central HOD parametrization for LRGs.
    """
    return 0.5*math.erfc((logM_cut - np.log10(M_h))/(1.41421356*sigma))

@njit(fastmath=True)
def N_sat_generic(M_h, M_cut, kappa, M_1, alpha, A_s=1.):
    """
    Standard Zheng et al. (2005) satellite HOD parametrization for all tracers with an optional amplitude parameter, A_s.
    """
    return A_s*((M_h-kappa*M_cut)/M_1)**alpha

@njit(fastmath=True)
def N_cen_ELG_v1(M_h, p_max, Q, logM_cut, sigma, gamma):
    """
    HOD function for ELG centrals taken from arXiv:1910.05095.
    """
    logM_h = np.log(M_h)
    phi = phi_fun(logM_h, logM_cut, sigma)
    Phi = Phi_fun(logM_h, logM_cut, sigma, gamma)
    A = A_fun(p_max, Q, phi, Phi)
    return 2.*A*phi*Phi + 0.5/Q*(1 + math.erf((logM_h-logM_cut)*100))

@njit(fastmath=True)
def N_cen_ELG_v2(M_h, p_max, logM_cut, sigma, gamma):
    """
    HOD function for ELG centrals taken from arXiv:2007.09012.
    """
    logM_h = np.log10(M_h)
    if logM_h <= logM_cut:
        return p_max*Gaussian_fun(logM_h, logM_cut, sigma)
    else:
        return p_max*(M_h/10**logM_cut)**gamma/(2.5066283*sigma)

@njit(fastmath=True)
def N_cen_QSO(M_h, p_max, logM_cut, sigma):
    """
    HOD function (Zheng et al. (2005) with p_max) for QSO centrals taken from arXiv:2007.09012.
    """
    return 0.5*p_max*(1 + math.erf((np.log10(M_h)-logM_cut)/1.41421356/sigma))


@njit(fastmath=True)
def phi_fun(logM_h, logM_cut, sigma):
    """
    Aiding function for N_cen_ELG_v1().
    """
    phi = Gaussian_fun(logM_h, logM_cut, sigma)
    return phi

@njit(fastmath=True)
def Phi_fun(logM_h, logM_cut, sigma, gamma):
    """
    Aiding function for N_cen_ELG_v1().
    """
    x = gamma*(logM_h-logM_cut)/sigma
    Phi = 0.5*(1 + math.erf(x/np.sqrt(2)))
    return Phi
    
@njit(fastmath=True)
def A_fun(p_max, Q, phi, Phi):
    """
    Aiding function for N_cen_ELG_v1().
    """
    A = (p_max-1./Q)
    return A
    
@njit(fastmath=True)
def Gaussian_fun(x, mean, sigma):
    """
    Gaussian function with centered at `mean' with standard deviation `sigma'.
    """
    return 0.3989422804014327/sigma*np.exp(-(x - mean)**2/2/sigma**2)


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
    LRG_design_array, ELG_design_array, QSO_design_array, ic, alpha_c, Ac, Bc, rsd, inv_velz2kms, lbox,
    want_LRG, want_ELG, want_QSO):
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
    logM_cut_L, logM1_L, sigma_L, alpha_L, kappa_L = \
    LRG_design_array[0], LRG_design_array[1], LRG_design_array[2], LRG_design_array[3], LRG_design_array[4]

    pmax_E, Q_E, logM_cut_E, kappa_E, sigma_E, logM1_E, alpha_E, gamma_E, As_E = \
    ELG_design_array[0], ELG_design_array[1], ELG_design_array[2], ELG_design_array[3], ELG_design_array[4],\
    ELG_design_array[5], ELG_design_array[6], ELG_design_array[7], ELG_design_array[8]

    pmax_Q, logM_cut_Q, kappa_Q, sigma_Q, logM1_Q, alpha_Q, As_Q = \
    QSO_design_array[0], QSO_design_array[1], QSO_design_array[2], QSO_design_array[3], QSO_design_array[4],\
    QSO_design_array[5], QSO_design_array[6]

    H = len(mass)

    Nthread = numba.get_num_threads()
    Nout = np.zeros((Nthread, 8), dtype = np.int64)
    hstart = np.rint(np.linspace(0, H, Nthread + 1)) # starting index of each thread

    keep = np.empty(H, dtype = np.int8) # mask array tracking which halos to keep

    # figuring out the number of halos kept for each thread
    for tid in numba.prange(Nthread):
        for i in range(hstart[tid], hstart[tid + 1]):
            # first create the markers between 0 and 1 for different tracers
            LRG_marker = 0
            if want_LRG:
                # do assembly bias and secondary bias
                logM_cut_L_temp = logM_cut_L + Ac * deltac[i] + Bc * fenv[i]
                LRG_marker += n_cen_LRG(mass[i], logM_cut_L_temp, sigma_L) * ic * multis[i]
            ELG_marker = LRG_marker
            if want_ELG:
                ELG_marker += N_cen_ELG_v1(mass[i], pmax_E, Q_E, logM_cut_E, sigma_E, gamma_E) * multis[i]
            QSO_marker = ELG_marker
            if want_QSO:
                QSO_marker += N_cen_QSO(mass[i], pmax_Q, logM_cut_Q, sigma_Q)

            if randoms[i] <= LRG_marker:
                Nout[tid, 0] += 1 # counting
                keep[i] = 1
            elif randoms[i] <= ELG_marker:
                Nout[tid, 0] += 1 # counting
                keep[i] = 2
            elif randoms[i] <= QSO_marker:
                Nout[tid, 0] += 1 # counting
                keep[i] = 3
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
    gtype = np.empty(gstart[-1], dtype = ids.dtype)

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
                gtype[j] = keep[i]
                j += 1
        # assert j == gstart[tid + 1]

    return gpos, gvel, gmass, gid, gtype


@njit(parallel = True, fastmath = True)
def gen_sats(ppos, pvel, hvel, hmass, hid, weights, randoms, hdeltac, hfenv, 
    enable_ranks, ranks, ranksv, ranksp, ranksr, 
    LRG_design_array, LRG_decorations_array, ELG_design_array, QSO_design_array, 
    rsd, inv_velz2kms, lbox, Mpart, want_LRG, want_ELG, want_QSO):

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
    logM_cut_L, logM1_L, sigma_L, alpha_L, kappa_L = \
    LRG_design_array[0], LRG_design_array[1], LRG_design_array[2], LRG_design_array[3], LRG_design_array[4]
    alpha_s, s, s_v, s_p, s_r, Ac, As, Bc, Bs, ic = \
    LRG_decorations_array[1], LRG_decorations_array[2], LRG_decorations_array[3], LRG_decorations_array[4], \
    LRG_decorations_array[5], LRG_decorations_array[6], LRG_decorations_array[7], LRG_decorations_array[8], \
    LRG_decorations_array[9], LRG_decorations_array[10]

    pmax_E, Q_E, logM_cut_E, kappa_E, sigma_E, logM1_E, alpha_E, gamma_E, As_E = \
    ELG_design_array[0], ELG_design_array[1], ELG_design_array[2], ELG_design_array[3], ELG_design_array[4],\
    ELG_design_array[5], ELG_design_array[6], ELG_design_array[7], ELG_design_array[8]

    pmax_Q, logM_cut_Q, kappa_Q, sigma_Q, logM1_Q, alpha_Q, As_Q = \
    QSO_design_array[0], QSO_design_array[1], QSO_design_array[2], QSO_design_array[3], QSO_design_array[4],\
    QSO_design_array[5], QSO_design_array[6]

    H = len(hmass) # num of particles

    Nthread = numba.get_num_threads()
    Nout = np.zeros((Nthread, 8), dtype = np.int64)
    hstart = np.rint(np.linspace(0, H, Nthread + 1)) # starting index of each thread

    keep = np.empty(H, dtype = np.int8) # mask array tracking which halos to keep

    # figuring out the number of particles kept for each thread
    for tid in numba.prange(Nthread): #numba.prange(Nthread):
        for i in range(hstart[tid], hstart[tid + 1]):
            # print(logM1, As, hdeltac[i], Bs, hfenv[i])
            LRG_marker = 0
            if want_LRG:
                M1_L_temp = 10**(logM1_L + As * hdeltac[i] + Bs * hfenv[i])
                logM_cut_L_temp = logM_cut_L + Ac * hdeltac[i] + Bc * hfenv[i]
                base_p = n_sat_LRG_modified(hmass[i], logM_cut_L_temp, 10**logM_cut_L_temp, M1_L_temp, sigma_L, alpha_L, kappa_L)\
                 * weights[i] * ic
                if enable_ranks:
                    decorator = 1 + s * ranks[i] + s_v * ranksv[i] + s_p * ranksp[i] + s_r * ranksr[i]
                    exp_sat = base_p * decorator
                else:
                    exp_sat = base_p
                LRG_marker += exp_sat
            ELG_marker = LRG_marker
            if want_ELG:
                ELG_marker += N_sat_generic(hmass[i], 10**logM_cut_E, kappa_E, 10**logM1_E, alpha_E, As_E) * weights[i]
            QSO_marker = ELG_marker
            if want_QSO:
                QSO_marker += N_sat_generic(hmass[i], 10**logM_cut_Q, kappa_Q, 10**logM1_Q, alpha_Q, As_Q) * weights[i]

            if randoms[i] <= LRG_marker:
                Nout[tid, 0] += 1 # counting
                keep[i] = 1
            elif randoms[i] <= ELG_marker:
                Nout[tid, 0] += 1 # counting
                keep[i] = 2
            elif randoms[i] <= QSO_marker:
                Nout[tid, 0] += 1 # counting
                keep[i] = 3    
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
    gtype = np.empty(gstart[-1], dtype = hid.dtype)

    # fill in the galaxy arrays
    for tid in numba.prange(Nthread):
        j = gstart[tid]
        for i in range(hstart[tid], hstart[tid + 1]):
            if keep[i]:
                for k in range(3):
                    gpos[j, k] = ppos[i, k]
                    gvel[j, k] = hvel[i, k] + alpha_s * (pvel[i, k] - hvel[i, k]) # velocity bias
                if rsd:
                    gpos[j, 2] = wrap(gpos[j, 2] + gvel[j, 2] * inv_velz2kms, lbox)
                gmass[j] = hmass[i]
                gid[j] = hid[i]
                gtype[j] = keep[i]
                j += 1
        # assert j == gstart[tid + 1]

    return gpos, gvel, gmass, gid, gtype




def gen_gals(halos_array, subsample, LRG_HOD, ELG_HOD, QSO_HOD, params, enable_ranks, rsd,
    want_LRG, want_ELG, want_QSO):
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

    # LRG design and decorations
    logM_cut_L, logM1_L, sigma_L, alpha_L, kappa_L = \
    map(LRG_HOD.get, ('logM_cut', 
                      'logM1', 
                      'sigma', 
                      'alpha', 
                      'kappa'))
    LRG_design_array = np.array([logM_cut_L, logM1_L, sigma_L, alpha_L, kappa_L])

    alpha_c, alpha_s, s, s_v, s_p, s_r, Ac, As, Bc, Bs, ic = \
    map(LRG_HOD.get, ('alpha_c', 
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
    LRG_decorations_array = np.array([alpha_c, alpha_s, s, s_v, s_p, s_r, Ac, As, Bc, Bs, ic])

    # ELG desig
    pmax_E, Q_E, logM_cut_E, kappa_E, sigma_E, logM1_E, alpha_E, gamma_E, As_E = \
    map(ELG_HOD.get, ('p_max',
                    'Q',
                    'logM_cut',
                    'kappa',
                    'sigma',
                    'logM1',
                    'alpha',
                    'gamma',
                    'A_s'))
    ELG_design_array = np.array([pmax_E, Q_E, logM_cut_E, kappa_E, sigma_E, logM1_E, alpha_E, gamma_E, As_E])

    # QSO desig
    pmax_Q, logM_cut_Q, kappa_Q, sigma_Q, logM1_Q, alpha_Q, As_Q = \
    map(QSO_HOD.get, ('p_max',
                    'logM_cut',
                    'kappa',
                    'sigma',
                    'logM1',
                    'alpha',
                    'A_s'))
    QSO_design_array = np.array([pmax_Q, logM_cut_Q, kappa_Q, sigma_Q, logM1_Q, alpha_Q, As_Q])

    start = time.time()

    velz2kms = params['velz2kms']
    inv_velz2kms = 1/velz2kms
    lbox = params['Lbox']
    # for each halo, generate central galaxies and output to file
    cent_pos, cent_vel, cent_mass, cent_id, cent_type = \
    gen_cent(halos_array['hpos'], halos_array['hvel'], halos_array['hmass'], halos_array['hid'], halos_array['hmultis'], 
     halos_array['hrandoms'], halos_array['hveldev'], halos_array['hdeltac'], halos_array['hfenv'], 
     LRG_design_array, ELG_design_array, QSO_design_array, ic, alpha_c, Ac, Bc, rsd, inv_velz2kms, lbox,
     want_LRG, want_ELG, want_QSO)
    print("generating centrals took ", time.time() - start, "number of centrals ", len(cent_mass))


    start = time.time()
    sat_pos, sat_vel, sat_mass, sat_id, sat_type = \
    gen_sats(subsample['ppos'], subsample['pvel'], subsample['phvel'], subsample['phmass'], subsample['phid'], 
        subsample['pweights'], subsample['prandoms'], subsample['pdeltac'], subsample['pfenv'], 
        enable_ranks, subsample['pranks'], subsample['pranksv'], subsample['pranksp'], subsample['pranksr'],
        LRG_design_array, LRG_decorations_array, ELG_design_array, QSO_design_array, rsd, inv_velz2kms, lbox, params['Mpart'],
        want_LRG, want_ELG, want_QSO)

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

    return cent_pos, cent_vel, cent_mass, cent_id, cent_type, sat_pos, sat_vel, sat_mass, sat_id, sat_type


def gen_gal_cat(halo_data, particle_data, LRG_HOD, ELG_HOD, QSO_HOD,
    params, enable_ranks = False, rsd = True, 
    want_LRG = True, want_ELG = False, want_QSO = False,
    write_to_disk = False, savedir = ("./")):
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
    cent_pos, cent_vel, cent_mass, cent_id, cent_type, sat_pos, sat_vel, sat_mass, sat_id, sat_type \
     = gen_gals(halo_data, particle_data, LRG_HOD, ELG_HOD, QSO_HOD, 
        params, enable_ranks, rsd, want_LRG, want_ELG, want_QSO)

    print("generated ", np.shape(cent_pos)[0], " centrals and ", np.shape(sat_pos)[0], " satellites.")
    if write_to_disk:
        print("outputting galaxies to disk")

        logM_cutn, logM1n, sigman, alphan, kappan \
        = map(LRG_HOD.get, ('logM_cut', 'logM1', 'sigma', 'alpha', 'kappa'))
        alpha_cn, alpha_sn, sn, s_vn, s_pn, s_rn, Acn, Asn, Bcn, Bsn \
        = map(LRG_HOD.get, 
        ('alpha_c', 'alpha_s', 's', 's_v', 's_p', 's_r', 'Acent', 'Asat', 'Bcent', 'Bsat'))    
    
        if params['rsd']:
            rsd_string = "_rsd"
        else:
            rsd_string = ""

        outdir = savedir / ("galaxies_"+str(logM_cutn)[0:10]+\
        "_"+str(logM1n)[0:10]+"_"+str(sigman)[0:6]+"_"+\
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
    return cent_pos, cent_vel, cent_mass, cent_id, cent_type, sat_pos, sat_vel, sat_mass, sat_id, sat_type

