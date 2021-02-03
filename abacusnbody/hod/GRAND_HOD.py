import os
import sys
import time
from pathlib import Path
import pkgutil
import math
from math import erfc

import numpy as np
from astropy.table import Table
from astropy.io import ascii

import numba
from numba import njit, types, jit
from numba.typed import Dict

# import yaml
# config = yaml.load(open('config/abacus_hod.yaml'))
# numba.set_num_threads(16)
float_array = types.float64[:]
int_array = types.int64[:]

@njit(fastmath=True)
def n_sat_LRG_modified(M_h, logM_cut, M_cut, M_1, sigma, alpha, kappa): 
    """
    Standard Zheng et al. (2005) satellite HOD parametrization for LRGs, modified with n_cent_LRG
    """
    if M_h - kappa*M_cut < 0:
        return 0
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
    if M_h - kappa*M_cut < 0:
        return 0
    return A_s*((M_h-kappa*M_cut)/M_1)**alpha

@njit(fastmath=True)
def N_cen_ELG_v1(M_h, p_max, Q, logM_cut, sigma, gamma):
    """
    HOD function for ELG centrals taken from arXiv:1910.05095.
    """
    logM_h = np.log10(M_h)
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
    LRG_design_array, LRG_decorations_array, ELG_design_array, 
    ELG_decorations_array, QSO_design_array, QSO_decorations_array, 
    rsd, inv_velz2kms, lbox, want_LRG, want_ELG, want_QSO, Nthread):
    """
    Generate central galaxies in place in memory with a two pass numba parallel implementation. 
    """

    # parse out the hod parameters 
    logM_cut_L, logM1_L, sigma_L, alpha_L, kappa_L = \
        LRG_design_array[0], LRG_design_array[1], LRG_design_array[2], LRG_design_array[3], LRG_design_array[4]
    ic_L, alpha_c_L, Ac_L, Bc_L = LRG_decorations_array[10], LRG_decorations_array[0], \
        LRG_decorations_array[6], LRG_decorations_array[8]

    pmax_E, Q_E, logM_cut_E, kappa_E, sigma_E, logM1_E, alpha_E, gamma_E, A_E = \
        ELG_design_array[0], ELG_design_array[1], ELG_design_array[2], ELG_design_array[3], ELG_design_array[4],\
        ELG_design_array[5], ELG_design_array[6], ELG_design_array[7], ELG_design_array[8]
    alpha_c_E, Ac_E, Bc_E = ELG_decorations_array[0], ELG_decorations_array[6], ELG_decorations_array[8]

    pmax_Q, logM_cut_Q, kappa_Q, sigma_Q, logM1_Q, alpha_Q, A_Q = \
        QSO_design_array[0], QSO_design_array[1], QSO_design_array[2], QSO_design_array[3], QSO_design_array[4],\
        QSO_design_array[5], QSO_design_array[6]
    alpha_c_Q, Ac_Q, Bc_Q = QSO_decorations_array[0], QSO_decorations_array[6], QSO_decorations_array[8]

    H = len(mass)

    numba.set_num_threads(Nthread)
    Nout = np.zeros((Nthread, 3, 8), dtype = np.int64)
    hstart = np.rint(np.linspace(0, H, Nthread + 1)) # starting index of each thread

    keep = np.empty(H, dtype = np.int8) # mask array tracking which halos to keep

    # figuring out the number of halos kept for each thread
    for tid in numba.prange(Nthread):
        for i in range(hstart[tid], hstart[tid + 1]):
            # first create the markers between 0 and 1 for different tracers
            LRG_marker = 0
            if want_LRG:
                # do assembly bias and secondary bias
                logM_cut_L_temp = logM_cut_L + Ac_L * deltac[i] + Bc_L * fenv[i]
                LRG_marker += n_cen_LRG(mass[i], logM_cut_L_temp, sigma_L) * ic_L * multis[i]
            ELG_marker = LRG_marker
            if want_ELG:
                logM_cut_E_temp = logM_cut_E + Ac_E * deltac[i] + Bc_E * fenv[i]
                ELG_marker += N_cen_ELG_v1(mass[i], pmax_E, Q_E, logM_cut_E_temp, sigma_E, gamma_E) * multis[i]
            QSO_marker = ELG_marker
            if want_QSO:
                logM_cut_Q_temp = logM_cut_Q + Ac_Q * deltac[i] + Bc_Q * fenv[i]
                QSO_marker += N_cen_QSO(mass[i], pmax_Q, logM_cut_Q, sigma_Q)

            if randoms[i] <= LRG_marker:
                Nout[tid, 0, 0] += 1 # counting
                keep[i] = 1
            elif randoms[i] <= ELG_marker:
                Nout[tid, 1, 0] += 1 # counting
                keep[i] = 2
            elif randoms[i] <= QSO_marker:
                Nout[tid, 2, 0] += 1 # counting
                keep[i] = 3
            else:
                keep[i] = 0

    # compose galaxy array, first create array of galaxy starting indices for the threads
    gstart = np.empty((Nthread + 1, 3), dtype = np.int64)
    gstart[0, :] = 0
    gstart[1:, 0] = Nout[:, 0, 0].cumsum()
    gstart[1:, 1] = Nout[:, 1, 0].cumsum()
    gstart[1:, 2] = Nout[:, 2, 0].cumsum()

    # galaxy arrays
    N_lrg = gstart[-1, 0]
    lrg_x = np.empty(N_lrg, dtype = mass.dtype)
    lrg_y = np.empty(N_lrg, dtype = mass.dtype)
    lrg_z = np.empty(N_lrg, dtype = mass.dtype)
    lrg_vx = np.empty(N_lrg, dtype = mass.dtype)
    lrg_vy = np.empty(N_lrg, dtype = mass.dtype)
    lrg_vz = np.empty(N_lrg, dtype = mass.dtype)
    lrg_mass = np.empty(N_lrg, dtype = mass.dtype)
    lrg_id = np.empty(N_lrg, dtype = ids.dtype)

    # galaxy arrays
    N_elg = gstart[-1, 1]
    elg_x = np.empty(N_elg, dtype = mass.dtype)
    elg_y = np.empty(N_elg, dtype = mass.dtype)
    elg_z = np.empty(N_elg, dtype = mass.dtype)
    elg_vx = np.empty(N_elg, dtype = mass.dtype)
    elg_vy = np.empty(N_elg, dtype = mass.dtype)
    elg_vz = np.empty(N_elg, dtype = mass.dtype)
    elg_mass = np.empty(N_elg, dtype = mass.dtype)
    elg_id = np.empty(N_elg, dtype = ids.dtype)

    # galaxy arrays
    N_qso = gstart[-1, 2]
    qso_x = np.empty(N_qso, dtype = mass.dtype)
    qso_y = np.empty(N_qso, dtype = mass.dtype)
    qso_z = np.empty(N_qso, dtype = mass.dtype)
    qso_vx = np.empty(N_qso, dtype = mass.dtype)
    qso_vy = np.empty(N_qso, dtype = mass.dtype)
    qso_vz = np.empty(N_qso, dtype = mass.dtype)
    qso_mass = np.empty(N_qso, dtype = mass.dtype)
    qso_id = np.empty(N_qso, dtype = ids.dtype)

    # fill in the galaxy arrays
    for tid in numba.prange(Nthread):
        j1, j2, j3 = gstart[tid]
        for i in range(hstart[tid], hstart[tid + 1]):
            if keep[i] == 1:
                # loop thru three directions to assign galaxy velocities and positions
                lrg_x[j1] = pos[i,0]
                lrg_vx[j1] = vel[i,0] + alpha_c_L * vdev[i] # velocity bias
                lrg_y[j1] = pos[i,1]
                lrg_vy[j1] = vel[i,1] + alpha_c_L * vdev[i] # velocity bias
                lrg_z[j1] = pos[i,2]
                lrg_vz[j1] = vel[i,2] + alpha_c_L * vdev[i] # velocity bias
                # rsd only applies to the z direction
                if rsd:
                    lrg_z[j1] = wrap(pos[i,2] + lrg_vz[j1] * inv_velz2kms, lbox)
                lrg_mass[j1] = mass[i]
                lrg_id[j1] = ids[i]
                j1 += 1
            elif keep[i] == 2:
                # loop thru three directions to assign galaxy velocities and positions
                elg_x[j2] = pos[i,0]
                elg_vx[j2] = vel[i,0] + alpha_c_E * vdev[i] # velocity bias
                elg_y[j2] = pos[i,1]
                elg_vy[j2] = vel[i,1] + alpha_c_E * vdev[i] # velocity bias
                elg_z[j2] = pos[i,2]
                elg_vz[j2] = vel[i,2] + alpha_c_E * vdev[i] # velocity bias
                # rsd only applies to the z direction
                if rsd:
                    elg_z[j2] = wrap(pos[i,2] + elg_vz[j2] * inv_velz2kms, lbox)
                elg_mass[j2] = mass[i]
                elg_id[j2] = ids[i]
                j2 += 1
            elif keep[i] == 3:
                # loop thru three directions to assign galaxy velocities and positions
                qso_x[j3] = pos[i,0]
                qso_vx[j3] = vel[i,0] + alpha_c_Q * vdev[i] # velocity bias
                qso_y[j3] = pos[i,1]
                qso_vy[j3] = vel[i,1] + alpha_c_Q * vdev[i] # velocity bias
                qso_z[j3] = pos[i,2]
                qso_vz[j3] = vel[i,2] + alpha_c_Q * vdev[i] # velocity bias
                # rsd only applies to the z direction
                if rsd:
                    qso_z[j3] = wrap(pos[i,2] + qso_vz[j3] * inv_velz2kms, lbox)
                qso_mass[j3] = mass[i]
                qso_id[j3] = ids[i]
                j3 += 1
        # assert j == gstart[tid + 1]

    LRG_dict = Dict.empty(key_type = types.unicode_type, value_type = float_array)
    ELG_dict = Dict.empty(key_type = types.unicode_type, value_type = float_array)
    QSO_dict = Dict.empty(key_type = types.unicode_type, value_type = float_array)
    ID_dict = Dict.empty(key_type = types.unicode_type, value_type = int_array)
    LRG_dict['x'] = lrg_x
    LRG_dict['y'] = lrg_y
    LRG_dict['z'] = lrg_z
    LRG_dict['vx'] = lrg_vx
    LRG_dict['vy'] = lrg_vy
    LRG_dict['vz'] = lrg_vz
    LRG_dict['mass'] = lrg_mass
    ID_dict['LRG'] = lrg_id

    ELG_dict['x'] = elg_x
    ELG_dict['y'] = elg_y
    ELG_dict['z'] = elg_z
    ELG_dict['vx'] = elg_vx
    ELG_dict['vy'] = elg_vy
    ELG_dict['vz'] = elg_vz
    ELG_dict['mass'] = elg_mass
    ID_dict['ELG'] = elg_id

    QSO_dict['x'] = qso_x
    QSO_dict['y'] = qso_y
    QSO_dict['z'] = qso_z
    QSO_dict['vx'] = qso_vx
    QSO_dict['vy'] = qso_vy
    QSO_dict['vz'] = qso_vz
    QSO_dict['mass'] = qso_mass
    ID_dict['QSO'] = qso_id
    return LRG_dict, ELG_dict, QSO_dict, ID_dict


@njit(parallel = True, fastmath = True)
def gen_sats(ppos, pvel, hvel, hmass, hid, weights, randoms, hdeltac, hfenv, 
    enable_ranks, ranks, ranksv, ranksp, ranksr, 
    LRG_design_array, LRG_decorations_array, ELG_design_array, ELG_decorations_array,
    QSO_design_array, QSO_decorations_array,
    rsd, inv_velz2kms, lbox, Mpart, want_LRG, want_ELG, want_QSO, Nthread):

    """
    Generate satellite galaxies in place in memory with a two pass numba parallel implementation. 
    """

    # standard hod design
    logM_cut_L, logM1_L, sigma_L, alpha_L, kappa_L = \
        LRG_design_array[0], LRG_design_array[1], LRG_design_array[2], LRG_design_array[3], LRG_design_array[4]
    alpha_s_L, s_L, s_v_L, s_p_L, s_r_L, Ac_L, As_L, Bc_L, Bs_L, ic_L = \
        LRG_decorations_array[1], LRG_decorations_array[2], LRG_decorations_array[3], LRG_decorations_array[4], \
        LRG_decorations_array[5], LRG_decorations_array[6], LRG_decorations_array[7], LRG_decorations_array[8], \
        LRG_decorations_array[9], LRG_decorations_array[10]

    pmax_E, Q_E, logM_cut_E, kappa_E, sigma_E, logM1_E, alpha_E, gamma_E, A_E = \
        ELG_design_array[0], ELG_design_array[1], ELG_design_array[2], ELG_design_array[3], ELG_design_array[4],\
        ELG_design_array[5], ELG_design_array[6], ELG_design_array[7], ELG_design_array[8]
    alpha_s_E, s_E, s_v_E, s_p_E, s_r_E, Ac_E, As_E, Bc_E, Bs_E = \
        ELG_decorations_array[1], ELG_decorations_array[2], ELG_decorations_array[3], ELG_decorations_array[4], \
        ELG_decorations_array[5], ELG_decorations_array[6], ELG_decorations_array[7], ELG_decorations_array[8], \
        ELG_decorations_array[9]

    pmax_Q, logM_cut_Q, kappa_Q, sigma_Q, logM1_Q, alpha_Q, A_Q = \
        QSO_design_array[0], QSO_design_array[1], QSO_design_array[2], QSO_design_array[3], QSO_design_array[4],\
        QSO_design_array[5], QSO_design_array[6]
    alpha_s_Q, s_Q, s_v_Q, s_p_Q, s_r_Q, Ac_Q, As_Q, Bc_Q, Bs_Q = \
        QSO_decorations_array[1], QSO_decorations_array[2], QSO_decorations_array[3], QSO_decorations_array[4], \
        QSO_decorations_array[5], QSO_decorations_array[6], QSO_decorations_array[7], QSO_decorations_array[8], \
        QSO_decorations_array[9]

    H = len(hmass) # num of particles

    numba.set_num_threads(Nthread)
    Nout = np.zeros((Nthread, 3, 8), dtype = np.int64)
    hstart = np.rint(np.linspace(0, H, Nthread + 1)) # starting index of each thread

    keep = np.empty(H, dtype = np.int8) # mask array tracking which halos to keep

    # figuring out the number of particles kept for each thread
    for tid in numba.prange(Nthread): #numba.prange(Nthread):
        for i in range(hstart[tid], hstart[tid + 1]):
            # print(logM1, As, hdeltac[i], Bs, hfenv[i])
            LRG_marker = 0
            if want_LRG:
                M1_L_temp = 10**(logM1_L + As_L * hdeltac[i] + Bs_L * hfenv[i])
                logM_cut_L_temp = logM_cut_L + Ac_L * hdeltac[i] + Bc_L * hfenv[i]
                base_p_L = n_sat_LRG_modified(hmass[i], logM_cut_L_temp, 
                    10**logM_cut_L_temp, M1_L_temp, sigma_L, alpha_L, kappa_L) * weights[i] * ic_L
                if enable_ranks:
                    decorator_L = 1 + s_L * ranks[i] + s_v_L * ranksv[i] + s_p_L * ranksp[i] + s_r_L * ranksr[i]
                    exp_sat = base_p_L * decorator_L
                else:
                    exp_sat = base_p_L
                LRG_marker += exp_sat
            ELG_marker = LRG_marker
            if want_ELG:
                M1_E_temp = 10**(logM1_E + As_E * hdeltac[i] + Bs_E * hfenv[i])
                logM_cut_E_temp = logM_cut_E + Ac_E * hdeltac[i] + Bc_E * hfenv[i]
                base_p_E = N_sat_generic(
                    hmass[i], 10**logM_cut_E_temp, kappa_E, M1_E_temp, alpha_E, A_E) * weights[i]
                if enable_ranks:
                    decorator_E = 1 + s_E * ranks[i] + s_v_E * ranksv[i] + s_p_E * ranksp[i] + s_r_E * ranksr[i]
                    exp_sat = base_p_E * decorator_E
                else:
                    exp_sat = base_p_E
                ELG_marker += exp_sat
            QSO_marker = ELG_marker
            if want_QSO:
                M1_Q_temp = 10**(logM1_Q + As_Q * hdeltac[i] + Bs_Q * hfenv[i])
                logM_cut_Q_temp = logM_cut_Q + Ac_Q * hdeltac[i] + Bc_Q * hfenv[i]
                base_p_Q = N_sat_generic(
                    hmass[i], 10**logM_cut_Q_temp, kappa_Q, M1_Q_temp, alpha_Q, A_Q) * weights[i]
                if enable_ranks:
                    decorator_Q = 1 + s_Q * ranks[i] + s_v_Q * ranksv[i] + s_p_Q * ranksp[i] + s_r_Q * ranksr[i]
                    exp_sat = base_p_Q * decorator_Q
                else:
                    exp_sat = base_p_Q
                QSO_marker += exp_sat

            if randoms[i] <= LRG_marker:
                Nout[tid, 0, 0] += 1 # counting
                keep[i] = 1
            elif randoms[i] <= ELG_marker:
                Nout[tid, 1, 0] += 1 # counting
                keep[i] = 2
            elif randoms[i] <= QSO_marker:
                Nout[tid, 2, 0] += 1 # counting
                keep[i] = 3    
            else:
                keep[i] = 0

    # compose galaxy array, first create array of galaxy starting indices for the threads
    gstart = np.empty((Nthread + 1, 3), dtype = np.int64)
    gstart[0, :] = 0
    gstart[1:, 0] = Nout[:, 0, 0].cumsum()
    gstart[1:, 1] = Nout[:, 1, 0].cumsum()
    gstart[1:, 2] = Nout[:, 2, 0].cumsum()

    # galaxy arrays
    N_lrg = gstart[-1, 0]
    lrg_x = np.empty(N_lrg, dtype = hmass.dtype)
    lrg_y = np.empty(N_lrg, dtype = hmass.dtype)
    lrg_z = np.empty(N_lrg, dtype = hmass.dtype)
    lrg_vx = np.empty(N_lrg, dtype = hmass.dtype)
    lrg_vy = np.empty(N_lrg, dtype = hmass.dtype)
    lrg_vz = np.empty(N_lrg, dtype = hmass.dtype)
    lrg_mass = np.empty(N_lrg, dtype = hmass.dtype)
    lrg_id = np.empty(N_lrg, dtype = hid.dtype)

    # galaxy arrays
    N_elg = gstart[-1, 1]
    elg_x = np.empty(N_elg, dtype = hmass.dtype)
    elg_y = np.empty(N_elg, dtype = hmass.dtype)
    elg_z = np.empty(N_elg, dtype = hmass.dtype)
    elg_vx = np.empty(N_elg, dtype = hmass.dtype)
    elg_vy = np.empty(N_elg, dtype = hmass.dtype)
    elg_vz = np.empty(N_elg, dtype = hmass.dtype)
    elg_mass = np.empty(N_elg, dtype = hmass.dtype)
    elg_id = np.empty(N_elg, dtype = hid.dtype)

    # galaxy arrays
    N_qso = gstart[-1, 2]
    qso_x = np.empty(N_qso, dtype = hmass.dtype)
    qso_y = np.empty(N_qso, dtype = hmass.dtype)
    qso_z = np.empty(N_qso, dtype = hmass.dtype)
    qso_vx = np.empty(N_qso, dtype = hmass.dtype)
    qso_vy = np.empty(N_qso, dtype = hmass.dtype)
    qso_vz = np.empty(N_qso, dtype = hmass.dtype)
    qso_mass = np.empty(N_qso, dtype = hmass.dtype)
    qso_id = np.empty(N_qso, dtype = hid.dtype)

    # fill in the galaxy arrays
    for tid in numba.prange(Nthread):
        j1, j2, j3 = gstart[tid]
        for i in range(hstart[tid], hstart[tid + 1]):
            if keep[i] == 1:
                lrg_x[j1] = ppos[i, 0]
                lrg_vx[j1] = hvel[i, 0] + alpha_s_L * (pvel[i, 0] - hvel[i, 0]) # velocity bias
                lrg_y[j1] = ppos[i, 1]
                lrg_vy[j1] = hvel[i, 1] + alpha_s_L * (pvel[i, 1] - hvel[i, 1]) # velocity bias
                lrg_z[j1] = ppos[i, 2]
                lrg_vz[j1] = hvel[i, 2] + alpha_s_L * (pvel[i, 2] - hvel[i, 2]) # velocity bias
                if rsd:
                    lrg_z[j1] = wrap(lrg_z[j1] + lrg_vz[j1] * inv_velz2kms, lbox)
                lrg_mass[j1] = hmass[i]
                lrg_id[j1] = hid[i]
                j1 += 1
            elif keep[i] == 2:
                elg_x[j2] = ppos[i, 0]
                elg_vx[j2] = hvel[i, 0] + alpha_s_E * (pvel[i, 0] - hvel[i, 0]) # velocity bias
                elg_y[j2] = ppos[i, 1]
                elg_vy[j2] = hvel[i, 1] + alpha_s_E * (pvel[i, 1] - hvel[i, 1]) # velocity bias
                elg_z[j2] = ppos[i, 2]
                elg_vz[j2] = hvel[i, 2] + alpha_s_E * (pvel[i, 2] - hvel[i, 2]) # velocity bias
                if rsd:
                    elg_z[j2] = wrap(elg_z[j2] + elg_vz[j2] * inv_velz2kms, lbox)
                elg_mass[j2] = hmass[i]
                elg_id[j2] = hid[i]
                j2 += 1
            elif keep[i] == 3:
                qso_x[j3] = ppos[i, 0]
                qso_vx[j3] = hvel[i, 0] + alpha_s_Q * (pvel[i, 0] - hvel[i, 0]) # velocity bias
                qso_y[j3] = ppos[i, 1]
                qso_vy[j3] = hvel[i, 1] + alpha_s_Q * (pvel[i, 1] - hvel[i, 1]) # velocity bias
                qso_z[j3] = ppos[i, 2]
                qso_vz[j3] = hvel[i, 2] + alpha_s_Q * (pvel[i, 2] - hvel[i, 2]) # velocity bias
                if rsd:
                    qso_z[j3] = wrap(qso_z[j3] + qso_vz[j3] * inv_velz2kms, lbox)
                qso_mass[j3] = hmass[i]
                qso_id[j3] = hid[i]
                j3 += 1
        # assert j == gstart[tid + 1]

    LRG_dict = Dict.empty(key_type = types.unicode_type, value_type = float_array)
    ELG_dict = Dict.empty(key_type = types.unicode_type, value_type = float_array)
    QSO_dict = Dict.empty(key_type = types.unicode_type, value_type = float_array)
    ID_dict = Dict.empty(key_type = types.unicode_type, value_type = int_array)
    LRG_dict['x'] = lrg_x
    LRG_dict['y'] = lrg_y
    LRG_dict['z'] = lrg_z
    LRG_dict['vx'] = lrg_vx
    LRG_dict['vy'] = lrg_vy
    LRG_dict['vz'] = lrg_vz
    LRG_dict['mass'] = lrg_mass
    ID_dict['LRG'] = lrg_id

    ELG_dict['x'] = elg_x
    ELG_dict['y'] = elg_y
    ELG_dict['z'] = elg_z
    ELG_dict['vx'] = elg_vx
    ELG_dict['vy'] = elg_vy
    ELG_dict['vz'] = elg_vz
    ELG_dict['mass'] = elg_mass
    ID_dict['ELG'] = elg_id

    QSO_dict['x'] = qso_x
    QSO_dict['y'] = qso_y
    QSO_dict['z'] = qso_z
    QSO_dict['vx'] = qso_vx
    QSO_dict['vy'] = qso_vy
    QSO_dict['vz'] = qso_vz
    QSO_dict['mass'] = qso_mass
    ID_dict['QSO'] = qso_id
    return LRG_dict, ELG_dict, QSO_dict, ID_dict

@njit(parallel = True, fastmath = True)
def fast_concatenate(array1, array2, Nthread):
    ''' Fast concatenate with numba parallel'''

    N1 = len(array1)
    N2 = len(array2)
    if N1 == 0:
        return array2
    elif N2 == 0:
        return array1

    final_array = np.empty(N1 + N2, dtype = array1.dtype)
    # if one thread, then no need to parallel
    if Nthread == 1:
        for i in range(N1):
            final_array[i] = array1[i]
        for j in range(N2):
            final_array[j + N1] = array2[j]
        return final_array

    numba.set_num_threads(Nthread)
    Nthread1 = int(np.floor(Nthread * N1 / (N1 + N2)))
    Nthread2 = Nthread - Nthread1
    hstart1 = np.rint(np.linspace(0, N1, Nthread1 + 1))
    hstart2 = np.rint(np.linspace(0, N2, Nthread2 + 1)) + N1

    for tid in numba.prange(Nthread): #numba.prange(Nthread):
        if tid < Nthread1:
            for i in range(hstart1[tid], hstart1[tid + 1]):
                final_array[i] = array1[i]
        else:
            for i in range(hstart2[tid - Nthread1], hstart2[tid + 1 - Nthread1]):
                final_array[i] = array2[i - N1]
    # final_array = np.concatenate((array1, array2))
    return final_array

def gen_gals(halos_array, subsample, tracers, params, Nthread, enable_ranks, rsd):
    """
    parse hod parameters, pass them on to central and satellite generators 
    and then format the results 

    Parameters
    ----------

    halos_array : dictionary of arrays 
        a dictionary of halo properties (pos, vel, mass, id, randoms, ...)

    subsample : dictionary of arrays
        a dictionary of particle propoerties (pos, vel, hmass, hid, Np, subsampling, randoms, ...)

    tracers : dictionary of dictionaries
        Dictionary of multi-tracer HODs

    enable_ranks : boolean
        Flag of whether to implement particle ranks. 

    rsd : boolean
        Flag of whether to implement RSD. 

    params : dict
        Dictionary of various simulation parameters. 

    """

    # B.H. TODO: pass as dictionary; make what's below more succinct
    for tracer in tracers.keys():
        if tracer == 'LRG':
            LRG_HOD = tracers[tracer]
        if tracer == 'ELG':
            ELG_HOD = tracers[tracer]
        if tracer == 'QSO':
            QSO_HOD = tracers[tracer]

    if 'LRG' in tracers.keys():
        want_LRG = True
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
                            'alpha_s',  
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
    else:
        # B.H. TODO: this will go when we switch to dictionaried and for loops
        want_LRG = False
        LRG_design_array = np.zeros(5)
        LRG_decorations_array = np.zeros(11)
        
    if 'ELG' in tracers.keys():
        # ELG design
        want_ELG = True
        pmax_E, Q_E, logM_cut_E, kappa_E, sigma_E, logM1_E, alpha_E, gamma_E, A_E = \
            map(ELG_HOD.get, ('p_max',
                            'Q',
                            'logM_cut',
                            'kappa',
                            'sigma',
                            'logM1',
                            'alpha',
                            'gamma',
                            'A_s'))
        ELG_design_array = np.array(
            [pmax_E, Q_E, logM_cut_E, kappa_E, sigma_E, logM1_E, alpha_E, gamma_E, A_E])
        alpha_c_E, alpha_s_E, s_E, s_v_E, s_p_E, s_r_E, Ac_E, As_E, Bc_E, Bs_E = \
            map(ELG_HOD.get, ('alpha_c', 
                            'alpha_s',  
                            's', 
                            's_v', 
                            's_p', 
                            's_r',
                            'Acent',
                            'Asat',
                            'Bcent',
                            'Bsat'))
        ELG_decorations_array = np.array(
            [alpha_c_E, alpha_s_E, s_E, s_v_E, s_p_E, s_r_E, Ac_E, As_E, Bc_E, Bs_E])
    else:
        # B.H. TODO: this will go when we switch to dictionaried and for loops
        ELG_design_array = np.zeros(9)
        ELG_decorations_array = np.zeros(10)
        want_ELG = False
        
    if 'QSO' in tracers.keys():
        # QSO design
        want_QSO = True
        pmax_Q, logM_cut_Q, kappa_Q, sigma_Q, logM1_Q, alpha_Q, A_Q = \
            map(QSO_HOD.get, ('p_max',
                            'logM_cut',
                            'kappa',
                            'sigma',
                            'logM1',
                            'alpha',
                            'A_s'))
        QSO_design_array = np.array(
            [pmax_Q, logM_cut_Q, kappa_Q, sigma_Q, logM1_Q, alpha_Q, A_Q])
        alpha_c_Q, alpha_s_Q, s_Q, s_v_Q, s_p_Q, s_r_Q, Ac_Q, As_Q, Bc_Q, Bs_Q = \
            map(QSO_HOD.get, ('alpha_c', 
                            'alpha_s',  
                            's', 
                            's_v', 
                            's_p', 
                            's_r',
                            'Acent',
                            'Asat',
                            'Bcent',
                            'Bsat'))
        QSO_decorations_array = np.array(
            [alpha_c_Q, alpha_s_Q, s_Q, s_v_Q, s_p_Q, s_r_Q, Ac_Q, As_Q, Bc_Q, Bs_Q])
    else:
        # B.H. TODO: this will go when we switch to dictionaried and for loops
        QSO_design_array = np.zeros(7)
        QSO_decorations_array = np.zeros(10)
        want_QSO = False
    
    start = time.time()

    velz2kms = params['velz2kms']
    inv_velz2kms = 1/velz2kms
    lbox = params['Lbox']
    # for each halo, generate central galaxies and output to file
    LRG_dict_cent, ELG_dict_cent, QSO_dict_cent, ID_dict_cent = \
    gen_cent(halos_array['hpos'], halos_array['hvel'], halos_array['hmass'], halos_array['hid'], halos_array['hmultis'], 
             halos_array['hrandoms'], halos_array['hveldev'], halos_array['hdeltac'], halos_array['hfenv'], 
             LRG_design_array, LRG_decorations_array, ELG_design_array, ELG_decorations_array, QSO_design_array, 
             QSO_decorations_array, rsd, inv_velz2kms, lbox, want_LRG, want_ELG, want_QSO, Nthread)
    print("generating centrals took ", time.time() - start)


    start = time.time()
    LRG_dict_sat, ELG_dict_sat, QSO_dict_sat, ID_dict_sat = \
    gen_sats(subsample['ppos'], subsample['pvel'], subsample['phvel'], subsample['phmass'], subsample['phid'], 
             subsample['pweights'], subsample['prandoms'], subsample['pdeltac'], subsample['pfenv'], 
             enable_ranks, subsample['pranks'], subsample['pranksv'], subsample['pranksp'], subsample['pranksr'],
             LRG_design_array, LRG_decorations_array, ELG_design_array, ELG_decorations_array,
             QSO_design_array, QSO_decorations_array, rsd, inv_velz2kms, lbox, params['Mpart'],
             want_LRG, want_ELG, want_QSO, Nthread)

    print("generating satellites took ", time.time() - start)

    # B.H. TODO: need a for loop above so we don't need to do this by hand
    HOD_dict_sat = {'LRG': LRG_dict_sat, 'ELG': ELG_dict_sat, 'QSO': QSO_dict_sat}
    HOD_dict_cent = {'LRG': LRG_dict_cent, 'ELG': ELG_dict_cent, 'QSO': QSO_dict_cent}
    
    # do a concatenate in numba parallel 
    start = time.time()
    HOD_dict = {}
    for tracer in tracers:
        tracer_dict = {'Ncent':len(HOD_dict_cent[tracer]['x'])}
        for k in HOD_dict_cent[tracer]:
            tracer_dict[k] = fast_concatenate(HOD_dict_cent[tracer][k], HOD_dict_sat[tracer][k], Nthread)
        tracer_dict['id'] = fast_concatenate(ID_dict_cent[tracer], ID_dict_sat[tracer], Nthread)
        print(tracer, "number of galaxies ", len(tracer_dict['x']), 
            ", satellite fraction ", len(HOD_dict_sat[tracer]['x'])/len(tracer_dict['x']))
        HOD_dict[tracer] = tracer_dict
    print("organizing outputs took ", time.time() - start)
    return HOD_dict


def gen_gal_cat(halo_data, particle_data, tracers, params, Nthread = 16,
    enable_ranks = False, rsd = True, write_to_disk = False, savedir = "./"):
    """
    pass on inputs to the gen_gals function and takes care of I/O

    Parameters
    ----------

    halos_data : dictionary of arrays 
        a dictionary of halo properties (pos, vel, mass, id, randoms, ...)

    particle_data : dictionary of arrays
        a dictionary of particle propoerties (pos, vel, hmass, hid, Np, subsampling, randoms, ...)

    tracers : dictionary of dictionaries
        Dictionary of multi-tracer HODs

    enable_ranks : boolean
        Flag of whether to implement particle ranks. 

    rsd : boolean
        Flag of whether to implement RSD. 

    write_to_disk : boolean
        Flag of whether to output to disk. 

    savedir : str
        where to save the output if write_to_disk == True. 

    params : dict
        Dictionary of various simulation parameters. 

    Output
    ------

    HOD_dict : dictionary of dictionaries
        Dictionary of the format: {tracer1_dict, tracer2_dict, ...}, 
        where tracer1_dict = {x, y, z, vx, vy, vz, mass, id}

    """

    if not type(rsd) is bool:
        raise ValueError("Error: rsd has to be a boolean")

    # find the halos, populate them with galaxies and write them to files
    HOD_dict = gen_gals(halo_data, particle_data, tracers, params, Nthread, enable_ranks, rsd)
    
    # how many galaxies were generated and write them to disk
    for tracer in tracers.keys():
        Ncent = HOD_dict[tracer]['Ncent']
        print("generated %ss:"%tracer, len(HOD_dict[tracer]['x']), 
            "satellite fraction ", 1 - Ncent/len(HOD_dict[tracer]['x']))
        
        if write_to_disk:
            print("outputting galaxies to disk")

            if rsd:
                rsd_string = "_rsd"
            else:
                rsd_string = ""

            outdir = (savedir) / ("galaxies"+rsd_string)

            # create directories if not existing
            os.makedirs(outdir, exist_ok = True)

            # save to file 
            outdict = HOD_dict[tracer].pop('Ncent', None)
            table = Table(HOD_dict[tracer], meta = {'Ncent': Ncent, 'Gal_type': tracer, **tracers[tracer]})
            ascii.write(table, outdir / ("%ss.dat"%tracer), overwrite = True, format = 'ecsv')

    return HOD_dict
