import math
import os
import time
import warnings

import numba
import numba as nb
import numpy as np
from astropy.io import ascii
from astropy.table import Table
from numba import njit, types
from numba.typed import Dict

# import yaml
# config = yaml.safe_load(open('config/abacus_hod.yaml'))
# numba.set_num_threads(16)
float_array = types.float64[:]
int_array = types.int64[:]
G = 4.302e-6  # in kpc/Msol (km.s)^2


@njit(fastmath=True)
def n_sat_LRG_modified(M_h, logM_cut, M_cut, M_1, sigma, alpha, kappa):
    """
    Standard Zheng et al. (2005) satellite HOD parametrization for LRGs, modified with n_cent_LRG
    """
    if M_h - kappa * M_cut < 0:
        return 0
    return (
        ((M_h - kappa * M_cut) / M_1) ** alpha
        * 0.5
        * math.erfc((logM_cut - np.log10(M_h)) / (1.41421356 * sigma))
    )


@njit(fastmath=True)
def n_cen_LRG(M_h, logM_cut, sigma):
    """
    Standard Zheng et al. (2005) central HOD parametrization for LRGs.
    """
    return 0.5 * math.erfc((logM_cut - np.log10(M_h)) / (1.41421356 * sigma))


@njit(fastmath=True)
def N_sat_generic(M_h, M_cut, kappa, M_1, alpha, A_s=1.0):
    """
    Standard Zheng et al. (2005) satellite HOD parametrization for all tracers with an optional amplitude parameter, A_s.
    """
    if M_h - kappa * M_cut < 0:
        return 0
    return A_s * ((M_h - kappa * M_cut) / M_1) ** alpha


@njit(fastmath=True)
def N_sat_elg(M_h, M_cut, kappa, M_1, alpha, A_s=1.0, alpha1=0.0, beta=0.0):
    """
    Standard power law modulated by an exponential fall off at small M
    """
    # return (M_h/M_1)**alpha/(1+np.exp(-A_s*(np.log10(M_h)-np.log10(kappa*M_cut)))) + beta*(M_h/M_1)**(-alpha1)/100
    if M_h - kappa * M_cut < 0:
        return 0
    return (
        A_s * ((M_h - kappa * M_cut) / M_1) ** alpha
    )  # + beta*(M_h/M_1)**(-alpha1)/100


@njit(fastmath=True)
def N_cen_ELG_v1(M_h, p_max, Q, logM_cut, sigma, gamma, Anorm=1):
    """
    HOD function for ELG centrals taken from arXiv:1910.05095.
    """
    logM_h = np.log10(M_h)
    phi = phi_fun(logM_h, logM_cut, sigma)
    Phi = Phi_fun(logM_h, logM_cut, sigma, gamma)
    return (
        2.0 * (p_max - 1.0 / Q) * phi * Phi / Anorm
    )  # + 0.5/Q*(1 + math.erf((logM_h-logM_cut-0.8)*3))


@njit(fastmath=True)
def N_cen_ELG_v2(M_h, p_max, logM_cut, sigma, gamma):
    """
    HOD function for ELG centrals taken from arXiv:2007.09012.
    """
    logM_h = np.log10(M_h)
    if logM_h <= logM_cut:
        return p_max * Gaussian_fun(logM_h, logM_cut, sigma)
    else:
        return p_max * (M_h / 10**logM_cut) ** gamma / (2.5066283 * sigma)


@njit(fastmath=True)
def N_cen_QSO(M_h, logM_cut, sigma):
    """
    HOD function (Zheng et al. (2005) with p_max) for QSO centrals taken from arXiv:2007.09012.
    """
    return 0.5 * (1 + math.erf((np.log10(M_h) - logM_cut) / 1.41421356 / sigma))


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
    x = gamma * (logM_h - logM_cut) / sigma
    Phi = 0.5 * (1 + math.erf(x / np.sqrt(2)))
    return Phi


@njit(fastmath=True)
def Gaussian_fun(x, mean, sigma):
    """
    Gaussian function with centered at `mean' with standard deviation `sigma'.
    """
    return 0.3989422804014327 / sigma * np.exp(-((x - mean) ** 2) / 2 / sigma**2)


@njit(fastmath=True)
def wrap(x, L):
    """Fast scalar mod implementation"""
    L2 = L / 2
    if x >= L2:
        return x - L
    elif x < -L2:
        return x + L
    return x


@njit(parallel=True, fastmath=True)
def gen_cent(
    pos,
    vel,
    mass,
    ids,
    multis,
    randoms,
    vdev,
    deltac,
    fenv,
    shear,
    LRG_hod_dict,
    ELG_hod_dict,
    QSO_hod_dict,
    rsd,
    inv_velz2kms,
    lbox,
    want_LRG,
    want_ELG,
    want_QSO,
    Nthread,
    origin,
):
    """
    Generate central galaxies in place in memory with a two pass numba parallel implementation.
    """

    if want_LRG:
        # parse out the hod parameters
        logM_cut_L, sigma_L = LRG_hod_dict['logM_cut'], LRG_hod_dict['sigma']
        ic_L, alpha_c_L, Ac_L, Bc_L = (
            LRG_hod_dict['ic'],
            LRG_hod_dict['alpha_c'],
            LRG_hod_dict['Acent'],
            LRG_hod_dict['Bcent'],
        )

    if want_ELG:
        pmax_E, Q_E, logM_cut_E, sigma_E, gamma_E = (
            ELG_hod_dict['p_max'],
            ELG_hod_dict['Q'],
            ELG_hod_dict['logM_cut'],
            ELG_hod_dict['sigma'],
            ELG_hod_dict['gamma'],
        )
        alpha_c_E, Ac_E, Bc_E, Cc_E, ic_E = (
            ELG_hod_dict['alpha_c'],
            ELG_hod_dict['Acent'],
            ELG_hod_dict['Bcent'],
            ELG_hod_dict['Ccent'],
            ELG_hod_dict['ic'],
        )

    if want_QSO:
        logM_cut_Q, sigma_Q = QSO_hod_dict['logM_cut'], QSO_hod_dict['sigma']
        alpha_c_Q, Ac_Q, Bc_Q, ic_Q = (
            QSO_hod_dict['alpha_c'],
            QSO_hod_dict['Acent'],
            QSO_hod_dict['Bcent'],
            QSO_hod_dict['ic'],
        )

    H = len(mass)

    numba.set_num_threads(Nthread)
    Nout = np.zeros((Nthread, 3, 8), dtype=np.int64)
    hstart = np.rint(np.linspace(0, H, Nthread + 1)).astype(
        np.int64
    )  # starting index of each thread

    keep = np.empty(H, dtype=np.int8)  # mask array tracking which halos to keep

    # figuring out the number of halos kept for each thread
    for tid in numba.prange(Nthread):
        for i in range(hstart[tid], hstart[tid + 1]):
            # first create the markers between 0 and 1 for different tracers
            LRG_marker = 0
            if want_LRG:
                # do assembly bias and secondary bias
                logM_cut_L_temp = logM_cut_L + Ac_L * deltac[i] + Bc_L * fenv[i]
                LRG_marker += (
                    n_cen_LRG(mass[i], logM_cut_L_temp, sigma_L) * ic_L * multis[i]
                )
            ELG_marker = LRG_marker
            if want_ELG:
                logM_cut_E_temp = (
                    logM_cut_E + Ac_E * deltac[i] + Bc_E * fenv[i] + Cc_E * shear[i]
                )
                ELG_marker += (
                    N_cen_ELG_v1(
                        mass[i], pmax_E, Q_E, logM_cut_E_temp, sigma_E, gamma_E
                    )
                    * ic_E
                    * multis[i]
                )
            QSO_marker = ELG_marker
            if want_QSO:
                logM_cut_Q_temp = logM_cut_Q + Ac_Q * deltac[i] + Bc_Q * fenv[i]
                QSO_marker += (
                    N_cen_QSO(mass[i], logM_cut_Q_temp, sigma_Q) * ic_Q * multis[i]
                )

            if randoms[i] <= LRG_marker:
                Nout[tid, 0, 0] += 1  # counting
                keep[i] = 1
            elif randoms[i] <= ELG_marker:
                Nout[tid, 1, 0] += 1  # counting
                keep[i] = 2
            elif randoms[i] <= QSO_marker:
                Nout[tid, 2, 0] += 1  # counting
                keep[i] = 3
            else:
                keep[i] = 0

    # compose galaxy array, first create array of galaxy starting indices for the threads
    gstart = np.empty((Nthread + 1, 3), dtype=np.int64)
    gstart[0, :] = 0
    gstart[1:, 0] = Nout[:, 0, 0].cumsum()
    gstart[1:, 1] = Nout[:, 1, 0].cumsum()
    gstart[1:, 2] = Nout[:, 2, 0].cumsum()

    # galaxy arrays
    N_lrg = gstart[-1, 0]
    lrg_x = np.empty(N_lrg, dtype=mass.dtype)
    lrg_y = np.empty(N_lrg, dtype=mass.dtype)
    lrg_z = np.empty(N_lrg, dtype=mass.dtype)
    lrg_vx = np.empty(N_lrg, dtype=mass.dtype)
    lrg_vy = np.empty(N_lrg, dtype=mass.dtype)
    lrg_vz = np.empty(N_lrg, dtype=mass.dtype)
    lrg_mass = np.empty(N_lrg, dtype=mass.dtype)
    lrg_id = np.empty(N_lrg, dtype=ids.dtype)

    # galaxy arrays
    N_elg = gstart[-1, 1]
    elg_x = np.empty(N_elg, dtype=mass.dtype)
    elg_y = np.empty(N_elg, dtype=mass.dtype)
    elg_z = np.empty(N_elg, dtype=mass.dtype)
    elg_vx = np.empty(N_elg, dtype=mass.dtype)
    elg_vy = np.empty(N_elg, dtype=mass.dtype)
    elg_vz = np.empty(N_elg, dtype=mass.dtype)
    elg_mass = np.empty(N_elg, dtype=mass.dtype)
    elg_id = np.empty(N_elg, dtype=ids.dtype)

    # galaxy arrays
    N_qso = gstart[-1, 2]
    qso_x = np.empty(N_qso, dtype=mass.dtype)
    qso_y = np.empty(N_qso, dtype=mass.dtype)
    qso_z = np.empty(N_qso, dtype=mass.dtype)
    qso_vx = np.empty(N_qso, dtype=mass.dtype)
    qso_vy = np.empty(N_qso, dtype=mass.dtype)
    qso_vz = np.empty(N_qso, dtype=mass.dtype)
    qso_mass = np.empty(N_qso, dtype=mass.dtype)
    qso_id = np.empty(N_qso, dtype=ids.dtype)

    # fill in the galaxy arrays
    for tid in numba.prange(Nthread):
        j1, j2, j3 = gstart[tid]
        for i in range(hstart[tid], hstart[tid + 1]):
            if keep[i] == 1:
                # loop thru three directions to assign galaxy velocities and positions
                lrg_x[j1] = pos[i, 0]
                lrg_vx[j1] = vel[i, 0] + alpha_c_L * vdev[i, 0]  # velocity bias
                lrg_y[j1] = pos[i, 1]
                lrg_vy[j1] = vel[i, 1] + alpha_c_L * vdev[i, 1]  # velocity bias
                lrg_z[j1] = pos[i, 2]
                lrg_vz[j1] = vel[i, 2] + alpha_c_L * vdev[i, 2]  # velocity bias
                # rsd only applies to the z direction
                if rsd and origin is not None:
                    nx = lrg_x[j1] - origin[0]
                    ny = lrg_y[j1] - origin[1]
                    nz = lrg_z[j1] - origin[2]
                    inv_norm = 1.0 / np.sqrt(nx * nx + ny * ny + nz * nz)
                    nx *= inv_norm
                    ny *= inv_norm
                    nz *= inv_norm
                    proj = inv_velz2kms * (
                        lrg_vx[j1] * nx + lrg_vy[j1] * ny + lrg_vz[j1] * nz
                    )
                    lrg_x[j1] = lrg_x[j1] + proj * nx
                    lrg_y[j1] = lrg_y[j1] + proj * ny
                    lrg_z[j1] = lrg_z[j1] + proj * nz
                elif rsd:
                    lrg_z[j1] = wrap(pos[i, 2] + lrg_vz[j1] * inv_velz2kms, lbox)
                lrg_mass[j1] = mass[i]
                lrg_id[j1] = ids[i]
                j1 += 1
            elif keep[i] == 2:
                # loop thru three directions to assign galaxy velocities and positions
                elg_x[j2] = pos[i, 0]
                elg_vx[j2] = vel[i, 0] + alpha_c_E * vdev[i, 0]  # velocity bias
                elg_y[j2] = pos[i, 1]
                elg_vy[j2] = vel[i, 1] + alpha_c_E * vdev[i, 1]  # velocity bias
                elg_z[j2] = pos[i, 2]
                elg_vz[j2] = vel[i, 2] + alpha_c_E * vdev[i, 2]  # velocity bias
                # rsd only applies to the z direction
                if rsd and origin is not None:
                    nx = elg_x[j2] - origin[0]
                    ny = elg_y[j2] - origin[1]
                    nz = elg_z[j2] - origin[2]
                    inv_norm = 1.0 / np.sqrt(nx * nx + ny * ny + nz * nz)
                    nx *= inv_norm
                    ny *= inv_norm
                    nz *= inv_norm
                    proj = inv_velz2kms * (
                        elg_vx[j2] * nx + elg_vy[j2] * ny + elg_vz[j2] * nz
                    )
                    elg_x[j2] = elg_x[j2] + proj * nx
                    elg_y[j2] = elg_y[j2] + proj * ny
                    elg_z[j2] = elg_z[j2] + proj * nz
                elif rsd:
                    elg_z[j2] = wrap(pos[i, 2] + elg_vz[j2] * inv_velz2kms, lbox)
                elg_mass[j2] = mass[i]
                elg_id[j2] = ids[i]
                j2 += 1
            elif keep[i] == 3:
                # loop thru three directions to assign galaxy velocities and positions
                qso_x[j3] = pos[i, 0]
                qso_vx[j3] = vel[i, 0] + alpha_c_Q * vdev[i, 0]  # velocity bias
                qso_y[j3] = pos[i, 1]
                qso_vy[j3] = vel[i, 1] + alpha_c_Q * vdev[i, 1]  # velocity bias
                qso_z[j3] = pos[i, 2]
                qso_vz[j3] = vel[i, 2] + alpha_c_Q * vdev[i, 2]  # velocity bias
                # rsd only applies to the z direction
                if rsd and origin is not None:
                    nx = qso_x[j3] - origin[0]
                    ny = qso_y[j3] - origin[1]
                    nz = qso_z[j3] - origin[2]
                    inv_norm = 1.0 / np.sqrt(nx * nx + ny * ny + nz * nz)
                    nx *= inv_norm
                    ny *= inv_norm
                    nz *= inv_norm
                    proj = inv_velz2kms * (
                        qso_vx[j3] * nx + qso_vy[j3] * ny + qso_vz[j3] * nz
                    )
                    qso_x[j3] = qso_x[j3] + proj * nx
                    qso_y[j3] = qso_y[j3] + proj * ny
                    qso_z[j3] = qso_z[j3] + proj * nz
                elif rsd:
                    qso_z[j3] = wrap(pos[i, 2] + qso_vz[j3] * inv_velz2kms, lbox)
                qso_mass[j3] = mass[i]
                qso_id[j3] = ids[i]
                j3 += 1
        # assert j == gstart[tid + 1]

    LRG_dict = Dict.empty(key_type=types.unicode_type, value_type=float_array)
    ELG_dict = Dict.empty(key_type=types.unicode_type, value_type=float_array)
    QSO_dict = Dict.empty(key_type=types.unicode_type, value_type=float_array)
    ID_dict = Dict.empty(key_type=types.unicode_type, value_type=int_array)
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
    return LRG_dict, ELG_dict, QSO_dict, ID_dict, keep


@njit(parallel=True, fastmath=True)
def getPointsOnSphere(nPoints, Nthread, seed=None):
    """
    --- Aiding function for NFW computation, generate random points in a sphere
    """
    numba.set_num_threads(Nthread)
    ind = min(Nthread, nPoints)
    # starting index of each thread
    hstart = np.rint(np.linspace(0, nPoints, ind + 1))
    ur = np.zeros((nPoints, 3), dtype=np.float64)
    cmin = -1
    cmax = +1

    for tid in numba.prange(Nthread):
        if seed is not None:
            np.random.seed(seed[tid])
        for i in range(hstart[tid], hstart[tid + 1]):
            u1, u2 = np.random.uniform(0, 1), np.random.uniform(0, 1)
            ra = 0 + u1 * (2 * np.pi - 0)
            dec = np.pi - (np.arccos(cmin + u2 * (cmax - cmin)))

            ur[i, 0] = np.sin(dec) * np.cos(ra)
            ur[i, 1] = np.sin(dec) * np.sin(ra)
            ur[i, 2] = np.cos(dec)
    return ur


@njit(fastmath=True, parallel=True)  # parallel=True,
def compute_fast_NFW(
    NFW_draw,
    h_id,
    x_h,
    y_h,
    z_h,
    vx_h,
    vy_h,
    vz_h,
    vrms_h,
    c,
    M,
    Rvir,
    rd_pos,
    num_sat,
    f_sigv,
    vel_sat='rd_normal',
    Nthread=16,
    exp_frac=0,
    exp_scale=1,
    nfw_rescale=1,
):
    """
    --- Compute NFW positions and velocities for satelitte galaxies
    c: r98/r25
    vrms_h: 'sigmav3d_L2com'
    """
    # numba.set_num_threads(Nthread)
    # figuring out the number of halos kept for each thread
    h_id = np.repeat(h_id, num_sat)
    M = np.repeat(M, num_sat)
    c = np.repeat(c, num_sat)
    Rvir = np.repeat(Rvir, num_sat)
    x_h = np.repeat(x_h, num_sat)
    y_h = np.repeat(y_h, num_sat)
    z_h = np.repeat(z_h, num_sat)
    vx_h = np.repeat(vx_h, num_sat)
    vy_h = np.repeat(vy_h, num_sat)
    vz_h = np.repeat(vz_h, num_sat)
    vrms_h = np.repeat(vrms_h, num_sat)
    x_sat = np.empty_like(x_h)
    y_sat = np.empty_like(y_h)
    z_sat = np.empty_like(z_h)
    vx_sat = np.empty_like(vx_h)
    vy_sat = np.empty_like(vy_h)
    vz_sat = np.empty_like(vz_h)

    # starting index of each thread
    hstart = np.rint(np.linspace(0, num_sat.sum(), Nthread + 1))
    for tid in numba.prange(Nthread):
        for i in range(int(hstart[tid]), int(hstart[tid + 1])):
            ind = i
            # while (NFW_draw[ind] > c[i]):
            #    ind = np.random.randint(0, len(NFW_draw))
            # etaVir = NFW_draw[ind]/c[i]  # =r/rvir
            if np.random.uniform(0, 1) < exp_frac:
                tt = np.random.exponential(exp_scale, size=1)[0]
                etaVir = tt / c[i]
            else:
                while NFW_draw[ind] > c[i]:
                    ind = np.random.randint(0, len(NFW_draw))
                etaVir = NFW_draw[ind] / c[i] * nfw_rescale

            p = etaVir * Rvir[i]
            x_sat[i] = x_h[i] + rd_pos[i, 0] * p
            y_sat[i] = y_h[i] + rd_pos[i, 1] * p
            z_sat[i] = z_h[i] + rd_pos[i, 2] * p
            if vel_sat == 'rd_normal':
                sig = vrms_h[i] * 0.577 * f_sigv
                vx_sat[i] = np.random.normal(loc=vx_h[i], scale=sig)
                vy_sat[i] = np.random.normal(loc=vy_h[i], scale=sig)
                vz_sat[i] = np.random.normal(loc=vz_h[i], scale=sig)
            else:
                raise ValueError('Wrong vel_sat argument only "rd_normal"')
    return h_id, x_sat, y_sat, z_sat, vx_sat, vy_sat, vz_sat, M


@njit(fastmath=True, parallel=True)
def gen_sats_nfw(
    NFW_draw,
    hpos,
    hvel,
    hmass,
    hid,
    hdeltac,
    hfenv,
    hshear,
    hvrms,
    hc,
    hrvir,
    LRG_hod_dict,
    ELG_hod_dict,
    QSO_hod_dict,
    want_LRG,
    want_ELG,
    want_QSO,
    rsd,
    inv_velz2kms,
    lbox,
    keep_cent,
    vel_sat='rd_normal',
    Nthread=16,
):
    """
    Generate satellite galaxies on an NFW profile, with option for an extended profile. See Rocher et al. 2023.

    Not yet on lightcone!! Different velocity bias treatment!! Not built for performance!!

    """

    if want_LRG:
        logM_cut_L, logM1_L, sigma_L, alpha_L, kappa_L = (
            LRG_hod_dict['logM_cut'],
            LRG_hod_dict['logM1'],
            LRG_hod_dict['sigma'],
            LRG_hod_dict['alpha'],
            LRG_hod_dict['kappa'],
        )
        Ac_L, As_L, Bc_L, Bs_L, ic_L = (
            LRG_hod_dict['Acent'],
            LRG_hod_dict['Asat'],
            LRG_hod_dict['Bcent'],
            LRG_hod_dict['Bsat'],
            LRG_hod_dict['ic'],
        )
        f_sigv_L = LRG_hod_dict['f_sigv']

    if want_ELG:
        logM_cut_E, kappa_E, logM1_E, alpha_E, A_E = (
            ELG_hod_dict['logM_cut'],
            ELG_hod_dict['kappa'],
            ELG_hod_dict['logM1'],
            ELG_hod_dict['alpha'],
            ELG_hod_dict['A_s'],
        )
        (
            Ac_E,
            As_E,
            Bc_E,
            Bs_E,
            Cc_E,
            Cs_E,
            ic_E,
        ) = (
            ELG_hod_dict['Acent'],
            ELG_hod_dict['Asat'],
            ELG_hod_dict['Bcent'],
            ELG_hod_dict['Bsat'],
            ELG_hod_dict['Ccent'],
            ELG_hod_dict['Csat'],
            ELG_hod_dict['ic'],
        )
        logM1_EE, alpha_EE, logM1_EL, alpha_EL = (
            ELG_hod_dict['logM1_EE'],
            ELG_hod_dict['alpha_EE'],
            ELG_hod_dict['logM1_EL'],
            ELG_hod_dict['alpha_EL'],
        )
        f_sigv_E = ELG_hod_dict['f_sigv']
        exp_frac = ELG_hod_dict['exp_frac']
        exp_scale = ELG_hod_dict['exp_scale']
        nfw_rescale = ELG_hod_dict['nfw_rescale']

    if want_QSO:
        logM_cut_Q, kappa_Q, logM1_Q, alpha_Q = (
            QSO_hod_dict['logM_cut'],
            QSO_hod_dict['kappa'],
            QSO_hod_dict['logM1'],
            QSO_hod_dict['alpha'],
        )
        Ac_Q, As_Q, Bc_Q, Bs_Q, ic_Q = (
            QSO_hod_dict['Acent'],
            QSO_hod_dict['Asat'],
            QSO_hod_dict['Bcent'],
            QSO_hod_dict['Bsat'],
            QSO_hod_dict['ic'],
        )
        f_sigv_Q = QSO_hod_dict['f_sigv']

    numba.set_num_threads(Nthread)

    # compute nsate for each halo
    # figuring out the number of particles kept for each thread
    num_sats_L = np.zeros(len(hid), dtype=np.int64)
    num_sats_E = np.zeros(len(hid), dtype=np.int64)
    num_sats_Q = np.zeros(len(hid), dtype=np.int64)
    hstart = np.rint(np.linspace(0, len(hid), Nthread + 1)).astype(
        np.int64
    )  # starting index of each thread
    for tid in range(Nthread):
        for i in range(hstart[tid], hstart[tid + 1]):
            if want_LRG:
                M1_L_temp = 10 ** (logM1_L + As_L * hdeltac[i] + Bs_L * hfenv[i])
                logM_cut_L_temp = logM_cut_L + Ac_L * hdeltac[i] + Bc_L * hfenv[i]
                base_p_L = (
                    n_sat_LRG_modified(
                        hmass[i],
                        logM_cut_L_temp,
                        10**logM_cut_L_temp,
                        M1_L_temp,
                        sigma_L,
                        alpha_L,
                        kappa_L,
                    )
                    * ic_L
                )
                num_sats_L[i] = np.random.poisson(base_p_L)
            if want_ELG:
                M1_E_temp = 10 ** (
                    logM1_E + As_E * hdeltac[i] + Bs_E * hfenv[i] + Cs_E * hshear[i]
                )
                logM_cut_E_temp = (
                    logM_cut_E + Ac_E * hdeltac[i] + Bc_E * hfenv[i] + Cc_E * hshear[i]
                )
                base_p_E = (
                    N_sat_elg(
                        hmass[i], 10**logM_cut_E_temp, kappa_E, M1_E_temp, alpha_E, A_E
                    )
                    * ic_E
                )
                # elg conformity
                if keep_cent[i] == 1:
                    M1_E_temp = 10 ** (logM1_EL + As_E * hdeltac[i] + Bs_E * hfenv[i])
                    base_p_E = (
                        N_sat_elg(
                            hmass[i],
                            10**logM_cut_E_temp,
                            kappa_E,
                            M1_E_temp,
                            alpha_EL,
                            A_E,
                        )
                        * ic_E
                    )
                elif keep_cent[i] == 2:
                    M1_E_temp = 10 ** (
                        logM1_EE + As_E * hdeltac[i] + Bs_E * hfenv[i]
                    )  # M1_E_temp*10**delta_M1
                    base_p_E = (
                        N_sat_elg(
                            hmass[i],
                            10**logM_cut_E_temp,
                            kappa_E,
                            M1_E_temp,
                            alpha_EE,
                            A_E,
                        )
                        * ic_E
                    )
                num_sats_E[i] = np.random.poisson(base_p_E)

            if want_QSO:
                M1_Q_temp = 10 ** (logM1_Q + As_Q * hdeltac[i] + Bs_Q * hfenv[i])
                logM_cut_Q_temp = logM_cut_Q + Ac_Q * hdeltac[i] + Bc_Q * hfenv[i]
                base_p_Q = (
                    N_sat_generic(
                        hmass[i], 10**logM_cut_Q_temp, kappa_Q, M1_Q_temp, alpha_Q
                    )
                    * ic_Q
                )
                num_sats_Q[i] = np.random.poisson(base_p_Q)

    # generate rdpos
    rd_pos_L = getPointsOnSphere(np.sum(num_sats_L), Nthread)
    rd_pos_E = getPointsOnSphere(np.sum(num_sats_E), Nthread)
    rd_pos_Q = getPointsOnSphere(np.sum(num_sats_Q), Nthread)

    # put satellites on NFW
    h_id_L, x_sat_L, y_sat_L, z_sat_L, vx_sat_L, vy_sat_L, vz_sat_L, M_L = (
        compute_fast_NFW(
            NFW_draw,
            hid,
            hpos[:, 0],
            hpos[:, 1],
            hpos[:, 2],
            hvel[:, 0],
            hvel[:, 1],
            hvel[:, 2],
            hvrms,
            hc,
            hmass,
            hrvir,
            rd_pos_L,
            num_sats_L,
            f_sigv_L,
            vel_sat,
            Nthread,
            exp_frac,
            exp_scale,
            nfw_rescale,
        )
    )
    h_id_E, x_sat_E, y_sat_E, z_sat_E, vx_sat_E, vy_sat_E, vz_sat_E, M_E = (
        compute_fast_NFW(
            NFW_draw,
            hid,
            hpos[:, 0],
            hpos[:, 1],
            hpos[:, 2],
            hvel[:, 0],
            hvel[:, 1],
            hvel[:, 2],
            hvrms,
            hc,
            hmass,
            hrvir,
            rd_pos_E,
            num_sats_E,
            f_sigv_E,
            vel_sat,
            Nthread,
            exp_frac,
            exp_scale,
            nfw_rescale,
        )
    )
    h_id_Q, x_sat_Q, y_sat_Q, z_sat_Q, vx_sat_Q, vy_sat_Q, vz_sat_Q, M_Q = (
        compute_fast_NFW(
            NFW_draw,
            hid,
            hpos[:, 0],
            hpos[:, 1],
            hpos[:, 2],
            hvel[:, 0],
            hvel[:, 1],
            hvel[:, 2],
            hvrms,
            hc,
            hmass,
            hrvir,
            rd_pos_Q,
            num_sats_Q,
            f_sigv_Q,
            vel_sat,
            Nthread,
            exp_frac,
            exp_scale,
            nfw_rescale,
        )
    )
    # do rsd
    if rsd:
        z_sat_L = (z_sat_L + vz_sat_L * inv_velz2kms) % lbox
        z_sat_E = (z_sat_E + vz_sat_E * inv_velz2kms) % lbox
        z_sat_Q = (z_sat_Q + vz_sat_Q * inv_velz2kms) % lbox

    LRG_dict = Dict.empty(key_type=types.unicode_type, value_type=float_array)
    ELG_dict = Dict.empty(key_type=types.unicode_type, value_type=float_array)
    QSO_dict = Dict.empty(key_type=types.unicode_type, value_type=float_array)
    ID_dict = Dict.empty(key_type=types.unicode_type, value_type=int_array)
    LRG_dict['x'] = x_sat_L
    LRG_dict['y'] = y_sat_L
    LRG_dict['z'] = z_sat_L
    LRG_dict['vx'] = vx_sat_L
    LRG_dict['vy'] = vy_sat_L
    LRG_dict['vz'] = vz_sat_L
    LRG_dict['mass'] = M_L
    ID_dict['LRG'] = h_id_L

    ELG_dict['x'] = x_sat_E
    ELG_dict['y'] = y_sat_E
    ELG_dict['z'] = z_sat_E
    ELG_dict['vx'] = vx_sat_E
    ELG_dict['vy'] = vy_sat_E
    ELG_dict['vz'] = vz_sat_E
    ELG_dict['mass'] = M_E
    ID_dict['ELG'] = h_id_E

    QSO_dict['x'] = x_sat_Q
    QSO_dict['y'] = y_sat_Q
    QSO_dict['z'] = z_sat_Q
    QSO_dict['vx'] = vx_sat_Q
    QSO_dict['vy'] = vy_sat_Q
    QSO_dict['vz'] = vz_sat_Q
    QSO_dict['mass'] = M_Q
    ID_dict['QSO'] = h_id_Q

    return LRG_dict, ELG_dict, QSO_dict, ID_dict


@njit(parallel=True, fastmath=True)
def gen_sats(
    ppos,
    pvel,
    hvel,
    hmass,
    hid,
    weights,
    randoms,
    hdeltac,
    hfenv,
    hshear,
    enable_ranks,
    ranks,
    ranksv,
    ranksp,
    ranksr,
    ranksc,
    LRG_hod_dict,
    ELG_hod_dict,
    QSO_hod_dict,
    rsd,
    inv_velz2kms,
    lbox,
    Mpart,
    want_LRG,
    want_ELG,
    want_QSO,
    Nthread,
    origin,
    keep_cent,
):
    """
    Generate satellite galaxies in place in memory with a two pass numba parallel implementation.
    """

    if want_LRG:
        logM_cut_L, logM1_L, sigma_L, alpha_L, kappa_L = (
            LRG_hod_dict['logM_cut'],
            LRG_hod_dict['logM1'],
            LRG_hod_dict['sigma'],
            LRG_hod_dict['alpha'],
            LRG_hod_dict['kappa'],
        )
        alpha_s_L, s_L, s_v_L, s_p_L, s_r_L, Ac_L, As_L, Bc_L, Bs_L, ic_L = (
            LRG_hod_dict['alpha_s'],
            LRG_hod_dict['s'],
            LRG_hod_dict['s_v'],
            LRG_hod_dict['s_p'],
            LRG_hod_dict['s_r'],
            LRG_hod_dict['Acent'],
            LRG_hod_dict['Asat'],
            LRG_hod_dict['Bcent'],
            LRG_hod_dict['Bsat'],
            LRG_hod_dict['ic'],
        )

    if want_ELG:
        logM_cut_E, kappa_E, logM1_E, alpha_E, A_E = (
            ELG_hod_dict['logM_cut'],
            ELG_hod_dict['kappa'],
            ELG_hod_dict['logM1'],
            ELG_hod_dict['alpha'],
            ELG_hod_dict['A_s'],
        )
        (
            alpha_s_E,
            s_E,
            s_v_E,
            s_p_E,
            s_r_E,
            Ac_E,
            As_E,
            Bc_E,
            Bs_E,
            Cc_E,
            Cs_E,
            ic_E,
            logM1_EE,
            alpha_EE,
            logM1_EL,
            alpha_EL,
        ) = (
            ELG_hod_dict['alpha_s'],
            ELG_hod_dict['s'],
            ELG_hod_dict['s_v'],
            ELG_hod_dict['s_p'],
            ELG_hod_dict['s_r'],
            ELG_hod_dict['Acent'],
            ELG_hod_dict['Asat'],
            ELG_hod_dict['Bcent'],
            ELG_hod_dict['Bsat'],
            ELG_hod_dict['Ccent'],
            ELG_hod_dict['Csat'],
            ELG_hod_dict['ic'],
            ELG_hod_dict['logM1_EE'],
            ELG_hod_dict['alpha_EE'],
            ELG_hod_dict['logM1_EL'],
            ELG_hod_dict['alpha_EL'],
        )

    if want_QSO:
        logM_cut_Q, kappa_Q, logM1_Q, alpha_Q = (
            QSO_hod_dict['logM_cut'],
            QSO_hod_dict['kappa'],
            QSO_hod_dict['logM1'],
            QSO_hod_dict['alpha'],
        )
        alpha_s_Q, s_Q, s_v_Q, s_p_Q, s_r_Q, Ac_Q, As_Q, Bc_Q, Bs_Q, ic_Q = (
            QSO_hod_dict['alpha_s'],
            QSO_hod_dict['s'],
            QSO_hod_dict['s_v'],
            QSO_hod_dict['s_p'],
            QSO_hod_dict['s_r'],
            QSO_hod_dict['Acent'],
            QSO_hod_dict['Asat'],
            QSO_hod_dict['Bcent'],
            QSO_hod_dict['Bsat'],
            QSO_hod_dict['ic'],
        )

    H = len(hmass)  # num of particles

    numba.set_num_threads(Nthread)
    Nout = np.zeros((Nthread, 3, 8), dtype=np.int64)
    hstart = np.rint(np.linspace(0, H, Nthread + 1)).astype(
        np.int64
    )  # starting index of each thread

    keep = np.empty(H, dtype=np.int8)  # mask array tracking which halos to keep

    # figuring out the number of particles kept for each thread
    for tid in numba.prange(Nthread):  # numba.prange(Nthread):
        for i in range(hstart[tid], hstart[tid + 1]):
            # print(logM1, As, hdeltac[i], Bs, hfenv[i])
            LRG_marker = 0
            if want_LRG:
                M1_L_temp = 10 ** (logM1_L + As_L * hdeltac[i] + Bs_L * hfenv[i])
                logM_cut_L_temp = logM_cut_L + Ac_L * hdeltac[i] + Bc_L * hfenv[i]
                base_p_L = (
                    n_sat_LRG_modified(
                        hmass[i],
                        logM_cut_L_temp,
                        10**logM_cut_L_temp,
                        M1_L_temp,
                        sigma_L,
                        alpha_L,
                        kappa_L,
                    )
                    * weights[i]
                    * ic_L
                )
                if enable_ranks:
                    decorator_L = (
                        1
                        + s_L * ranks[i]
                        + s_v_L * ranksv[i]
                        + s_p_L * ranksp[i]
                        + s_r_L * ranksr[i]
                    )
                    exp_sat = base_p_L * decorator_L
                else:
                    exp_sat = base_p_L
                LRG_marker += exp_sat

            ELG_marker = LRG_marker
            if want_ELG:
                M1_E_temp = 10 ** (
                    logM1_E + As_E * hdeltac[i] + Bs_E * hfenv[i] + Cs_E * hshear[i]
                )
                logM_cut_E_temp = (
                    logM_cut_E + Ac_E * hdeltac[i] + Bc_E * hfenv[i] + Cc_E * hshear[i]
                )
                base_p_E = (
                    N_sat_elg(
                        hmass[i], 10**logM_cut_E_temp, kappa_E, M1_E_temp, alpha_E, A_E
                    )
                    * weights[i]
                    * ic_E
                )
                # elg conformity
                if keep_cent[i] == 1:
                    M1_E_temp = 10 ** (logM1_EL + As_E * hdeltac[i] + Bs_E * hfenv[i])
                    base_p_E = (
                        N_sat_elg(
                            hmass[i],
                            10**logM_cut_E_temp,
                            kappa_E,
                            M1_E_temp,
                            alpha_EL,
                            A_E,
                        )
                        * weights[i]
                        * ic_E
                    )
                elif keep_cent[i] == 2:
                    M1_E_temp = 10 ** (
                        logM1_EE + As_E * hdeltac[i] + Bs_E * hfenv[i]
                    )  # M1_E_temp*10**delta_M1
                    base_p_E = (
                        N_sat_elg(
                            hmass[i],
                            10**logM_cut_E_temp,
                            kappa_E,
                            M1_E_temp,
                            alpha_EE,
                            A_E,
                        )
                        * weights[i]
                        * ic_E
                    )

                    # if base_p_E > 1:
                    #     print("ExE new p", base_p_E, np.log10(hmass[i]), N_sat_elg(
                    #     hmass[i], 10**logM_cut_E_temp, kappa_E, M1_E_temp, alpha_E_temp, A_E, alpha1, beta), weights[i], ic_E)

                # rank mods
                if enable_ranks:
                    decorator_E = (
                        1
                        + s_E * ranks[i]
                        + s_v_E * ranksv[i]
                        + s_p_E * ranksp[i]
                        + s_r_E * ranksr[i]
                    )
                    base_p_E = base_p_E * decorator_E

                ELG_marker += base_p_E

            QSO_marker = ELG_marker
            if want_QSO:
                M1_Q_temp = 10 ** (logM1_Q + As_Q * hdeltac[i] + Bs_Q * hfenv[i])
                logM_cut_Q_temp = logM_cut_Q + Ac_Q * hdeltac[i] + Bc_Q * hfenv[i]
                base_p_Q = (
                    N_sat_generic(
                        hmass[i], 10**logM_cut_Q_temp, kappa_Q, M1_Q_temp, alpha_Q
                    )
                    * weights[i]
                    * ic_Q
                )
                if enable_ranks:
                    decorator_Q = (
                        1
                        + s_Q * ranks[i]
                        + s_v_Q * ranksv[i]
                        + s_p_Q * ranksp[i]
                        + s_r_Q * ranksr[i]
                    )
                    exp_sat = base_p_Q * decorator_Q
                else:
                    exp_sat = base_p_Q
                QSO_marker += exp_sat

            if randoms[i] <= LRG_marker:
                Nout[tid, 0, 0] += 1  # counting
                keep[i] = 1
            elif randoms[i] <= ELG_marker:
                Nout[tid, 1, 0] += 1  # counting
                keep[i] = 2
            elif randoms[i] <= QSO_marker:
                Nout[tid, 2, 0] += 1  # counting
                keep[i] = 3
            else:
                keep[i] = 0

    # compose galaxy array, first create array of galaxy starting indices for the threads
    gstart = np.empty((Nthread + 1, 3), dtype=np.int64)
    gstart[0, :] = 0
    gstart[1:, 0] = Nout[:, 0, 0].cumsum()
    gstart[1:, 1] = Nout[:, 1, 0].cumsum()
    gstart[1:, 2] = Nout[:, 2, 0].cumsum()

    # galaxy arrays
    N_lrg = gstart[-1, 0]
    lrg_x = np.empty(N_lrg, dtype=hmass.dtype)
    lrg_y = np.empty(N_lrg, dtype=hmass.dtype)
    lrg_z = np.empty(N_lrg, dtype=hmass.dtype)
    lrg_vx = np.empty(N_lrg, dtype=hmass.dtype)
    lrg_vy = np.empty(N_lrg, dtype=hmass.dtype)
    lrg_vz = np.empty(N_lrg, dtype=hmass.dtype)
    lrg_mass = np.empty(N_lrg, dtype=hmass.dtype)
    lrg_id = np.empty(N_lrg, dtype=hid.dtype)

    # galaxy arrays
    N_elg = gstart[-1, 1]
    elg_x = np.empty(N_elg, dtype=hmass.dtype)
    elg_y = np.empty(N_elg, dtype=hmass.dtype)
    elg_z = np.empty(N_elg, dtype=hmass.dtype)
    elg_vx = np.empty(N_elg, dtype=hmass.dtype)
    elg_vy = np.empty(N_elg, dtype=hmass.dtype)
    elg_vz = np.empty(N_elg, dtype=hmass.dtype)
    elg_mass = np.empty(N_elg, dtype=hmass.dtype)
    elg_id = np.empty(N_elg, dtype=hid.dtype)

    # galaxy arrays
    N_qso = gstart[-1, 2]
    qso_x = np.empty(N_qso, dtype=hmass.dtype)
    qso_y = np.empty(N_qso, dtype=hmass.dtype)
    qso_z = np.empty(N_qso, dtype=hmass.dtype)
    qso_vx = np.empty(N_qso, dtype=hmass.dtype)
    qso_vy = np.empty(N_qso, dtype=hmass.dtype)
    qso_vz = np.empty(N_qso, dtype=hmass.dtype)
    qso_mass = np.empty(N_qso, dtype=hmass.dtype)
    qso_id = np.empty(N_qso, dtype=hid.dtype)

    # fill in the galaxy arrays
    for tid in numba.prange(Nthread):
        j1, j2, j3 = gstart[tid]
        for i in range(hstart[tid], hstart[tid + 1]):
            if keep[i] == 1:
                lrg_x[j1] = ppos[i, 0]
                lrg_vx[j1] = hvel[i, 0] + alpha_s_L * (
                    pvel[i, 0] - hvel[i, 0]
                )  # velocity bias
                lrg_y[j1] = ppos[i, 1]
                lrg_vy[j1] = hvel[i, 1] + alpha_s_L * (
                    pvel[i, 1] - hvel[i, 1]
                )  # velocity bias
                lrg_z[j1] = ppos[i, 2]
                lrg_vz[j1] = hvel[i, 2] + alpha_s_L * (
                    pvel[i, 2] - hvel[i, 2]
                )  # velocity bias
                if rsd and origin is not None:
                    nx = lrg_x[j1] - origin[0]
                    ny = lrg_y[j1] - origin[1]
                    nz = lrg_z[j1] - origin[2]
                    inv_norm = 1.0 / np.sqrt(nx * nx + ny * ny + nz * nz)
                    nx *= inv_norm
                    ny *= inv_norm
                    nz *= inv_norm
                    proj = inv_velz2kms * (
                        lrg_vx[j1] * nx + lrg_vy[j1] * ny + lrg_vz[j1] * nz
                    )
                    lrg_x[j1] = lrg_x[j1] + proj * nx
                    lrg_y[j1] = lrg_y[j1] + proj * ny
                    lrg_z[j1] = lrg_z[j1] + proj * nz
                elif rsd:
                    lrg_z[j1] = wrap(lrg_z[j1] + lrg_vz[j1] * inv_velz2kms, lbox)
                lrg_mass[j1] = hmass[i]
                lrg_id[j1] = hid[i]
                j1 += 1
            elif keep[i] == 2:
                elg_x[j2] = ppos[i, 0]
                elg_vx[j2] = hvel[i, 0] + alpha_s_E * (
                    pvel[i, 0] - hvel[i, 0]
                )  # velocity bias
                elg_y[j2] = ppos[i, 1]
                elg_vy[j2] = hvel[i, 1] + alpha_s_E * (
                    pvel[i, 1] - hvel[i, 1]
                )  # velocity bias
                elg_z[j2] = ppos[i, 2]
                elg_vz[j2] = hvel[i, 2] + alpha_s_E * (
                    pvel[i, 2] - hvel[i, 2]
                )  # velocity bias
                if rsd and origin is not None:
                    nx = elg_x[j2] - origin[0]
                    ny = elg_y[j2] - origin[1]
                    nz = elg_z[j2] - origin[2]
                    inv_norm = 1.0 / np.sqrt(nx * nx + ny * ny + nz * nz)
                    nx *= inv_norm
                    ny *= inv_norm
                    nz *= inv_norm
                    proj = inv_velz2kms * (
                        elg_vx[j2] * nx + elg_vy[j2] * ny + elg_vz[j2] * nz
                    )
                    elg_x[j2] = elg_x[j2] + proj * nx
                    elg_y[j2] = elg_y[j2] + proj * ny
                    elg_z[j2] = elg_z[j2] + proj * nz
                elif rsd:
                    elg_z[j2] = wrap(elg_z[j2] + elg_vz[j2] * inv_velz2kms, lbox)
                elg_mass[j2] = hmass[i]
                elg_id[j2] = hid[i]
                j2 += 1
            elif keep[i] == 3:
                qso_x[j3] = ppos[i, 0]
                qso_vx[j3] = hvel[i, 0] + alpha_s_Q * (
                    pvel[i, 0] - hvel[i, 0]
                )  # velocity bias
                qso_y[j3] = ppos[i, 1]
                qso_vy[j3] = hvel[i, 1] + alpha_s_Q * (
                    pvel[i, 1] - hvel[i, 1]
                )  # velocity bias
                qso_z[j3] = ppos[i, 2]
                qso_vz[j3] = hvel[i, 2] + alpha_s_Q * (
                    pvel[i, 2] - hvel[i, 2]
                )  # velocity bias
                if rsd and origin is not None:
                    nx = qso_x[j3] - origin[0]
                    ny = qso_y[j3] - origin[1]
                    nz = qso_z[j3] - origin[2]
                    inv_norm = 1.0 / np.sqrt(nx * nx + ny * ny + nz * nz)
                    nx *= inv_norm
                    ny *= inv_norm
                    nz *= inv_norm
                    proj = inv_velz2kms * (
                        qso_vx[j3] * nx + qso_vy[j3] * ny + qso_vz[j3] * nz
                    )
                    qso_x[j3] = qso_x[j3] + proj * nx
                    qso_y[j3] = qso_y[j3] + proj * ny
                    qso_z[j3] = qso_z[j3] + proj * nz
                elif rsd:
                    qso_z[j3] = wrap(qso_z[j3] + qso_vz[j3] * inv_velz2kms, lbox)
                qso_mass[j3] = hmass[i]
                qso_id[j3] = hid[i]
                j3 += 1
        # assert j == gstart[tid + 1]

    LRG_dict = Dict.empty(key_type=types.unicode_type, value_type=float_array)
    ELG_dict = Dict.empty(key_type=types.unicode_type, value_type=float_array)
    QSO_dict = Dict.empty(key_type=types.unicode_type, value_type=float_array)
    ID_dict = Dict.empty(key_type=types.unicode_type, value_type=int_array)
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


@njit(parallel=True, fastmath=True)
def fast_concatenate(array1, array2, Nthread):
    """Fast concatenate with numba parallel"""

    N1 = len(array1)
    N2 = len(array2)
    if N1 == 0:
        return array2
    elif N2 == 0:
        return array1

    final_array = np.empty(N1 + N2, dtype=array1.dtype)
    # if one thread, then no need to parallel
    if Nthread == 1:
        for i in range(N1):
            final_array[i] = array1[i]
        for j in range(N2):
            final_array[j + N1] = array2[j]
        return final_array

    numba.set_num_threads(Nthread)
    Nthread1 = max(1, int(np.floor(Nthread * N1 / (N1 + N2))))
    Nthread2 = Nthread - Nthread1
    hstart1 = np.rint(np.linspace(0, N1, Nthread1 + 1)).astype(np.int64)
    hstart2 = np.rint(np.linspace(0, N2, Nthread2 + 1)).astype(np.int64) + N1

    for tid in numba.prange(Nthread):  # numba.prange(Nthread):
        if tid < Nthread1:
            for i in range(hstart1[tid], hstart1[tid + 1]):
                final_array[i] = array1[i]
        else:
            for i in range(hstart2[tid - Nthread1], hstart2[tid + 1 - Nthread1]):
                final_array[i] = array2[i - N1]
    # final_array = np.concatenate((array1, array2))
    return final_array


def gen_gals(
    halos_array,
    subsample,
    tracers,
    params,
    Nthread,
    enable_ranks,
    rsd,
    verbose,
    nfw,
    NFW_draw=None,
):
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

        LRG_hod_dict = nb.typed.Dict.empty(
            key_type=nb.types.unicode_type, value_type=nb.types.float64
        )
        for key, value in LRG_HOD.items():
            LRG_hod_dict[key] = value

        # LRG design and decorations
        logM_cut_L, logM1_L = map(LRG_HOD.get, ('logM_cut', 'logM1'))
        # z-evolving HOD
        Delta_a = 1.0 / (1 + params['z']) - 1.0 / (
            1 + LRG_HOD.get('z_pivot', params['z'])
        )
        logM_cut_pr = LRG_HOD.get('logM_cut_pr', 0.0)
        logM1_pr = LRG_HOD.get('logM1_pr', 0.0)
        logM_cut_L = logM_cut_L + logM_cut_pr * Delta_a
        logM1_L = logM1_L + logM1_pr * Delta_a

        LRG_hod_dict['logM_cut'] = logM_cut_L
        LRG_hod_dict['logM1'] = logM1_L

        LRG_hod_dict['Acent'] = LRG_HOD.get('Acent', 0.0)
        LRG_hod_dict['Asat'] = LRG_HOD.get('Asat', 0.0)
        LRG_hod_dict['Bcent'] = LRG_HOD.get('Bcent', 0.0)
        LRG_hod_dict['Bsat'] = LRG_HOD.get('Bsat', 0.0)
        LRG_hod_dict['ic'] = LRG_HOD.get('ic', 1.0)

        LRG_hod_dict['f_sigv'] = LRG_HOD.get('f_sigv', 0)

    else:
        want_LRG = False
        LRG_hod_dict = nb.typed.Dict.empty(
            key_type=nb.types.unicode_type, value_type=nb.types.float64
        )

    if 'ELG' in tracers.keys():
        # ELG design
        want_ELG = True

        ELG_hod_dict = nb.typed.Dict.empty(
            key_type=nb.types.unicode_type, value_type=nb.types.float64
        )
        for key, value in ELG_HOD.items():
            ELG_hod_dict[key] = value

        logM_cut_E, logM1_E = map(ELG_HOD.get, ('logM_cut', 'logM1'))

        # z-evolving HOD
        Delta_a = 1.0 / (1 + params['z']) - 1.0 / (
            1 + ELG_HOD.get('z_pivot', params['z'])
        )
        logM_cut_pr = ELG_HOD.get('logM_cut_pr', 0.0)
        logM1_pr = ELG_HOD.get('logM1_pr', 0.0)
        logM_cut_E = logM_cut_E + logM_cut_pr * Delta_a
        logM1_E = logM1_E + logM1_pr * Delta_a

        ELG_hod_dict['logM_cut'] = logM_cut_E
        ELG_hod_dict['logM1'] = logM1_E

        ELG_hod_dict['Acent'] = ELG_HOD.get('Acent', 0.0)
        ELG_hod_dict['Asat'] = ELG_HOD.get('Asat', 0.0)
        ELG_hod_dict['Bcent'] = ELG_HOD.get('Bcent', 0.0)
        ELG_hod_dict['Bsat'] = ELG_HOD.get('Bsat', 0.0)
        ELG_hod_dict['Ccent'] = ELG_HOD.get('Ccent', 0.0)
        ELG_hod_dict['Csat'] = ELG_HOD.get('Csat', 0.0)
        ELG_hod_dict['ic'] = ELG_HOD.get('ic', 1.0)
        ELG_hod_dict['logM1_EE'] = ELG_HOD.get('logM1_EE', ELG_hod_dict['logM1'])
        ELG_hod_dict['alpha_EE'] = ELG_HOD.get('alpha_EE', ELG_hod_dict['alpha'])
        ELG_hod_dict['logM1_EL'] = ELG_HOD.get('logM1_EL', ELG_hod_dict['logM1'])
        ELG_hod_dict['alpha_EL'] = ELG_HOD.get('alpha_EL', ELG_hod_dict['alpha'])

        ELG_hod_dict['f_sigv'] = ELG_HOD.get('f_sigv', 0)
        ELG_hod_dict['exp_frac'] = ELG_HOD.get('exp_frac', 0)
        ELG_hod_dict['exp_scale'] = ELG_HOD.get('exp_scale', 1)
        ELG_hod_dict['nfw_rescale'] = ELG_HOD.get('nfw_rescale', 1)

    else:
        want_ELG = False
        ELG_hod_dict = nb.typed.Dict.empty(
            key_type=nb.types.unicode_type, value_type=nb.types.float64
        )

    if 'QSO' in tracers.keys():
        # QSO design
        want_QSO = True
        QSO_hod_dict = nb.typed.Dict.empty(
            key_type=nb.types.unicode_type, value_type=nb.types.float64
        )
        for key, value in QSO_HOD.items():
            QSO_hod_dict[key] = value

        logM_cut_Q, logM1_Q = map(QSO_HOD.get, ('logM_cut', 'logM1'))
        # z-evolving HOD
        Delta_a = 1.0 / (1 + params['z']) - 1.0 / (
            1 + QSO_HOD.get('z_pivot', params['z'])
        )
        logM_cut_pr = QSO_HOD.get('logM_cut_pr', 0.0)
        logM1_pr = QSO_HOD.get('logM1_pr', 0.0)
        logM_cut_Q = logM_cut_Q + logM_cut_pr * Delta_a
        logM1_Q = logM1_Q + logM1_pr * Delta_a

        QSO_hod_dict['logM_cut'] = logM_cut_Q
        QSO_hod_dict['logM1'] = logM1_Q

        QSO_hod_dict['Acent'] = QSO_HOD.get('Acent', 0.0)
        QSO_hod_dict['Asat'] = QSO_HOD.get('Asat', 0.0)
        QSO_hod_dict['Bcent'] = QSO_HOD.get('Bcent', 0.0)
        QSO_hod_dict['Bsat'] = QSO_HOD.get('Bsat', 0.0)
        QSO_hod_dict['ic'] = QSO_HOD.get('ic', 1.0)

        QSO_hod_dict['f_sigv'] = QSO_HOD.get('f_sigv', 0)

    else:
        want_QSO = False
        QSO_hod_dict = nb.typed.Dict.empty(
            key_type=nb.types.unicode_type, value_type=nb.types.float64
        )

    start = time.time()

    velz2kms = params['velz2kms']
    inv_velz2kms = 1 / velz2kms
    lbox = params['Lbox']
    origin = params['origin']

    LRG_dict_cent, ELG_dict_cent, QSO_dict_cent, ID_dict_cent, keep_cent = gen_cent(
        halos_array['hpos'],
        halos_array['hvel'],
        halos_array['hmass'],
        halos_array['hid'],
        halos_array['hmultis'],
        halos_array['hrandoms'],
        halos_array['hveldev'],
        halos_array.get('hdeltac', np.zeros(len(halos_array['hmass']))),
        halos_array.get('hfenv', np.zeros(len(halos_array['hmass']))),
        halos_array.get('hshear', np.zeros(len(halos_array['hmass']))),
        LRG_hod_dict,
        ELG_hod_dict,
        QSO_hod_dict,
        rsd,
        inv_velz2kms,
        lbox,
        want_LRG,
        want_ELG,
        want_QSO,
        Nthread,
        origin,
    )
    if verbose:
        print('generating centrals took ', time.time() - start)

    start = time.time()
    if nfw:
        warnings.warn(
            'NFW profile is unoptimized. It has different velocity bias. It does not support lightcone.'
        )
        LRG_dict_sat, ELG_dict_sat, QSO_dict_sat, ID_dict_sat = gen_sats_nfw(
            NFW_draw,
            halos_array['hpos'],
            halos_array['hvel'],
            halos_array['hmass'],
            halos_array['hid'],
            halos_array.get('hdeltac', np.zeros(len(halos_array['hmass']))),
            halos_array.get('hfenv', np.zeros(len(halos_array['hmass']))),
            halos_array.get('hshear', np.zeros(len(halos_array['hmass']))),
            halos_array['hsigma3d'],
            halos_array['hc'],
            halos_array['hrvir'],
            LRG_hod_dict,
            ELG_hod_dict,
            QSO_hod_dict,
            want_LRG,
            want_ELG,
            want_QSO,
            rsd,
            inv_velz2kms,
            lbox,
            keep_cent,
            Nthread=Nthread,
        )
    else:
        LRG_dict_sat, ELG_dict_sat, QSO_dict_sat, ID_dict_sat = gen_sats(
            subsample['ppos'],
            subsample['pvel'],
            subsample['phvel'],
            subsample['phmass'],
            subsample['phid'],
            subsample['pweights'],
            subsample['prandoms'],
            subsample.get('pdeltac', np.zeros(len(subsample['phid']))),
            subsample.get('pfenv', np.zeros(len(subsample['phid']))),
            subsample.get('pshear', np.zeros(len(subsample['phid']))),
            enable_ranks,
            subsample['pranks'],
            subsample['pranksv'],
            subsample['pranksp'],
            subsample['pranksr'],
            subsample['pranksc'],
            LRG_hod_dict,
            ELG_hod_dict,
            QSO_hod_dict,
            rsd,
            inv_velz2kms,
            lbox,
            params['Mpart'],
            want_LRG,
            want_ELG,
            want_QSO,
            Nthread,
            origin,
            keep_cent[subsample['pinds']],
        )
    if verbose:
        print('generating satellites took ', time.time() - start)

    # B.H. TODO: need a for loop above so we don't need to do this by hand
    HOD_dict_sat = {'LRG': LRG_dict_sat, 'ELG': ELG_dict_sat, 'QSO': QSO_dict_sat}
    HOD_dict_cent = {'LRG': LRG_dict_cent, 'ELG': ELG_dict_cent, 'QSO': QSO_dict_cent}

    # do a concatenate in numba parallel
    start = time.time()
    HOD_dict = {}
    for tracer in tracers:
        tracer_dict = {'Ncent': len(HOD_dict_cent[tracer]['x'])}
        for k in HOD_dict_cent[tracer]:
            tracer_dict[k] = fast_concatenate(
                HOD_dict_cent[tracer][k], HOD_dict_sat[tracer][k], Nthread
            )
        tracer_dict['id'] = fast_concatenate(
            ID_dict_cent[tracer], ID_dict_sat[tracer], Nthread
        )
        if verbose:
            print(tracer, 'number of galaxies ', len(tracer_dict['x']))
            print(
                'satellite fraction ',
                len(HOD_dict_sat[tracer]['x']) / len(tracer_dict['x']),
            )
        HOD_dict[tracer] = tracer_dict
    if verbose:
        print('organizing outputs took ', time.time() - start)
    return HOD_dict


def gen_gal_cat(
    halo_data,
    particle_data,
    tracers,
    params,
    Nthread=16,
    enable_ranks=False,
    rsd=True,
    nfw=False,
    NFW_draw=None,
    write_to_disk=False,
    savedir='./',
    verbose=False,
    fn_ext=None,
):
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

    nfw : boolean
        Flag of whether to generate satellites from an NFW profile.

    write_to_disk : boolean
        Flag of whether to output to disk.

    verbose : boolean
        Whether to output detailed outputs.

    savedir : str
        where to save the output if write_to_disk == True.

    params : dict
        Dictionary of various simulation parameters.

    fn_ext: str
        filename extension for saved files. Only relevant when ``write_to_disk = True``.

    Output
    ------

    HOD_dict : dictionary of dictionaries
        Dictionary of the format: {tracer1_dict, tracer2_dict, ...},
        where tracer1_dict = {x, y, z, vx, vy, vz, mass, id}

    """

    if not isinstance(rsd, bool):
        raise ValueError('Error: rsd has to be a boolean')

    # find the halos, populate them with galaxies and write them to files
    HOD_dict = gen_gals(
        halo_data,
        particle_data,
        tracers,
        params,
        Nthread,
        enable_ranks,
        rsd,
        verbose,
        nfw,
        NFW_draw,
    )

    # how many galaxies were generated and write them to disk
    for tracer in tracers.keys():
        Ncent = HOD_dict[tracer]['Ncent']
        if verbose:
            print(
                'generated %ss:' % tracer,
                len(HOD_dict[tracer]['x']),
                'satellite fraction ',
                1 - Ncent / len(HOD_dict[tracer]['x']),
            )

        if write_to_disk:
            if verbose:
                print('outputting galaxies to disk')

            if rsd:
                rsd_string = '_rsd'
            else:
                rsd_string = ''

            if fn_ext is None:
                outdir = (savedir) / ('galaxies' + rsd_string)
            else:
                outdir = (savedir) / ('galaxies' + rsd_string + fn_ext)

            # create directories if not existing
            os.makedirs(outdir, exist_ok=True)

            # save to file
            # outdict =
            HOD_dict[tracer].pop('Ncent', None)
            table = Table(
                HOD_dict[tracer],
                meta={'Ncent': Ncent, 'Gal_type': tracer, **tracers[tracer]},
            )
            if params['chunk'] == -1:
                ascii.write(
                    table, outdir / (f'{tracer}s.dat'), overwrite=True, format='ecsv'
                )
            else:
                ascii.write(
                    table,
                    outdir / (f'{tracer}s_chunk{params["chunk"]:d}.dat'),
                    overwrite=True,
                    format='ecsv',
                )

    return HOD_dict
