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
def n_cen_CSMF(M_h, Mstar_low, Mstar_up, M_1, M_0, gamma1, gamma2, sigma_c):
    """
    Standard Cacciato et al. (2008) centrals HOD parametrization for CSMF
    """
    M_c_value = M_c(M_h, M_1, M_0, gamma1, gamma2)
    x_low = np.log10(Mstar_low / M_c_value) / (1.41421356 * sigma_c)
    x_up = np.log10(Mstar_up / M_c_value) / (1.41421356 * sigma_c)

    return 0.5 * (math.erf(x_up) - math.erf(x_low))


@njit(fastmath=True)
def CSMF_centrals(M_h, Mstar, M_1, M_0, gamma1, gamma2, sigma_c):
    """
    Eq. (34) from Cacciato et al. (2008)
    """

    M_c_value = M_c(M_h, M_1, M_0, gamma1, gamma2)

    return (
        1
        / (1.41421356 * np.sqrt(np.pi) * np.log(10) * sigma_c * Mstar)
        * np.exp(-((np.log10(Mstar) - np.log10(M_c_value)) ** 2) / (2 * sigma_c**2))
    )


@njit(fastmath=True)
def M_c(M_h, M_1, M_0, gamma1, gamma2):
    """
    Eq. (37) from Cacciato et al. (2008)
    """
    return M_0 * (M_h / M_1) ** gamma1 / (1 + M_h / M_1) ** (gamma1 - gamma2)


@njit(fastmath=True)
def get_random_cen_stellarmass(
    M_h, Mstar_low, Mstar_up, M_1, M_0, gamma1, gamma2, sigma_c
):
    nbins = 1000
    stellarmass = np.logspace(np.log10(Mstar_low), np.log10(Mstar_up), nbins)
    stellarmass_centers = stellarmass[1:] / 2 + stellarmass[:-1] / 2
    delta_stellar_masses = stellarmass[1:] - stellarmass[:-1]

    CSMF_cen = CSMF_centrals(
        M_h, stellarmass_centers, M_1, M_0, gamma1, gamma2, sigma_c
    )

    cdf = np.cumsum(CSMF_cen * delta_stellar_masses)
    cdf = cdf / cdf[-1]

    random_rv = np.random.uniform(cdf.min(), cdf.max())
    bin_clostest = (np.abs(cdf - random_rv)).argmin()

    return np.random.uniform(stellarmass[bin_clostest], stellarmass[bin_clostest + 1])


@njit(fastmath=True)
def get_random_cen_stellarmass_linearinterpolation(
    M_h, Mstar_low, Mstar_up, M_1, M_0, gamma1, gamma2, sigma_c
):
    nbins = 1000
    stellarmass = np.logspace(np.log10(Mstar_low), np.log10(Mstar_up), nbins)
    stellarmass_centers = stellarmass[1:] / 2 + stellarmass[:-1] / 2
    delta_stellar_masses = stellarmass[1:] - stellarmass[:-1]

    CSMF_cen = CSMF_centrals(M_h, stellarmass, M_1, M_0, gamma1, gamma2, sigma_c)

    cdf = np.cumsum(CSMF_cen[:-1] * delta_stellar_masses)
    cdf = cdf / cdf[-1]

    random_rv = np.random.uniform(cdf.min(), cdf.max())
    bin = np.where(cdf > random_rv)[0][0]

    m = (stellarmass[bin] - stellarmass[bin - 1]) / (cdf[bin] - cdf[bin - 1])
    return m * (random_rv - cdf[bin - 1]) + stellarmass[bin - 1]


@njit(fastmath=True)
def n_sat_CSMF(
    M_h,
    Mstar_low,
    Mstar_up,
    M_1,
    M_0,
    gamma1,
    gamma2,
    sigma_c,
    a1,
    a2,
    M2,
    b0,
    b1,
    b2,
    delta1,
    delta2,
):
    """
    Standard Cacciato et al. (2008) satellite HOD parametrization for CSMF
    """
    nbins = 1000
    stellarmass = np.logspace(np.log10(Mstar_low), np.log10(Mstar_up), nbins)

    CSMF_sat = CSMF_satelites(
        M_h,
        stellarmass,
        M_1,
        M_0,
        gamma1,
        gamma2,
        a1,
        a2,
        M2,
        b0,
        b1,
        b2,
        delta1,
        delta2,
    )

    nsat = 0
    for i in range(nbins - 1):
        nsat += (CSMF_sat[i + 1] - CSMF_sat[i]) * (
            stellarmass[i + 1] - stellarmass[i]
        ) / 2 + (stellarmass[i + 1] - stellarmass[i]) * CSMF_sat[i]

    return nsat  # *ncen


@njit(fastmath=True)
def CSMF_satelites(
    M_h, Mstar, M_1, M_0, gamma1, gamma2, a1, a2, M2, b0, b1, b2, delta1, delta2
):
    """
    Eq. (36) from Cacciato et al. (2008)
    """
    M_s_value = M_s(M_h, M_1, M_0, gamma1, gamma2)
    alpha_s_value = alpha_s(M_h, a1, a2, M2)
    phi_s_value = phi_s(M_h, b0, b1, b2)

    delta = 10 ** (delta1 + delta2 * (np.log10(M_h) - 12))

    return (
        phi_s_value
        / M_s_value
        * (Mstar / M_s_value) ** alpha_s_value
        * np.exp(-delta * (Mstar / M_s_value) ** 2)
    )


@njit(fastmath=True)
def M_s(M_h, M_1, M_0, gamma1, gamma2):
    """
    Eq. (38) from Cacciato et al. (2008)
    """
    return 0.562 * M_c(M_h, M_1, M_0, gamma1, gamma2)


@njit(fastmath=True)
def alpha_s(M_h, a1, a2, M2):
    """
    Eq. (39) from Cacciato et al. (2008)
    """
    return -2.0 + a1 * (1 - 2 / np.pi * np.arctan(a2 * np.log10(M_h / M2)))


@njit(fastmath=True)
def phi_s(M_h, b0, b1, b2):
    """
    Eq. (40) from Cacciato et al. (2008)
    """
    M12 = M_h / 1e12
    log_phi_s = b0 + b1 * np.log10(M12) + b2 * np.log10(M12) ** 2
    return 10**log_phi_s


@njit(fastmath=True)
def get_random_sat_stellarmass(
    M_h,
    Mstar_low,
    Mstar_up,
    M_1,
    M_0,
    gamma1,
    gamma2,
    a1,
    a2,
    M2,
    b0,
    b1,
    b2,
    delta1,
    delta2,
):
    nbins = 1000
    stellarmass = np.logspace(np.log10(Mstar_low), np.log10(Mstar_up), nbins)
    stellarmass_centers = stellarmass[1:] / 2 + stellarmass[:-1] / 2
    delta_stellar_masses = stellarmass[1:] - stellarmass[:-1]

    CSMF_sat = CSMF_satelites(
        M_h,
        stellarmass_centers,
        M_1,
        M_0,
        gamma1,
        gamma2,
        a1,
        a2,
        M2,
        b0,
        b1,
        b2,
        delta1,
        delta2,
    )

    cdf = np.cumsum(CSMF_sat * delta_stellar_masses)
    cdf = cdf / cdf[-1]

    random_rv = np.random.uniform(cdf.min(), cdf.max())
    bin_clostest = (np.abs(cdf - random_rv)).argmin()

    return np.random.uniform(stellarmass[bin_clostest], stellarmass[bin_clostest + 1])


@njit(fastmath=True)
def get_random_sat_stellarmass_linearinterpolation(
    M_h,
    Mstar_low,
    Mstar_up,
    M_1,
    M_0,
    gamma1,
    gamma2,
    a1,
    a2,
    M2,
    b0,
    b1,
    b2,
    delta1,
    delta2,
):
    nbins = 100
    stellarmass = np.logspace(np.log10(Mstar_low), np.log10(Mstar_up), nbins)
    stellarmass_centers = stellarmass[1:] / 2 + stellarmass[:-1] / 2
    delta_stellar_masses = stellarmass[1:] - stellarmass[:-1]

    CSMF_sat = CSMF_satelites(
        M_h,
        stellarmass,
        M_1,
        M_0,
        gamma1,
        gamma2,
        a1,
        a2,
        M2,
        b0,
        b1,
        b2,
        delta1,
        delta2,
    )

    cdf = np.cumsum(CSMF_sat[:-1] * delta_stellar_masses)
    cdf = cdf / cdf[-1]

    random_rv = np.random.uniform(cdf.min(), cdf.max())
    bin = np.where(cdf > random_rv)[0][0]

    m = (stellarmass[bin] - stellarmass[bin - 1]) / (cdf[bin] - cdf[bin - 1])
    return m * (random_rv - cdf[bin - 1]) + stellarmass[bin - 1]


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
    CSMF_hod_dict,
    rsd,
    inv_velz2kms,
    lbox,
    want_CSMF,
    Nthread,
    origin,
):
    """
    Generate central galaxies in place in memory with a two pass numba parallel implementation.
    """
    if want_CSMF:
        Mstar_low_C, Mstar_up_C, M_1_C, M_0_C, gamma1_C, gamma2_C, sigma_c_C = (
            CSMF_hod_dict['Mstar_low'],
            CSMF_hod_dict['Mstar_up'],
            CSMF_hod_dict['M_1'],
            CSMF_hod_dict['M_0'],
            CSMF_hod_dict['gamma_1'],
            CSMF_hod_dict['gamma_2'],
            CSMF_hod_dict['sigma_c'],
        )
        ic_C, alpha_c_C, Ac_C, Bc_C = (
            CSMF_hod_dict['ic'],
            CSMF_hod_dict['alpha_c'],
            CSMF_hod_dict['Acent'],
            CSMF_hod_dict['Bcent'],
        )

    H = len(mass)

    numba.set_num_threads(Nthread)
    Nout = np.zeros((Nthread, 1, 8), dtype=np.int64)
    hstart = np.rint(np.linspace(0, H, Nthread + 1)).astype(
        np.int64
    )  # starting index of each thread

    keep = np.empty(H, dtype=np.int8)  # mask array tracking which halos to keep

    # figuring out the number of halos kept for each thread
    for tid in numba.prange(Nthread):
        for i in range(hstart[tid], hstart[tid + 1]):
            # first create the markers between 0 and 1 for different tracers
            CSMF_marker = 0
            if want_CSMF:
                M_1_C_temp = 10 ** (np.log10(M_1_C) + Ac_C * deltac[i] + Bc_C * fenv[i])

                ncen = n_cen_CSMF(
                    mass[i],
                    Mstar_low_C,
                    Mstar_up_C,
                    M_1_C_temp,
                    M_0_C,
                    gamma1_C,
                    gamma2_C,
                    sigma_c_C,
                )

                CSMF_marker += ncen * ic_C * multis[i]

            if randoms[i] <= CSMF_marker:
                Nout[tid, 0, 0] += 1  # counting
                keep[i] = 1
            else:
                keep[i] = 0

    # compose galaxy array, first create array of galaxy starting indices for the threads
    gstart = np.empty((Nthread + 1, 1), dtype=np.int64)
    gstart[0, :] = 0
    gstart[1:, 0] = Nout[:, 0, 0].cumsum()

    # galaxy arrays
    N_CSMF = gstart[-1, 0]
    CSMF_x = np.empty(N_CSMF, dtype=mass.dtype)
    CSMF_y = np.empty(N_CSMF, dtype=mass.dtype)
    CSMF_z = np.empty(N_CSMF, dtype=mass.dtype)
    CSMF_vx = np.empty(N_CSMF, dtype=mass.dtype)
    CSMF_vy = np.empty(N_CSMF, dtype=mass.dtype)
    CSMF_vz = np.empty(N_CSMF, dtype=mass.dtype)
    CSMF_mass = np.empty(N_CSMF, dtype=mass.dtype)
    CSMF_stellarmass = np.empty(N_CSMF, dtype=mass.dtype)
    CSMF_id = np.empty(N_CSMF, dtype=ids.dtype)

    # fill in the galaxy arrays
    for tid in numba.prange(Nthread):
        j1 = gstart[tid]
        for i in range(hstart[tid], hstart[tid + 1]):
            if keep[i] == 1:
                # loop thru three directions to assign galaxy velocities and positions
                CSMF_x[j1] = pos[i, 0]
                CSMF_vx[j1] = vel[i, 0] + alpha_c_C * vdev[i, 0]  # velocity bias
                CSMF_y[j1] = pos[i, 1]
                CSMF_vy[j1] = vel[i, 1] + alpha_c_C * vdev[i, 1]  # velocity bias
                CSMF_z[j1] = pos[i, 2]
                CSMF_vz[j1] = vel[i, 2] + alpha_c_C * vdev[i, 2]  # velocity bias

                # rsd only applies to the z direction
                if rsd and origin is not None:
                    nx = CSMF_x[j1] - origin[0]
                    ny = CSMF_y[j1] - origin[1]
                    nz = CSMF_z[j1] - origin[2]
                    inv_norm = 1.0 / np.sqrt(nx * nx + ny * ny + nz * nz)
                    nx *= inv_norm
                    ny *= inv_norm
                    nz *= inv_norm
                    proj = inv_velz2kms * (
                        CSMF_vx[j1] * nx + CSMF_vy[j1] * ny + CSMF_vz[j1] * nz
                    )
                    CSMF_x[j1] = CSMF_x[j1] + proj * nx
                    CSMF_y[j1] = CSMF_y[j1] + proj * ny
                    CSMF_z[j1] = CSMF_z[j1] + proj * nz
                elif rsd:
                    CSMF_z[j1] = wrap(pos[i, 2] + CSMF_vz[j1] * inv_velz2kms, lbox)

                CSMF_mass[j1] = mass[i]
                M_1_C_temp = 10 ** (np.log10(M_1_C) + Ac_C * deltac[i] + Bc_C * fenv[i])
                CSMF_stellarmass[j1] = get_random_cen_stellarmass_linearinterpolation(
                    mass[i],
                    Mstar_low_C,
                    Mstar_up_C,
                    M_1_C_temp,
                    M_0_C,
                    gamma1_C,
                    gamma2_C,
                    sigma_c_C,
                )
                CSMF_id[j1] = ids[i]
                j1 += 1
        # assert j == gstart[tid + 1]

    CSMF_dict = Dict.empty(key_type=types.unicode_type, value_type=float_array)
    ID_dict = Dict.empty(key_type=types.unicode_type, value_type=int_array)

    CSMF_dict['x'] = CSMF_x
    CSMF_dict['y'] = CSMF_y
    CSMF_dict['z'] = CSMF_z
    CSMF_dict['vx'] = CSMF_vx
    CSMF_dict['vy'] = CSMF_vy
    CSMF_dict['vz'] = CSMF_vz
    CSMF_dict['mass'] = CSMF_mass
    CSMF_dict['stellarmass'] = CSMF_stellarmass
    ID_dict['CSMF'] = CSMF_id
    return CSMF_dict, ID_dict, keep


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
    CSMF_hod_dict,
    want_CSMF,
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
    if want_CSMF:
        (
            Mstar_low_C,
            Mstar_up_C,
            M_1_C,
            M_0_C,
            gamma1_C,
            gamma2_C,
            sigma_c_C,
            a1_C,
            a2_C,
            M2_C,
            b0_C,
            b1_C,
            b2_C,
            delta1_C,
            delta2_C,
        ) = (
            CSMF_hod_dict['Mstar_low'],
            CSMF_hod_dict['Mstar_up'],
            CSMF_hod_dict['M_1'],
            CSMF_hod_dict['M_0'],
            CSMF_hod_dict['gamma_1'],
            CSMF_hod_dict['gamma_2'],
            CSMF_hod_dict['sigma_c'],
            CSMF_hod_dict['a_1'],
            CSMF_hod_dict['a_2'],
            CSMF_hod_dict['M_2'],
            CSMF_hod_dict['b_0'],
            CSMF_hod_dict['b_1'],
            CSMF_hod_dict['b_2'],
            CSMF_hod_dict['delta_1'],
            CSMF_hod_dict['delta_2'],
        )
        Ac_C, As_C, Bc_C, Bs_C, ic_C = (
            CSMF_hod_dict['Acent'],
            CSMF_hod_dict['Asat'],
            CSMF_hod_dict['Bcent'],
            CSMF_hod_dict['Bsat'],
            CSMF_hod_dict['ic'],
        )
        f_sigv_C = CSMF_hod_dict['f_sigv']

    numba.set_num_threads(Nthread)

    # compute nsate for each halo
    # figuring out the number of particles kept for each thread
    num_sats_C = np.zeros(len(hid), dtype=np.int64)
    stellarmass_C = np.zeros(len(hid), dtype=np.int64)
    hstart = np.rint(np.linspace(0, len(hid), Nthread + 1)).astype(
        np.int64
    )  # starting index of each thread
    for tid in range(Nthread):
        for i in range(hstart[tid], hstart[tid + 1]):
            if want_CSMF:
                M_1_C_temp = 10 ** (
                    np.log10(M_1_C) + Ac_C * hdeltac[i] + Bc_C * hfenv[i]
                )
                a1_C_temp = a1_C + As_C * hdeltac[i] + Bs_C * hfenv[i]
                base_p_C = (
                    n_sat_CSMF(
                        hmass[i],
                        Mstar_low_C,
                        Mstar_up_C,
                        M_1_C_temp,
                        M_0_C,
                        gamma1_C,
                        gamma2_C,
                        sigma_c_C,
                        a1_C_temp,
                        a2_C,
                        M2_C,
                        b0_C,
                        b1_C,
                        b2_C,
                        delta1_C,
                        delta2_C,
                    )
                    * ic_C
                )
                num_sats_C[i] = np.random.poisson(base_p_C)

    # generate rdpos
    rd_pos_C = getPointsOnSphere(np.sum(num_sats_C), Nthread)

    # put satellites on NFW
    h_id_C, x_sat_C, y_sat_C, z_sat_C, vx_sat_C, vy_sat_C, vz_sat_C, M_C = (
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
            rd_pos_C,
            num_sats_C,
            f_sigv_C,
            vel_sat,
            Nthread,
            exp_frac,
            exp_scale,
            nfw_rescale,
        )
    )

    # do rsd
    if rsd:
        z_sat_C = (z_sat_C + vz_sat_C * inv_velz2kms) % lbox

    CSMF_dict = Dict.empty(key_type=types.unicode_type, value_type=float_array)
    ID_dict = Dict.empty(key_type=types.unicode_type, value_type=int_array)

    CSMF_dict['x'] = x_sat_C
    CSMF_dict['y'] = y_sat_C
    CSMF_dict['z'] = z_sat_C
    CSMF_dict['vx'] = vx_sat_C
    CSMF_dict['vy'] = vy_sat_C
    CSMF_dict['vz'] = vz_sat_C
    CSMF_dict['mass'] = M_C
    stellarmass_C = np.empty_like(M_C)
    ## compute stellarmass of all the satelites
    if want_CSMF:
        hstart = np.rint(np.linspace(0, num_sats_C.sum(), Nthread + 1))
        for tid in numba.prange(Nthread):
            for i in range(int(hstart[tid]), int(hstart[tid + 1])):
                M_1_C_temp = 10 ** (
                    np.log10(M_1_C) + Ac_C * hdeltac[i] + Bc_C * hfenv[i]
                )
                a1_C_temp = a1_C + As_C * hdeltac[i] + Bs_C * hfenv[i]
                stellarmass_C[i] = get_random_sat_stellarmass_linearinterpolation(
                    M_C[i],
                    Mstar_low_C,
                    Mstar_up_C,
                    M_1_C_temp,
                    M_0_C,
                    gamma1_C,
                    gamma2_C,
                    a1_C_temp,
                    s,
                    a2_C,
                    M2_C,
                    b0_C,
                    b1_C,
                    b2_C,
                    delta1_C,
                    delta2_C,
                )

    CSMF_dict['stellarmass'] = stellarmass_C
    ID_dict['CSMF'] = h_id_C

    return CSMF_dict, ID_dict


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
    CSMF_hod_dict,
    rsd,
    inv_velz2kms,
    lbox,
    Mpart,
    want_CSMF,
    Nthread,
    origin,
    keep_cent,
):
    """
    Generate satellite galaxies in place in memory with a two pass numba parallel implementation.
    """

    if want_CSMF:
        (
            Mstar_low_C,
            Mstar_up_C,
            M_1_C,
            M_0_C,
            gamma1_C,
            gamma2_C,
            sigma_c_C,
            a1_C,
            a2_C,
            M2_C,
            b0_C,
            b1_C,
            b2_C,
            delta1_C,
            delta2_C,
        ) = (
            CSMF_hod_dict['Mstar_low'],
            CSMF_hod_dict['Mstar_up'],
            CSMF_hod_dict['M_1'],
            CSMF_hod_dict['M_0'],
            CSMF_hod_dict['gamma_1'],
            CSMF_hod_dict['gamma_2'],
            CSMF_hod_dict['sigma_c'],
            CSMF_hod_dict['a_1'],
            CSMF_hod_dict['a_2'],
            CSMF_hod_dict['M_2'],
            CSMF_hod_dict['b_0'],
            CSMF_hod_dict['b_1'],
            CSMF_hod_dict['b_2'],
            CSMF_hod_dict['delta_1'],
            CSMF_hod_dict['delta_2'],
        )

        alpha_s_C, s_C, s_v_C, s_p_C, s_r_C, Ac_C, As_C, Bc_C, Bs_C, ic_C = (
            CSMF_hod_dict['alpha_s'],
            CSMF_hod_dict['s'],
            CSMF_hod_dict['s_v'],
            CSMF_hod_dict['s_p'],
            CSMF_hod_dict['s_r'],
            CSMF_hod_dict['Acent'],
            CSMF_hod_dict['Asat'],
            CSMF_hod_dict['Bcent'],
            CSMF_hod_dict['Bsat'],
            CSMF_hod_dict['ic'],
        )

    H = len(hmass)  # num of particles

    numba.set_num_threads(Nthread)
    Nout = np.zeros((Nthread, 1, 8), dtype=np.int64)
    hstart = np.rint(np.linspace(0, H, Nthread + 1)).astype(
        np.int64
    )  # starting index of each thread

    keep = np.empty(H, dtype=np.int8)  # mask array tracking which halos to keep

    # figuring out the number of particles kept for each thread
    for tid in numba.prange(Nthread):  # numba.prange(Nthread):
        for i in range(hstart[tid], hstart[tid + 1]):
            # print(logM1, As, hdeltac[i], Bs, hfenv[i])
            CSMF_marker = 0
            if want_CSMF:
                M_1_C_temp = 10 ** (
                    np.log10(M_1_C) + Ac_C * hdeltac[i] + Bc_C * hfenv[i]
                )
                a1_C_temp = a1_C + As_C * hdeltac[i] + Bs_C * hfenv[i]
                base_p_C = (
                    n_sat_CSMF(
                        hmass[i],
                        Mstar_low_C,
                        Mstar_up_C,
                        M_1_C_temp,
                        M_0_C,
                        gamma1_C,
                        gamma2_C,
                        sigma_c_C,
                        a1_C_temp,
                        a2_C,
                        M2_C,
                        b0_C,
                        b1_C,
                        b2_C,
                        delta1_C,
                        delta2_C,
                    )
                    * weights[i]
                    * ic_C
                )
                if enable_ranks:
                    decorator_C = (
                        1
                        + s_C * ranks[i]
                        + s_v_C * ranksv[i]
                        + s_p_C * ranksp[i]
                        + s_r_C * ranksr[i]
                    )
                    exp_sat = base_p_C * decorator_C
                else:
                    exp_sat = base_p_C
                CSMF_marker += exp_sat

            if randoms[i] <= CSMF_marker:
                Nout[tid, 0, 0] += 1  # counting
                keep[i] = 1
            else:
                keep[i] = 0

    # compose galaxy array, first create array of galaxy starting indices for the threads
    gstart = np.empty((Nthread + 1, 1), dtype=np.int64)
    gstart[0, :] = 0
    gstart[1:, 0] = Nout[:, 0, 0].cumsum()

    # galaxy arrays
    N_CSMF = gstart[-1, 0]
    CSMF_x = np.empty(N_CSMF, dtype=hmass.dtype)
    CSMF_y = np.empty(N_CSMF, dtype=hmass.dtype)
    CSMF_z = np.empty(N_CSMF, dtype=hmass.dtype)
    CSMF_vx = np.empty(N_CSMF, dtype=hmass.dtype)
    CSMF_vy = np.empty(N_CSMF, dtype=hmass.dtype)
    CSMF_vz = np.empty(N_CSMF, dtype=hmass.dtype)
    CSMF_mass = np.empty(N_CSMF, dtype=hmass.dtype)
    CSMF_stellarmass = np.empty(N_CSMF, dtype=hmass.dtype)
    CSMF_id = np.empty(N_CSMF, dtype=hid.dtype)

    # fill in the galaxy arrays
    for tid in numba.prange(Nthread):
        j1 = gstart[tid]
        for i in range(hstart[tid], hstart[tid + 1]):
            if keep[i] == 1:
                CSMF_x[j1] = ppos[i, 0]
                CSMF_vx[j1] = hvel[i, 0] + alpha_s_C * (
                    pvel[i, 0] - hvel[i, 0]
                )  # velocity bias
                CSMF_y[j1] = ppos[i, 1]
                CSMF_vy[j1] = hvel[i, 1] + alpha_s_C * (
                    pvel[i, 1] - hvel[i, 1]
                )  # velocity bias
                CSMF_z[j1] = ppos[i, 2]
                CSMF_vz[j1] = hvel[i, 2] + alpha_s_C * (
                    pvel[i, 2] - hvel[i, 2]
                )  # velocity bias
                if rsd and origin is not None:
                    nx = CSMF_x[j1] - origin[0]
                    ny = CSMF_y[j1] - origin[1]
                    nz = CSMF_z[j1] - origin[2]
                    inv_norm = 1.0 / np.sqrt(nx * nx + ny * ny + nz * nz)
                    nx *= inv_norm
                    ny *= inv_norm
                    nz *= inv_norm
                    proj = inv_velz2kms * (
                        CSMF_vx[j1] * nx + CSMF_vy[j1] * ny + CSMF_vz[j1] * nz
                    )
                    CSMF_x[j1] = CSMF_x[j1] + proj * nx
                    CSMF_y[j1] = CSMF_y[j1] + proj * ny
                    CSMF_z[j1] = CSMF_z[j1] + proj * nz
                elif rsd:
                    CSMF_z[j1] = wrap(CSMF_z[j1] + CSMF_vz[j1] * inv_velz2kms, lbox)

                M_1_C_temp = 10 ** (
                    np.log10(M_1_C) + Ac_C * hdeltac[i] + Bc_C * hfenv[i]
                )
                a1_C_temp = a1_C + As_C * hdeltac[i] + Bs_C * hfenv[i]
                CSMF_stellarmass[j1] = get_random_sat_stellarmass(
                    hmass[i],
                    Mstar_low_C,
                    Mstar_up_C,
                    M_1_C_temp,
                    M_0_C,
                    gamma1_C,
                    gamma2_C,
                    a1_C_temp,
                    a2_C,
                    M2_C,
                    b0_C,
                    b1_C,
                    b2_C,
                    delta1_C,
                    delta2_C,
                )
                CSMF_mass[j1] = hmass[i]
                CSMF_id[j1] = hid[i]
                j1 += 1
        # assert j == gstart[tid + 1]

    CSMF_dict = Dict.empty(key_type=types.unicode_type, value_type=float_array)
    ID_dict = Dict.empty(key_type=types.unicode_type, value_type=int_array)

    CSMF_dict['x'] = CSMF_x
    CSMF_dict['y'] = CSMF_y
    CSMF_dict['z'] = CSMF_z
    CSMF_dict['vx'] = CSMF_vx
    CSMF_dict['vy'] = CSMF_vy
    CSMF_dict['vz'] = CSMF_vz
    CSMF_dict['mass'] = CSMF_mass
    CSMF_dict['stellarmass'] = CSMF_stellarmass
    ID_dict['CSMF'] = CSMF_id

    return CSMF_dict, ID_dict


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
        if tracer == 'CSMF':
            CSMF_HOD = tracers[tracer]

    if 'CSMF' in tracers.keys():
        want_CSMF = True

        CSMF_hod_dict = nb.typed.Dict.empty(
            key_type=nb.types.unicode_type, value_type=nb.types.float64
        )
        for key, value in CSMF_HOD.items():
            CSMF_hod_dict[key] = value

        CSMF_hod_dict['Acent'] = CSMF_HOD.get('Acent', 0.0)
        CSMF_hod_dict['Asat'] = CSMF_HOD.get('Asat', 0.0)
        CSMF_hod_dict['Bcent'] = CSMF_HOD.get('Bcent', 0.0)
        CSMF_hod_dict['Bsat'] = CSMF_HOD.get('Bsat', 0.0)
        CSMF_hod_dict['ic'] = CSMF_HOD.get('ic', 1.0)
        CSMF_hod_dict['f_sigv'] = CSMF_HOD.get('f_sigv', 0)

    else:
        want_CSMF = False
        CSMF_hod_dict = nb.typed.Dict.empty(
            key_type=nb.types.unicode_type, value_type=nb.types.float64
        )

    start = time.time()

    velz2kms = params['velz2kms']
    inv_velz2kms = 1 / velz2kms
    lbox = params['Lbox']
    origin = params['origin']

    CSMF_dict_cent, ID_dict_cent, keep_cent = gen_cent(
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
        CSMF_hod_dict,
        rsd,
        inv_velz2kms,
        lbox,
        want_CSMF,
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
        CSMF_dict_sat, ID_dict_sat = gen_sats_nfw(
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
            CSMF_hod_dict,
            want_CSMF,
            rsd,
            inv_velz2kms,
            lbox,
            keep_cent,
            Nthread=Nthread,
        )
    else:
        CSMF_dict_sat, ID_dict_sat = gen_sats(
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
            CSMF_hod_dict,
            rsd,
            inv_velz2kms,
            lbox,
            params['Mpart'],
            want_CSMF,
            Nthread,
            origin,
            keep_cent[subsample['pinds']],
        )
    if verbose:
        print('generating satellites took ', time.time() - start)

    # B.H. TODO: need a for loop above so we don't need to do this by hand
    HOD_dict_sat = {'CSMF': CSMF_dict_sat}
    HOD_dict_cent = {'CSMF': CSMF_dict_cent}

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


def gen_gal_cat_CSMF(
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
                    outdir / (f"{tracer}s_chunk{params['chunk']:d}.dat"),
                    overwrite=True,
                    format='ecsv',
                )

    return HOD_dict
