"""
Tools for applying variance reduction (ZCV and LCV) (based on scripts from Joe DeRose).
"""

import gc
from pathlib import Path
import warnings

import asdf
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from scipy.signal import savgol_filter

from abacusnbody.analysis.power_spectrum import (
    get_k_mu_edges,
    project_3d_to_poles,
    expand_poles_to_3d,
    get_smoothing,
)
from abacusnbody.metadata import get_meta
from .ic_fields import compress_asdf

try:
    from classy import Class
except ImportError as e:
    raise ImportError(
        'Missing imports for zcv. Install abacusutils with '
        '"pip install abacusutils[all]" to install zcv dependencies.'
    ) from e

from asdf.exceptions import AsdfWarning

warnings.filterwarnings('ignore', category=AsdfWarning)


def combine_spectra(k, spectra, bias_params, rsd=False, numerical_nabla=False):
    """
    ZCV: Given some bias parameters, compute the model power spectra.
    """

    if rsd:
        # shape is power spectrum templates, len(multipoles), len(k_modes)
        pkvec = np.zeros((14, spectra.shape[1], spectra.shape[2]))
        pkvec[:10, ...] = spectra[:10, ...]

        # read out bias terms
        bias_params = np.hstack([bias_params, np.zeros(5 - len(bias_params))])
        b1, b2, bs, bk2, sn = bias_params
        bias_monomials = np.array(
            [
                1,
                2 * b1,
                b1**2,
                b2,
                b1 * b2,
                0.25 * b2**2,
                2 * bs,
                2 * b1 * bs,
                b2 * bs,
                bs**2,
                2 * bk2,
                2 * bk2 * b1,
                bk2 * b2,
                2 * bk2 * bs,
            ]
        )

        # sum for each multipole and combine into pk
        p0 = np.sum(bias_monomials[:, np.newaxis] * pkvec[:, 0, :], axis=0)
        p2 = np.sum(bias_monomials[:, np.newaxis] * pkvec[:, 1, :], axis=0)
        p4 = np.sum(bias_monomials[:, np.newaxis] * pkvec[:, 2, :], axis=0)

        pk = np.stack([p0, p2, p4])
    else:
        # shape is power spectrum templates, len(k_modes)
        pkvec = np.zeros((14, spectra.shape[1]))

        # approximation for the nabla^2 terms: -k^2 <1, X> approximation.
        if numerical_nabla:
            pkvec[...] = spectra[:14]
        else:
            pkvec[:10, ...] = spectra[:10]
            nabla_idx = [0, 1, 3, 6]
            pkvec[10:, ...] = -(k[np.newaxis, :] ** 2) * pkvec[nabla_idx, ...]

        # read out bias terms
        bias_params = np.hstack([bias_params, np.zeros(5 - len(bias_params))])
        b1, b2, bs, bk2, sn = bias_params
        bias_monomials = np.array(
            [
                1,
                2 * b1,
                b1**2,
                b2,
                b2 * b1,
                0.25 * b2**2,
                2 * bs,
                2 * bs * b1,
                bs * b2,
                bs**2,
                2 * bk2,
                2 * bk2 * b1,
                bk2 * b2,
                2 * bk2 * bs,
            ]
        )

        # bterms has b entries and pkvec has b,k entries and we sum over the b's
        pk = np.einsum('b, bk->k', bias_monomials, pkvec) + sn
    return pk


def combine_cross_spectra(k, spectra, bias_params, rsd=False):
    """
    ZCV: Given some bias parameters, compute the model power spectra (no shotnoise in cross).
    """
    if rsd:
        # shape is cross power spectrum templates, len(multipoles), len(k_modes)
        pkvec = np.zeros((5, spectra.shape[1], spectra.shape[2]))
        pkvec[:5, ...] = spectra[:5, ...]
        bias_params = np.hstack([bias_params, np.zeros(5 - len(bias_params))])
        b1, b2, bs, bk, sn = bias_params
        bias_monomials = np.array([1, b1, 0.5 * b2, bs, bk])
        p0 = np.sum(bias_monomials[:, np.newaxis] * pkvec[:, 0, :], axis=0)  # + sn
        p2 = np.sum(bias_monomials[:, np.newaxis] * pkvec[:, 1, :], axis=0)  # + sn2
        p4 = np.sum(bias_monomials[:, np.newaxis] * pkvec[:, 2, :], axis=0)  # + sn4
        pk = np.stack([p0, p2, p4])
    else:
        # shape is cross power spectrum templates, len(k_modes)
        pkvec = np.zeros((5, spectra.shape[1]))
        pkvec[:5, ...] = spectra[:5, ...]
        bias_params = np.hstack([bias_params, np.zeros(5 - len(bias_params))])
        b1, b2, bs, bk, sn = bias_params
        bias_monomials = np.array([1, b1, 0.5 * b2, bs, bk])
        pk = np.sum(bias_monomials[:, np.newaxis] * pkvec[:, :], axis=0)  # + sn
    return pk


def combine_cross_kaiser_spectra(
    k, spectra_dict, D, bias, f_growth, rec_algo, R, rsd=False
):
    """
    LCV: Convert measured templates into tracer-model power spectrum given bias parameters
    assuming the Kaiser approximation (Chen et al. 2019, 1907.00043).

    RecSym equation:
        < D (b delta + f mu2 delta, tr > = D * (b < delta, tr> + f < mu2 delta, tr >)

    RecIso equation:
        < D ((b+f mu^2)(1-S) + bS) delta, tr > = D * (b < delta, tr > + f (1-S) < delta, mu^2 >)
    """
    if rec_algo == 'recsym':
        if rsd:
            pk = D * (
                bias * spectra_dict['P_ell_delta_tr']
                + f_growth * spectra_dict['P_ell_deltamu2_tr']
            )
        else:
            pk = D * (
                bias * spectra_dict['P_kmu_delta_tr']
                + f_growth * spectra_dict['P_kmu_deltamu2_tr']
            )
    elif rec_algo == 'reciso':
        assert R is not None
        S = np.exp(-(k**2) * R**2 / 2.0)
        f_eff = f_growth * (1.0 - S)
        if rsd:
            f_eff = f_eff.reshape(1, len(k), 1)
            pk = D * (
                bias * spectra_dict['P_ell_delta_tr']
                + f_eff * spectra_dict['P_ell_deltamu2_tr']
            )
        else:
            pk = D * (
                bias * spectra_dict['P_kmu_delta_tr']
                + f_eff * spectra_dict['P_kmu_deltamu2_tr']
            )
    return pk


def combine_kaiser_spectra(k, spectra_dict, D, bias, f_growth, rec_algo, R, rsd=False):
    """
    LCV: Convert measured templates into model-model power spectrum given bias parameters
    assuming the Kaiser approximation (Chen et al. 2019, 1907.00043).

    RecSym equation:
        < D (b delta + f mu2 delta, D (b delta + f mu2 delta) > =
        = D^2 (b^2 < delta, delta> + f^2 < mu2 delta, mu2 delta > + 2 b f < delta, mu2 delta >)

    RecIso Equation:
        ((b+f mu^2)(1-S) + bS) delta = (b + f (1-S) mu2) delta
        < D ((b+f mu^2)(1-S) + bS) delta, D ((b+f mu^2)(1-S) + bS) delta > =
        = D^2 (b^2 < delta, delta> + fefff^2 < mu2 delta, mu2 delta > + 2 b feff < delta, mu2 delta >)
    """
    if rec_algo == 'recsym':
        if rsd:
            pk = D**2 * (
                2.0 * bias * f_growth * spectra_dict['P_ell_deltamu2_delta']
                + f_growth**2 * spectra_dict['P_ell_deltamu2_deltamu2']
                + bias**2 * spectra_dict['P_ell_delta_delta']
            )
        else:
            pk = D**2 * (
                2.0 * bias * f_growth * spectra_dict['P_kmu_deltamu2_delta']
                + f_growth**2 * spectra_dict['P_kmu_deltamu2_deltamu2']
                + bias**2 * spectra_dict['P_kmu_delta_delta']
            )
    elif rec_algo == 'reciso':
        assert R is not None
        S = np.exp(-(k**2) * R**2 / 2.0)
        f_eff = f_growth * (1.0 - S)
        if rsd:
            f_eff = f_eff.reshape(1, len(k), 1)
            pk = D**2 * (
                2.0 * bias * f_eff * spectra_dict['P_ell_deltamu2_delta']
                + f_eff**2 * spectra_dict['P_ell_deltamu2_deltamu2']
                + bias**2 * spectra_dict['P_ell_delta_delta']
            )
        else:
            pk = D**2 * (
                2.0 * bias * f_eff * spectra_dict['P_kmu_deltamu2_delta']
                + f_eff**2 * spectra_dict['P_kmu_deltamu2_deltamu2']
                + bias**2 * spectra_dict['P_kmu_delta_delta']
            )
    return pk


def get_poles(k, pk, D, bias, f_growth, poles=[0, 2, 4]):
    """
    Compute the len(poles) multipoles given the linear power spectrum, pk, the growth function,
    the growth factor and the bias.
    """
    beta = f_growth / bias
    p_ell = np.zeros((len(poles), len(k)))
    for i, pole in enumerate(poles):
        if pole == 0:
            p_ell[i] = (1.0 + 2.0 / 3.0 * beta + 1.0 / 5 * beta**2) * pk
        elif pole == 2:
            p_ell[i] = (4.0 / 3.0 * beta + 4.0 / 7 * beta**2) * pk
        elif pole == 4:
            p_ell[i] = (8.0 / 35 * beta**2) * pk
    p_ell *= bias**2 * D**2
    return k, p_ell


def multipole_cov(pell, ell):
    """
    Factors appearing in the covariance matrix that couple the multipoles.
    """
    if ell == 0:
        cov = 2 * pell[0, :] ** 2 + 2 / 5 * pell[1, :] ** 2 + 2 / 9 * pell[2, :] ** 2

    elif ell == 2:
        cov = (
            2 / 5 * pell[0, :] ** 2
            + 6 / 35 * pell[1, :] ** 2
            + 3578 / 45045 * pell[2, :] ** 2
            + 8 / 35 * pell[0, :] * pell[1, :]
            + 8 / 35 * pell[0, :] * pell[2, :]
            + 48 / 385 * pell[1, :] * pell[2, :]
        )

    elif ell == 4:
        cov = (
            2 / 9 * pell[0, :] ** 2
            + 3578 / 45045 * pell[1, :] ** 2
            + 1058 / 17017 * pell[2, :] ** 2
            + 80 / 693 * pell[0, :] * pell[1, :]
            + 72 / 1001 * pell[0, :] * pell[2, :]
            + 80 / 1001 * pell[1, :] * pell[2, :]
        )

    return cov


def measure_2pt_bias(k, pk_ij, pk_tt, kmax, keynames, kmin=0.0, rsd=False):
    """
    ZCV: Infer the bias based on the template power spectrum and tracer measurements.
    Note that the bias parameter ordering corresponds to rsd == False.
    """
    # apply cuts to power spectra
    kidx_max = k.searchsorted(kmax)
    kidx_min = k.searchsorted(kmin)
    kidx_min = np.max([kidx_min, 1])  # matters!
    kcut = k[kidx_min:kidx_max]
    pk_tt_kcut = pk_tt[kidx_min:kidx_max]
    pk_ij_kcut = pk_ij[:, kidx_min:kidx_max]

    # initial guesses for the biases (keynames starts with 1cb)
    bvec0 = np.zeros(len(keynames))  # b1, b2, bs, bn, sn

    # minimize loss function
    def loss(bvec):
        return np.sum(
            (
                pk_tt_kcut
                - combine_spectra(
                    kcut,
                    pk_ij_kcut,
                    np.hstack([bvec[:-1], np.zeros(5 - len(bvec)), bvec[-1]]),
                    rsd=rsd,
                )
            )
            ** 2
            / (2 * pk_tt_kcut**2)
        )

    out = minimize(loss, bvec0)
    return out


def combine_field_spectra_k3D_lcv(
    bias, f_growth, D, power_lin_fns, power_rsd_tr_fns, nmesh, Lbox, R, rec_algo
):
    """
    LCV: Given bias parameters, compute the model-model auto and cross correlation for the 3D k-vector.
    """
    if rec_algo == 'reciso':
        S = get_smoothing(nmesh, Lbox, R)
        f_eff = f_growth * (1.0 - S)
        f_eff = f_eff.reshape(nmesh, nmesh, nmesh)
    elif rec_algo == 'recsym':
        f_eff = f_growth
    pk_tt = asdf.open(power_rsd_tr_fns[0])['data']['P_k3D_tr_tr']
    pk_ll = D**2 * (
        2.0 * bias * f_eff * asdf.open(power_lin_fns[1])['data']['P_k3D_deltamu2_delta']
        + f_eff**2 * asdf.open(power_lin_fns[2])['data']['P_k3D_deltamu2_deltamu2']
        + bias**2 * asdf.open(power_lin_fns[0])['data']['P_k3D_delta_delta']
    )
    pk_lt = D * (
        bias * asdf.open(power_rsd_tr_fns[1])['data']['P_k3D_delta_tr']
        + f_eff * asdf.open(power_rsd_tr_fns[2])['data']['P_k3D_deltamu2_tr']
    )
    return pk_tt, pk_ll, pk_lt


def combine_field_spectra_k3D(bias, power_ij_fns, keynames):
    """
    ZCV: Given bias parameters, compute the model-model cross correlation for the 3D k-vector.
    """
    # match convention of combine_spectra
    if len(bias) >= 3:
        bias[2] *= 0.5
    counter = 0
    for i in range(len(keynames)):
        for j in range(len(keynames)):
            if i < j:
                continue
            if i == 0 and j == 0:
                power = np.zeros_like(
                    asdf.open(power_ij_fns[counter])['data'][
                        f'P_k3D_{keynames[i]}_{keynames[j]}'
                    ]
                )
            if i == j:
                power += (
                    bias[i]
                    * bias[j]
                    * asdf.open(power_ij_fns[counter])['data'][
                        f'P_k3D_{keynames[i]}_{keynames[j]}'
                    ]
                )
            else:
                power += (
                    2.0
                    * bias[i]
                    * bias[j]
                    * asdf.open(power_ij_fns[counter])['data'][
                        f'P_k3D_{keynames[i]}_{keynames[j]}'
                    ]
                )
            counter += 1
    return power


def combine_field_cross_spectra_k3D(bias, power_tr_fns, keynames):
    """
    ZCV: Given bias parameters, compute the model-tracer cross correlation for the 3D k-vector.
    """
    # match convention of combine_spectra
    if len(bias) >= 3:
        bias[2] *= 0.5
    counter = 1  # first file (i.e., 0) is tracer-tracer, so we skip it
    for i in range(len(keynames)):
        if i == 0:
            power = np.zeros_like(
                asdf.open(power_tr_fns[counter])['data'][f'P_k3D_{keynames[i]}_tr']
            )
        power += (
            bias[i]
            * asdf.open(power_tr_fns[counter])['data'][f'P_k3D_{keynames[i]}_tr']
        )
        counter += 1
    return power


def measure_2pt_bias_lcv(
    k,
    power_dict,
    power_rsd_tr_dict,
    D,
    f_growth,
    kmax,
    rsd,
    rec_algo,
    R,
    ellmax=2,
    kmin=0.0,
):
    """
    LCV: Function for getting the linear bias in the Kaiser approximation.
    """
    # cut the tracer power spectrum in the k range of interest
    pk_tt = power_rsd_tr_dict['P_ell_tr_tr'][:ellmax, :]  # , 0]
    kidx_max = k.searchsorted(kmax)
    kidx_min = k.searchsorted(kmin)
    kcut = k[kidx_min:kidx_max]
    pk_tt_kcut = pk_tt[:ellmax, kidx_min:kidx_max]

    # cut the template power spectra for the same k range
    power_lin_dict = power_dict.copy()
    for key in power_lin_dict.keys():
        if 'P_ell' not in key:
            continue
        power_lin_dict[key] = power_lin_dict[key][:, kidx_min:kidx_max]

    # define loss function as the fractional difference squared
    def loss(bias):
        return np.sum(
            (
                pk_tt_kcut
                - combine_kaiser_spectra(
                    kcut, power_lin_dict, D, bias, f_growth, rec_algo, R, rsd=rsd
                )[:ellmax, :]
            )
            ** 2
            / (2 * pk_tt_kcut**2)
        )

    # fit for the bias
    out = minimize(loss, 1.0)
    return out


def read_power_dict(power_tr_dict, power_ij_dict, want_rsd, keynames, poles):
    """
    ZCV: Function for reading the power spectra and saving them in the same format as Zenbu.
    """
    k = power_tr_dict['k_binc'].flatten()
    mu = np.zeros((len(k), 1))
    if want_rsd:
        pk_tt = np.zeros((1, len(poles), len(k)))
        pk_ij_zz = np.zeros((15, len(poles), len(k)))
        pk_ij_zt = np.zeros((5, len(poles), len(k)))
    else:
        pk_tt = np.zeros((1, len(k), 1))
        pk_ij_zz = np.zeros((15, len(k), 1))
        pk_ij_zt = np.zeros((5, len(k), 1))

    if want_rsd:
        pk_tt[0, :, :] = power_tr_dict['P_ell_tr_tr'].reshape(len(poles), len(k))
        nmodes = power_tr_dict['N_ell_tr_tr'].flatten()
    else:
        pk_tt[0, :, :] = power_tr_dict['P_kmu_tr_tr'].reshape(len(k), 1)
        nmodes = power_tr_dict['N_kmu_tr_tr'].flatten()

    count = 0
    for i in range(len(keynames)):
        if want_rsd:
            pk_ij_zt[i, :, :] = power_tr_dict[f'P_ell_{keynames[i]}_tr'].reshape(
                len(poles), len(k)
            )
        else:
            pk_ij_zt[i, :, :] = power_tr_dict[f'P_kmu_{keynames[i]}_tr'].reshape(
                len(k), 1
            )
        for j in range(len(keynames)):
            if i < j:
                continue
            if want_rsd:
                pk_ij_zz[count, :, :] = power_ij_dict[
                    f'P_ell_{keynames[i]}_{keynames[j]}'
                ].reshape(len(poles), len(k))
            else:
                pk_ij_zz[count, :, :] = power_ij_dict[
                    f'P_kmu_{keynames[i]}_{keynames[j]}'
                ].reshape(len(k), 1)
            count += 1

    print(
        'zeros in the measured power spectra = ',
        np.sum(pk_tt == 0.0),
        np.sum(pk_ij_zz == 0.0),
        np.sum(pk_ij_zt == 0.0),
    )
    return k, mu, pk_tt, pk_ij_zz, pk_ij_zt, nmodes


def get_cfg(sim_name, z_this, nmesh):
    """
    ZCV: Configuration parameters.
    """
    meta = get_meta(sim_name, redshift=z_this)
    Lbox = meta['BoxSize']
    z_ic = meta['InitialRedshift']
    # k_Ny = np.pi*nmesh/Lbox
    cosmo = {}
    cosmo['output'] = 'mPk mTk'
    cosmo['P_k_max_h/Mpc'] = 20.0
    int(sim_name.split('ph')[-1])
    for k in (
        'H0',
        'omega_b',
        'omega_cdm',
        'omega_ncdm',
        'N_ncdm',
        'N_ur',
        'n_s',
        'A_s',
        'alpha_s',
        #'wa', 'w0',
    ):
        cosmo[k] = meta[k]

    # create a dict with everything you would ever need
    cfg = {}
    cfg['lbox'] = Lbox
    cfg['Cosmology'] = cosmo
    cfg['z_ic'] = z_ic
    return cfg


def run_zcv(power_rsd_tr_dict, power_rsd_ij_dict, power_tr_dict, power_ij_dict, config):
    """
    Apply Zel'dovich control variates (ZCV) reduction to some measured power spectrum.
    """
    # read out some parameters from the config function
    sim_name = config['sim_params']['sim_name']
    z_this = config['sim_params']['z_mock']
    zcv_dir = config['zcv_params']['zcv_dir']
    nmesh = config['zcv_params']['nmesh']
    kcut = config['zcv_params']['kcut']
    keynames = np.array(config['zcv_params']['fields'])
    kmax = config['zcv_params'].get('kmax_fit', 0.15)
    want_rsd = config['HOD_params']['want_rsd']
    rsd_str = '_rsd' if want_rsd else ''
    fields = np.array(['1cb', 'delta', 'delta2', 'tidal2', 'nabla2'])
    assert (fields[: len(keynames)] == keynames).all(), (
        'Requested keynames should follow the standard order'
    )
    assert nmesh == config['power_params']['nmesh'], (
        'nmesh from power_params need to match nmesh from zcv_params'
    )

    # smoothing parameters
    sg_window = config['zcv_params'].get('sg_window', 21)
    k0 = config['zcv_params'].get('k0_window', 0.618)
    dk_cv = config['zcv_params'].get('dk_window', 0.167)
    beta1_k = config['zcv_params'].get('beta1_k', 0.05)

    # power params
    k_hMpc_max = config['power_params']['k_hMpc_max']
    logk = config['power_params']['logk']
    n_k_bins = config['power_params']['nbins_k']
    n_mu_bins = config['power_params']['nbins_mu']
    poles = config['power_params']['poles']

    # create save directory
    save_dir = Path(zcv_dir) / sim_name
    save_z_dir = save_dir / f'z{z_this:.3f}'

    # linear power
    pk_lin_fn = save_dir / 'abacus_pk_lin_ic.dat'

    # read the config params
    cfg = get_cfg(sim_name, z_this, nmesh)
    cfg['p_lin_ic_file'] = str(pk_lin_fn)
    cfg['nmesh_in'] = nmesh
    cfg['poles'] = poles
    cfg['surrogate_gaussian_cutoff'] = kcut
    Lbox = cfg['lbox']

    # define k bins
    k_bins, mu_bins = get_k_mu_edges(Lbox, k_hMpc_max, n_k_bins, n_mu_bins, logk)
    k_binc = (k_bins[1:] + k_bins[:-1]) * 0.5

    # name of files to read from
    if not logk:
        dk = k_bins[1] - k_bins[0]
    else:
        dk = np.log(k_bins[1] / k_bins[0])
    if n_k_bins == nmesh // 2:
        zenbu_fn = save_z_dir / f'zenbu_pk{rsd_str}_ij_lpt_nmesh{nmesh:d}.npz'
        window_fn = save_dir / f'window_nmesh{nmesh:d}.npz'
    else:
        zenbu_fn = (
            save_z_dir / f'zenbu_pk{rsd_str}_ij_lpt_nmesh{nmesh:d}_dk{dk:.3f}.npz'
        )
        window_fn = save_dir / f'window_nmesh{nmesh:d}_dk{dk:.3f}.npz'

    # change names
    if not want_rsd:
        power_tr_dict, power_ij_dict = power_rsd_tr_dict, power_rsd_ij_dict

    # load real-space version (used for bias fitting)
    k, mu, pk_tt_real, pk_ij_zz_real, pk_ij_zt_real, nmodes = read_power_dict(
        power_tr_dict, power_ij_dict, want_rsd=False, keynames=keynames, poles=poles
    )

    # load either real or redshift
    k, mu, pk_tt_poles, pk_ij_zz_poles, pk_ij_zt_poles, nmodes = read_power_dict(
        power_rsd_tr_dict,
        power_rsd_ij_dict,
        want_rsd=want_rsd,
        keynames=keynames,
        poles=poles,
    )
    assert np.isclose(k, k_binc).all()

    # measure bias in real space
    bvec_opt = measure_2pt_bias(
        k, pk_ij_zz_real[:, :, 0], pk_tt_real[0, :, 0], kmax, keynames, rsd=False
    )
    bias_vec = np.hstack(
        [1.0, bvec_opt['x'][:-1], np.zeros(5 - len(bvec_opt['x'])), bvec_opt['x'][-1]]
    )  # 1, b1, b2, bs, bn, sn
    print('bias', bias_vec)

    # decide what to input depending on whether rsd requested or not
    if want_rsd:
        pk_tt_input = pk_tt_poles[0, ...]
        pk_ij_zz_input = pk_ij_zz_poles
        pk_ij_zt_input = pk_ij_zt_poles
    else:
        pk_tt_input = pk_tt_poles[0, :, 0]
        pk_ij_zz_input = pk_ij_zz_poles[:, :, 0]
        pk_ij_zt_input = pk_ij_zt_poles[:, :, 0]

    # load the presaved window function
    data = np.load(window_fn)
    window = data['window']
    keff = data['keff']
    assert len(keff) == len(k_binc), f'Mismatching file: {str(window_fn)}'
    assert np.abs(keff[-1] - k_binc[-1]) / k_binc[-1] < 0.1, (
        f'Mismatching file: {str(window_fn)}'
    )

    # load the presaved zenbu power spectra
    data = np.load(zenbu_fn)
    pk_ij_zenbu = data['pk_ij_zenbu']
    assert np.allclose(data['k_binc'], k_binc), f'Mismatching file: {str(zenbu_fn)}'
    assert np.isclose(data['kcut'], kcut), f'Mismatching file: {str(zenbu_fn)}'

    # combine spectra drops the bias = 1 element
    pk_zz = combine_spectra(k_binc, pk_ij_zz_input, bias_vec[1:], rsd=want_rsd)
    pk_zenbu = combine_spectra(k_binc, pk_ij_zenbu, bias_vec[1:], rsd=want_rsd)
    pk_zn = combine_cross_spectra(k_binc, pk_ij_zt_input, bias_vec[1:], rsd=want_rsd)

    # compute the stochasticity power spectrum
    shotnoise = (pk_tt_input - 2.0 * pk_zn + pk_zz)[0]
    pk_nn_nosn = pk_tt_input.copy()
    pk_nn_nosn[0] -= shotnoise

    # compute disconnected covariance
    if want_rsd:
        cov_zn = np.stack([multipole_cov(pk_zn, ell) for ell in poles])
        var_zz = np.stack([multipole_cov(pk_zz, ell) for ell in poles])
        var_nn = np.stack([multipole_cov(pk_tt_input, ell) for ell in poles])
        var_nn_nosn = np.stack([multipole_cov(pk_nn_nosn, ell) for ell in poles])
    else:
        cov_zn = 2 * pk_zn**2
        var_zz = 2 * pk_zz**2
        var_nn = 2 * pk_tt_input**2
        var_nn_nosn = 2.0 * (pk_nn_nosn) ** 2

    with np.errstate(divide='ignore'):
        # shotnoise limit
        r_zt_sn_lim = var_nn_nosn / np.sqrt(var_nn * var_nn_nosn)
        # r_zt_sn_lim[np.isclose(var_nn * var_nn_nosn, 0.)] = 0.

        # beta parameter
        beta = cov_zn / var_zz
        # beta[np.isclose(var_zz, 0.)] = 0.
    beta_damp = 0.5 * (1 - np.tanh((k_binc - k0) / dk_cv)) * beta
    beta_damp = np.atleast_2d(beta_damp)
    beta_damp[beta_damp != beta_damp] = 0
    beta_damp[:, : k_binc.searchsorted(beta1_k)] = 1
    beta_smooth = np.zeros_like(beta_damp)
    for i in range(beta_smooth.shape[0]):
        try:
            beta_smooth[i, :] = savgol_filter(beta_damp.T[:, i], sg_window, 3)
        except ValueError:
            warnings.warn('This message should only appear when doing a smoke test.')

    # cross-correlation coefficient
    with np.errstate(divide='ignore'):
        r_zt = cov_zn / np.sqrt(var_zz * var_nn)
        r_zt[np.isclose(r_zt, 0.0)] = 0.0
    r_zt = np.atleast_2d(r_zt)
    r_zt[r_zt != r_zt] = 0  # takes care of NaN's

    # apply window function if in rsd
    if want_rsd:
        pk_zenbu = np.hstack(pk_zenbu)
        pk_zenbu = np.dot(window.T, pk_zenbu).reshape(len(poles), -1)

    # beta needs to be smooth for best results
    pk_nn_betasmooth = pk_tt_input - beta_smooth * (pk_zz - pk_zenbu)

    # save results to a dictionary
    zcv_dict = {}
    zcv_dict['k_binc'] = k_binc
    zcv_dict['poles'] = poles
    zcv_dict['rho_tr_ZD'] = r_zt
    zcv_dict['rho_tr_ZD_sn_lim'] = r_zt_sn_lim
    zcv_dict['Pk_ZD_ZD_ell'] = pk_zz
    zcv_dict['Pk_tr_ZD_ell'] = pk_zn
    zcv_dict['Pk_tr_tr_ell'] = pk_tt_input
    zcv_dict['Nk_tr_tr_ell'] = nmodes
    zcv_dict['Pk_tr_tr_ell_zcv'] = pk_nn_betasmooth
    zcv_dict['Pk_ZD_ZD_ell_ZeNBu'] = pk_zenbu
    zcv_dict['bias'] = bias_vec[1:]
    return zcv_dict


def run_zcv_field(
    power_rsd_tr_fns, power_rsd_ij_fns, power_tr_fns, power_ij_fns, config
):
    """
    Apply Zel'dovich control variates (ZCV) reduction to some measured 3D power spectrum.
    """
    # read out some parameters from the config function
    sim_name = config['sim_params']['sim_name']
    z_this = config['sim_params']['z_mock']
    zcv_dir = config['zcv_params']['zcv_dir']
    nmesh = config['zcv_params']['nmesh']
    kcut = config['zcv_params']['kcut']
    keynames = np.array(config['zcv_params']['fields'])
    kmax = config['zcv_params'].get('kmax_fit', 0.15)
    want_rsd = config['HOD_params']['want_rsd']
    rsd_str = '_rsd' if want_rsd else ''
    fields = np.array(['1cb', 'delta', 'delta2', 'tidal2', 'nabla2'])
    assert (fields[: len(keynames)] == keynames).all(), (
        'Requested keynames should follow the standard order'
    )
    assert nmesh == config['power_params']['nmesh'], (
        'nmesh from power_params need to match nmesh from zcv_params'
    )

    # create save directory
    save_dir = Path(zcv_dir) / sim_name
    save_z_dir = save_dir / f'z{z_this:.3f}'

    # linear power
    pk_lin_fn = save_dir / 'abacus_pk_lin_ic.dat'

    # read the config params
    cfg = get_cfg(sim_name, z_this, nmesh)
    cfg['p_lin_ic_file'] = str(pk_lin_fn)
    cfg['nmesh_in'] = nmesh
    cfg['poles'] = config['power_params']['poles']
    cfg['surrogate_gaussian_cutoff'] = kcut
    Lbox = cfg['lbox']

    # smoothing parameters
    sg_window = config['zcv_params'].get('sg_window', 21)
    k0 = config['zcv_params'].get('k0_window', 0.618)
    dk_cv = config['zcv_params'].get('dk_window', 0.167)
    beta1_k = config['zcv_params'].get('beta1_k', 0.05)

    # power params
    k_hMpc_max = config['power_params'].get('k_hMpc_max', np.pi * nmesh / Lbox)
    logk = config['power_params'].get('logk', False)
    n_k_bins = config['power_params'].get('nbins_k', nmesh // 2)
    n_mu_bins = config['power_params'].get('nbins_mu', 1)
    poles = config['power_params']['poles']

    # make sure that the parameters are set correctly
    if not (
        np.isclose(k_hMpc_max, np.pi * nmesh / Lbox) & logk
        == False & n_k_bins
        == nmesh // 2 & n_mu_bins
        == 1
    ):
        warnings.warn('Setting the parameters correctly for Xi computation')
        k_hMpc_max = np.pi * nmesh / Lbox
        logk = False
        n_k_bins = nmesh // 2
        n_mu_bins = 1

    # name of files to read from could add the _dk version
    zenbu_fn = save_z_dir / f'zenbu_pk{rsd_str}_ij_lpt_nmesh{nmesh:d}.npz'
    window_fn = save_dir / f'window_nmesh{nmesh:d}.npz'

    # file to save to
    power_cv_tr_fn = Path(save_z_dir) / f'power{rsd_str}_ZCV_tr_nmesh{nmesh:d}.asdf'

    # define k bins
    k_bins, mu_bins = get_k_mu_edges(Lbox, k_hMpc_max, n_k_bins, n_mu_bins, logk)
    k_binc = 0.5 * (k_bins[1:] + k_bins[:-1])

    # project 3d measurements in real space to monopole
    pk_nn = asdf.open(power_tr_fns[0])['data']['P_k3D_tr_tr']
    pk_nn = project_3d_to_poles(k_bins, pk_nn, Lbox, poles=[0])[0].flatten() / Lbox**3
    pk_ij = np.zeros((15, len(pk_nn)))
    counter = 0
    for i in range(len(keynames)):
        for j in range(len(keynames)):
            if i < j:
                continue
            print('Projecting', keynames[i], keynames[j])
            pk = asdf.open(power_ij_fns[counter])['data'][
                f'P_k3D_{keynames[i]}_{keynames[j]}'
            ]
            pk = project_3d_to_poles(k_bins, pk, Lbox, poles=[0])
            pk_ij[counter] = pk[0].flatten() / Lbox**3
            nmodes = pk[1].flatten()
            counter += 1

    # infer the bias in real space
    bvec_opt = measure_2pt_bias(k_binc, pk_ij, pk_nn, kmax, keynames, rsd=False)
    bias_vec = np.hstack(
        [1.0, bvec_opt['x'][:-1], np.zeros(5 - len(bvec_opt['x'])), bvec_opt['x'][-1]]
    )  # 1, b1, b2, bs, bn, sn
    print('bias', bias_vec)

    # load the presaved window function
    data = np.load(window_fn)

    # load the presaved zenbu power spectra
    data = np.load(zenbu_fn)
    pk_ij_zenbu = data['pk_ij_zenbu']
    assert np.allclose(data['k_binc'], k_binc), f'Mismatching file: {str(zenbu_fn)}'
    assert np.isclose(data['kcut'], kcut), f'Mismatching file: {str(zenbu_fn)}'

    # combine zenbu multipoles
    pk_zenbu = combine_spectra(k_binc, pk_ij_zenbu, bias_vec[1:], rsd=want_rsd)

    # combine the 3D power spectrum templates using the biases
    assert want_rsd, 'Currently only rsd version implemented'
    pk_nn = asdf.open(power_rsd_tr_fns[0])['data']['P_k3D_tr_tr']
    pk_zz = combine_field_spectra_k3D(bias_vec, power_rsd_ij_fns, keynames)
    pk_zn = combine_field_cross_spectra_k3D(bias_vec, power_rsd_tr_fns, keynames)

    # project 3D power spectra into multipoles
    pk_nn_proj = (
        project_3d_to_poles(k_bins, pk_nn, Lbox, poles)[0].reshape(
            len(poles), len(k_binc)
        )
        / Lbox**3
    )
    pk_zn_proj = (
        project_3d_to_poles(k_bins, pk_zn, Lbox, poles)[0].reshape(
            len(poles), len(k_binc)
        )
        / Lbox**3
    )
    del pk_zn
    gc.collect()
    pk_zz_proj = (
        project_3d_to_poles(k_bins, pk_zz, Lbox, poles)[0].reshape(
            len(poles), len(k_binc)
        )
        / Lbox**3
    )

    # expand zenbu to 3D power spectrum
    assert np.isclose(np.min(np.diff(k_binc)), np.max(np.diff(k_binc))), (
        'For custom interpolation, need equidistant k-values'
    )
    pk_zz[:, :, :] -= expand_poles_to_3d(
        k_binc, pk_zenbu, nmesh, Lbox, np.asarray(poles)
    ) / np.float32(Lbox**3)

    # disconnected covariance
    if want_rsd:
        cov_zn = np.stack([multipole_cov(pk_zn_proj, ell) for ell in poles])
        var_zz = np.stack([multipole_cov(pk_zz_proj, ell) for ell in poles])
        var_nn = np.stack([multipole_cov(pk_nn_proj, ell) for ell in poles])
    else:
        cov_zn = 2 * pk_zn_proj**2
        var_zz = 2 * pk_zz_proj**2
        var_nn = 2 * pk_nn_proj**2

    # cross-correlation coefficient
    with np.errstate(divide='ignore'):
        r_zt_proj = cov_zn / np.sqrt(var_zz * var_nn)
        # r_zt_proj[np.isclose(var_zz * var_nn, 0.)] = 0.
        r_zt_proj = np.atleast_2d(r_zt_proj)

    # beta parameter
    with np.errstate(divide='ignore'):
        beta_proj = cov_zn / var_zz
        # beta_proj[np.isclose(var_zz, 0.)] = 0.
    beta_damp = 0.5 * (1 - np.tanh((k_binc - k0) / dk_cv)) * beta_proj
    beta_damp = np.atleast_2d(beta_damp)
    beta_damp[:, : k_binc.searchsorted(beta1_k)] = 1.0
    beta_smooth = np.zeros_like(beta_damp)
    for i in range(beta_smooth.shape[0]):
        beta_smooth[i, :] = savgol_filter(beta_damp.T[:, i], sg_window, 3)
    beta_smooth = expand_poles_to_3d(k_binc, beta_smooth, nmesh, Lbox, np.array([0]))

    # get reduced fields
    pk_nn -= beta_smooth * pk_zz
    del beta_smooth
    gc.collect()
    del pk_zz
    gc.collect()

    # save CV-reduced 3D power spectrum
    pk_tr_dict = {}
    pk_tr_dict['P_k3D_tr_tr_zcv'] = pk_nn
    header = {}
    header['sim_name'] = sim_name
    header['Lbox'] = Lbox
    header['nmesh'] = nmesh
    header['kcut'] = kcut
    compress_asdf(str(power_cv_tr_fn), pk_tr_dict, header)
    print('Compressed')

    # project to multipoles
    pk_nn_betasmooth, nmodes = project_3d_to_poles(k_bins, pk_nn, Lbox, poles)
    pk_nn = pk_nn_proj.reshape(len(poles), len(k_binc))
    pk_zz = pk_zz_proj
    pk_zn = pk_zn_proj
    r_zt = r_zt_proj

    # changing array shapes (note that projection multiplies by L^3, so we get rid of that)
    pk_nn_betasmooth = pk_nn_betasmooth.reshape(len(poles), len(k_binc)) / Lbox**3
    pk_zenbu = pk_zenbu.reshape(len(poles), len(k_binc)) / Lbox**3
    nmodes = nmodes.flatten()[: len(k_binc)]

    # save result
    zcv_dict = {}
    zcv_dict['k_binc'] = k_binc
    zcv_dict['poles'] = poles
    zcv_dict['rho_tr_ZD'] = r_zt
    zcv_dict['Pk_ZD_ZD_ell'] = pk_zz * Lbox**3
    zcv_dict['Pk_tr_ZD_ell'] = pk_zn * Lbox**3
    zcv_dict['Pk_tr_tr_ell'] = pk_nn * Lbox**3
    zcv_dict['Nk_tr_tr_ell'] = nmodes
    zcv_dict['Pk_tr_tr_ell_zcv'] = pk_nn_betasmooth * Lbox**3
    zcv_dict['Pk_ZD_ZD_ell_ZeNBu'] = pk_zenbu * Lbox**3
    zcv_dict['bias'] = bias_vec[1:]
    return zcv_dict


def run_lcv(power_rsd_tr_dict, power_lin_dict, config):
    """
    Apply Linear control variates (LCV) reduction to some measured power spectrum.
    """
    # read out some parameters from the config function
    sim_name = config['sim_params']['sim_name']
    z_this = config['sim_params']['z_mock']
    lcv_dir = config['lcv_params']['lcv_dir']
    nmesh = config['lcv_params']['nmesh']
    kcut = config['lcv_params']['kcut']
    kmax = config['lcv_params'].get('kmax_fit', 0.08)
    want_rsd = config['HOD_params']['want_rsd']
    assert nmesh == config['power_params']['nmesh'], (
        'nmesh from power_params need to match nmesh from lcv_params'
    )

    # smoothing parameters
    sg_window = config['lcv_params'].get('sg_window', 21)
    k0 = config['lcv_params'].get('k0_window', 0.618)
    dk_cv = config['lcv_params'].get('dk_window', 0.167)
    beta1_k = config['lcv_params'].get('beta1_k', 0.05)

    # power params
    k_hMpc_max = config['power_params']['k_hMpc_max']
    logk = config['power_params']['logk']
    n_k_bins = config['power_params']['nbins_k']
    n_mu_bins = config['power_params']['nbins_mu']
    poles = config['power_params']['poles']

    # reconstruction algorithm
    rec_algo = config['HOD_params']['rec_algo']
    if rec_algo == 'recsym':
        R = None
    elif rec_algo == 'reciso':
        R = config['HOD_params']['smoothing']

    # create save directory
    save_dir = Path(lcv_dir) / sim_name
    save_dir / f'z{z_this:.3f}'

    # read meta data
    meta = get_meta(sim_name, redshift=z_this)
    Lbox = meta['BoxSize']
    z_ic = meta['InitialRedshift']
    D_ratio = meta['GrowthTable'][z_ic] / meta['GrowthTable'][1.0]
    np.pi * nmesh / Lbox
    cosmo = {}
    cosmo['output'] = 'mPk mTk'
    cosmo['P_k_max_h/Mpc'] = 20.0
    int(sim_name.split('ph')[-1])
    for k in (
        'H0',
        'omega_b',
        'omega_cdm',
        'omega_ncdm',
        'N_ncdm',
        'N_ur',
        'n_s',
        'A_s',
        'alpha_s',
        #'wa', 'w0',
    ):
        cosmo[k] = meta[k]

    # load input linear power
    kth = meta['CLASS_power_spectrum']['k (h/Mpc)']
    pk_z1 = meta['CLASS_power_spectrum']['P (Mpc/h)^3']

    # rewind back to initial redshift of the simulation
    p_m_lin = D_ratio**2 * pk_z1

    # apply gaussian cutoff to linear power
    p_m_lin *= np.exp(-((kth / kcut) ** 2))

    # compute growth factor
    pkclass = Class()
    pkclass.set(cosmo)
    pkclass.compute()
    D = pkclass.scale_independent_growth_factor(z_this)
    D /= pkclass.scale_independent_growth_factor(z_ic)
    pkclass.Hubble(z_this) * 299792.458
    if want_rsd:
        f_growth = pkclass.scale_independent_growth_factor_f(z_this)
    else:
        f_growth = 0.0

    # define k bins
    k_bins, mu_bins = get_k_mu_edges(Lbox, k_hMpc_max, n_k_bins, n_mu_bins, logk)
    k_binc = (k_bins[1:] + k_bins[:-1]) * 0.5
    if not logk:
        dk = k_bins[1] - k_bins[0]
        assert np.isclose(dk, k_bins[-1] - k_bins[-2]), (
            'Spacing between k_bins is uneven'
        )
    else:
        dk = np.log(k_bins[1] / k_bins[0])
        assert np.isclose(dk, np.log(k_bins[-1] / k_bins[-2])), (
            'Spacing between k_bins is uneven'
        )

    # name of files to read from
    if n_k_bins == nmesh // 2:
        window_fn = save_dir / f'window_nmesh{nmesh:d}.npz'
    else:
        window_fn = save_dir / f'window_nmesh{nmesh:d}_dk{dk:.3f}.npz'

    # get the bias
    bvec_opt = measure_2pt_bias_lcv(
        k_binc,
        power_lin_dict,
        power_rsd_tr_dict,
        D,
        f_growth,
        kmax,
        want_rsd,
        rec_algo,
        R,
        ellmax=1,
    )
    bias = np.array(bvec_opt['x'])[0]
    print('bias', bias)

    # get linear prediction for the multipoles
    if rec_algo == 'reciso':
        S = np.exp(-(kth**2) * R**2 / 2.0)
        f_eff = f_growth * (1.0 - S)
    elif rec_algo == 'recsym':
        f_eff = f_growth
    kth, p_m_lin_poles = get_poles(kth, p_m_lin, D, bias, f_eff, poles=poles)

    # interpolate linear prediction at the k's of interest
    assert want_rsd, 'Real space not implemented'
    p_m_lin_input = []
    for i in range(len(poles)):
        p_m_lin_input.append(
            interp1d(kth, p_m_lin_poles[i], fill_value='extrapolate')(k_binc)
        )
    p_m_lin_input = np.array(p_m_lin_input)

    # convert into kaiser-corrected power spectra
    pk_ll_input = combine_kaiser_spectra(
        k_binc, power_lin_dict, D, bias, f_growth, rec_algo, R, rsd=want_rsd
    ).reshape(len(poles), len(k_binc))
    pk_tl_input = combine_cross_kaiser_spectra(
        k_binc, power_rsd_tr_dict, D, bias, f_growth, rec_algo, R, rsd=want_rsd
    ).reshape(len(poles), len(k_binc))
    pk_tt_input = power_rsd_tr_dict['P_ell_tr_tr'].reshape(len(poles), len(k_binc))
    nmodes = power_rsd_tr_dict['N_ell_tr_tr'].flatten()

    # load the presaved window function
    data = np.load(window_fn)
    window = data['window']
    keff = data['keff']
    assert len(keff) == len(k_binc), f'Mismatching file: {str(window_fn)}'
    assert np.abs(keff[-1] - k_binc[-1]) / k_binc[-1] < 0.1, (
        f'Mismatching file: {str(window_fn)}'
    )

    # stochasticity and model error
    shotnoise = (pk_tt_input - 2.0 * pk_tl_input + pk_ll_input)[0]
    pk_tt_nosn = pk_tt_input.copy()
    pk_tt_nosn[0] -= shotnoise  # subtracting only from ell = 0

    # disconnected covariance
    if want_rsd:
        cov_tl = np.stack([multipole_cov(pk_tl_input, ell) for ell in poles])
        var_ll = np.stack([multipole_cov(pk_ll_input, ell) for ell in poles])
        var_tt = np.stack([multipole_cov(pk_tt_input, ell) for ell in poles])
        var_tt_nosn = np.stack([multipole_cov(pk_tt_nosn, ell) for ell in poles])
    else:
        cov_tl = 2 * pk_tl_input**2
        var_ll = 2 * pk_ll_input**2
        var_tt = 2 * pk_tt_input**2
        var_tt_nosn = 2.0 * (pk_tt_input - shotnoise[0]) ** 2

    # cross-correlation coefficient
    with np.errstate(divide='ignore'):
        r_tl = cov_tl / np.sqrt(var_ll * var_tt)
        # r_tl[np.isclose(var_ll * var_tt, 0.)] = 0.
        r_tl = np.atleast_2d(r_tl)
        r_tl[r_tl != r_tl] = 0  # takes care of NaN's

    # shotnoise limit
    with np.errstate(divide='ignore'):
        r_tl_sn_lim = var_tt_nosn / np.sqrt(var_tt * var_tt_nosn)
        # r_tl_sn_lim[np.isclose(var_tt * var_tt_nosn, 0.)] = 0.

    # beta parameter
    with np.errstate(divide='ignore'):
        beta = cov_tl / var_ll
        # beta[np.isclose(var_ll), 0.] = 0.
    beta_damp = 0.5 * (1 - np.tanh((k_binc - k0) / dk_cv)) * beta
    beta_damp = np.atleast_2d(beta_damp)
    beta_damp[beta_damp != beta_damp] = 0  # takes care of NaN's
    beta_damp[:, : k_binc.searchsorted(beta1_k)] = 1
    beta_smooth = np.zeros_like(beta_damp)
    for i in range(beta_smooth.shape[0]):
        try:
            beta_smooth[i, :] = savgol_filter(beta_damp.T[:, i], sg_window, 3)
        except ValueError:
            warnings.warn('This message should only appear when doing a smoke test.')

    # apply window function
    if want_rsd:
        p_m_lin = np.hstack(p_m_lin_input)
        p_m_lin = np.dot(window.T, p_m_lin).reshape(len(poles), -1)

    # beta needs to be smooth for best results
    pk_tt_betasmooth = pk_tt_input - beta_smooth * (pk_ll_input - p_m_lin)

    # save output in a dictionary
    lcv_dict = {}
    lcv_dict['k_binc'] = k_binc
    lcv_dict['poles'] = poles
    lcv_dict['rho_tr_lf'] = r_tl
    lcv_dict['rho_tr_lf_sn_lim'] = r_tl_sn_lim
    lcv_dict['Pk_lf_lf_ell'] = pk_ll_input
    lcv_dict['Pk_tr_lf_ell'] = pk_tl_input
    lcv_dict['Pk_tr_tr_ell'] = pk_tt_input
    lcv_dict['Nk_tr_tr_ell'] = nmodes
    lcv_dict['Pk_tr_tr_ell_lcv'] = pk_tt_betasmooth
    lcv_dict['Pk_lf_lf_ell_CLASS'] = p_m_lin_input
    lcv_dict['bias'] = bias
    return lcv_dict


def run_lcv_field(power_rsd_tr_fns, power_lin_fns, config):
    """
    Apply Linear control variates (LCV) reduction to some measured 3D power spectrum.
    """
    # read out some parameters from the config function
    sim_name = config['sim_params']['sim_name']
    z_this = config['sim_params']['z_mock']
    lcv_dir = config['lcv_params']['lcv_dir']
    nmesh = config['lcv_params']['nmesh']
    kcut = config['lcv_params']['kcut']
    kmax = config['lcv_params'].get('kmax_fit', 0.08)
    want_rsd = config['HOD_params']['want_rsd']
    rsd_str = '_rsd' if want_rsd else ''
    keynames = ['delta', 'deltamu2']
    assert nmesh == config['power_params']['nmesh'], (
        'nmesh from power_params need to match nmesh from lcv_params'
    )

    # smoothing parameters
    sg_window = config['lcv_params'].get('sg_window', 21)
    k0 = config['lcv_params'].get('k0_window', 0.618)
    dk_cv = config['lcv_params'].get('dk_window', 0.167)
    beta1_k = config['lcv_params'].get('beta1_k', 0.05)

    # read meta data
    meta = get_meta(sim_name, redshift=z_this)
    Lbox = meta['BoxSize']
    z_ic = meta['InitialRedshift']
    D_ratio = meta['GrowthTable'][z_ic] / meta['GrowthTable'][1.0]
    cosmo = {}
    cosmo['output'] = 'mPk mTk'
    cosmo['P_k_max_h/Mpc'] = 20.0
    int(sim_name.split('ph')[-1])
    for k in (
        'H0',
        'omega_b',
        'omega_cdm',
        'omega_ncdm',
        'N_ncdm',
        'N_ur',
        'n_s',
        'A_s',
        'alpha_s',
        #'wa', 'w0',
    ):
        cosmo[k] = meta[k]

    # power params
    k_hMpc_max = config['power_params'].get('k_hMpc_max', np.pi * nmesh / Lbox)
    logk = config['power_params'].get('logk', False)
    n_k_bins = config['power_params'].get('nbins_k', nmesh // 2)
    n_mu_bins = config['power_params'].get('nbins_mu', 1)
    poles = config['power_params']['poles']

    # create save directory
    save_dir = Path(lcv_dir) / sim_name
    save_z_dir = save_dir / f'z{z_this:.3f}'

    # make sure that the parameters are set correctly
    if not (
        np.isclose(k_hMpc_max, np.pi * nmesh / Lbox) & logk
        == False & n_k_bins
        == nmesh // 2 & n_mu_bins
        == 1
    ):
        warnings.warn('Setting the parameters correctly for Xi computation')
        k_hMpc_max = np.pi * nmesh / Lbox
        logk = False
        n_k_bins = nmesh // 2
        n_mu_bins = 1

    # reconstruction algorithm
    rec_algo = config['HOD_params']['rec_algo']
    if rec_algo == 'recsym':
        R = None
    elif rec_algo == 'reciso':
        R = config['HOD_params']['smoothing']

    # create save directory
    save_dir = Path(lcv_dir) / sim_name
    save_z_dir = save_dir / f'z{z_this:.3f}'

    # load input linear power
    kth = meta['CLASS_power_spectrum']['k (h/Mpc)']
    pk_z1 = meta['CLASS_power_spectrum']['P (Mpc/h)^3']

    # make the k-values equidistant
    choice = kth < np.sqrt(3.0) * 1.2 * np.pi * nmesh / Lbox  # sqrt(3)*1.2*Nyquist
    kth, pk_z1 = kth[choice], pk_z1[choice]
    kth_new = np.arange(kth.min(), kth.max(), np.min(np.diff(kth)))
    pk_z1_new = np.interp(kth_new, kth, pk_z1)
    kth, pk_z1 = kth_new, pk_z1_new

    # rewind back to initial redshift of the simulation
    p_m_lin = D_ratio**2 * pk_z1

    # apply gaussian cutoff to linear power
    p_m_lin *= np.exp(-((kth / kcut) ** 2))

    # compute growth factor
    pkclass = Class()
    pkclass.set(cosmo)
    pkclass.compute()
    D = pkclass.scale_independent_growth_factor(z_this)
    D /= pkclass.scale_independent_growth_factor(z_ic)
    pkclass.Hubble(z_this) * 299792.458
    if want_rsd:
        f_growth = pkclass.scale_independent_growth_factor_f(z_this)
    else:
        f_growth = 0.0
    print('D, f = ', D, f_growth)

    # define k bins
    k_bins, mu_bins = get_k_mu_edges(Lbox, k_hMpc_max, n_k_bins, n_mu_bins, logk)
    k_binc = (k_bins[1:] + k_bins[:-1]) * 0.5
    if not logk:
        k_bins[1] - k_bins[0]
    else:
        np.log(k_bins[1] / k_bins[0])

    # file to save to
    power_cv_tr_fn = (
        Path(save_z_dir) / f'power{rsd_str}_LCV_tr_{rec_algo}_nmesh{nmesh:d}.asdf'
    )

    # compute bias from the monopole
    pk_tt = asdf.open(power_rsd_tr_fns[0])['data']['P_k3D_tr_tr']
    pk_tt = project_3d_to_poles(k_bins, pk_tt, Lbox, poles=[0])[0].flatten() / Lbox**3
    pk_ij = {}
    counter = 0
    for i in range(len(keynames)):
        for j in range(len(keynames)):
            if i < j:
                continue
            print('Projecting', i, j)
            pk = asdf.open(power_lin_fns[counter])['data'][
                f'P_k3D_{keynames[i]}_{keynames[j]}'
            ]
            pk = project_3d_to_poles(k_bins, pk, Lbox, poles=[0])
            pk_ij[f'P_ell_{keynames[i]}_{keynames[j]}'] = (
                pk[0].flatten() / Lbox**3
            ).reshape(1, len(pk_tt), 1)
            nmodes = pk[1].flatten()
            counter += 1

    # get the bias
    bvec_opt = measure_2pt_bias_lcv(
        k_binc,
        pk_ij,
        {'P_ell_tr_tr': pk_tt.reshape(1, len(pk_tt), 1)},
        D,
        f_growth,
        kmax,
        want_rsd,
        rec_algo,
        R,
        ellmax=1,
    )
    bias = np.array(bvec_opt['x'])[0]
    print('bias', bias)

    # get linear prediction
    if rec_algo == 'reciso':
        S = np.exp(-(kth**2) * R**2 / 2.0)
        f_eff = f_growth * (1.0 - S)
    elif rec_algo == 'recsym':
        f_eff = f_growth
    kth, p_m_lin_poles = get_poles(kth, p_m_lin, D, bias, f_eff, poles=poles)
    assert want_rsd, 'Real space not implemented'

    # calculate the 3d power spectra
    pk_tt, pk_ll, pk_lt = combine_field_spectra_k3D_lcv(
        bias, f_growth, D, power_lin_fns, power_rsd_tr_fns, nmesh, Lbox, R, rec_algo
    )

    # project 3D power spectra to multipoles
    pk_lt_proj = (
        project_3d_to_poles(k_bins, pk_lt, Lbox, poles)[0].reshape(
            len(poles), len(k_binc)
        )
        / Lbox**3
    )
    del pk_lt
    gc.collect()
    pk_tt_proj = (
        project_3d_to_poles(k_bins, pk_tt, Lbox, poles)[0].reshape(
            len(poles), len(k_binc)
        )
        / Lbox**3
    )
    pk_ll_proj = (
        project_3d_to_poles(k_bins, pk_ll, Lbox, poles)[0].reshape(
            len(poles), len(k_binc)
        )
        / Lbox**3
    )

    # expand multipole to 3D power spectra (this is the C-mu_C part)
    assert np.isclose(np.min(np.diff(kth)), np.max(np.diff(kth))), (
        'For custom interpolation, need equidistant k-values'
    )
    pk_ll[:, :, :] -= expand_poles_to_3d(
        kth, p_m_lin_poles, nmesh, Lbox, np.asarray(poles)
    ) / np.float32(Lbox**3)
    gc.collect()

    # disconnected covariance
    if want_rsd:
        cov_lt = np.stack([multipole_cov(pk_lt_proj, ell) for ell in poles])
        var_ll = np.stack([multipole_cov(pk_ll_proj, ell) for ell in poles])
        var_tt = np.stack([multipole_cov(pk_tt_proj, ell) for ell in poles])
    else:
        cov_lt = 2 * pk_lt_proj**2
        var_ll = 2 * pk_ll_proj**2
        var_tt = 2 * pk_tt_proj**2

    # beta parameter
    with np.errstate(divide='ignore'):
        beta_proj = cov_lt / var_ll
        # beta_proj[np.isclose(var_ll, 0.)] = 0.
    beta_damp = 1 / 2 * (1 - np.tanh((k_binc - k0) / dk_cv)) * beta_proj
    beta_damp = np.atleast_2d(beta_damp)
    beta_damp[:, : k_binc.searchsorted(beta1_k)] = 1.0
    beta_smooth = np.zeros_like(beta_damp)
    for i in range(beta_smooth.shape[0]):
        beta_smooth[i, :] = savgol_filter(beta_damp.T[:, i], sg_window, 3)
    beta_smooth = expand_poles_to_3d(k_binc, beta_smooth, nmesh, Lbox, np.array([0]))

    # cross-correlation coefficient
    with np.errstate(divide='ignore'):
        r_lt_proj = cov_lt / np.sqrt(var_ll * var_tt)
        # r_lt_proj[np.isclose(var_ll * var_tt, 0.)] = 0.
        r_lt_proj = np.atleast_2d(r_lt_proj)

    # get reduced fields
    pk_tt -= beta_smooth * pk_ll
    del beta_smooth
    gc.collect()
    del pk_ll
    gc.collect()

    # save 3d
    pk_tr_dict = {}
    pk_tr_dict['P_k3D_tr_tr_lcv'] = pk_tt
    header = {}
    header['sim_name'] = sim_name
    header['Lbox'] = Lbox
    header['nmesh'] = nmesh
    header['kcut'] = kcut
    compress_asdf(str(power_cv_tr_fn), pk_tr_dict, header)
    print('Compressed')

    # project to multipoles
    pk_tt_betasmooth, nmodes = project_3d_to_poles(k_bins, pk_tt, Lbox, poles)

    # changing format (note that projection multiplies by L^3, so we get rid of that)
    pk_tt_betasmooth = pk_tt_betasmooth.reshape(len(poles), len(k_binc)) / Lbox**3
    nmodes = nmodes.flatten()[: len(k_binc)]

    # interpolate
    p_m_lin_input = np.zeros((len(poles), len(k_binc)))
    for i in range(len(poles)):
        p_m_lin_input[i] = (
            interp1d(kth, p_m_lin_poles[i], fill_value='extrapolate')(k_binc)
        ) / Lbox**3

    # save output in a dictionary
    lcv_dict = {}
    lcv_dict['k_binc'] = k_binc
    lcv_dict['poles'] = poles
    lcv_dict['rho_tr_lf'] = r_lt_proj
    lcv_dict['Pk_lf_lf_ell'] = pk_ll_proj * Lbox**3
    lcv_dict['Pk_tr_lf_ell'] = pk_lt_proj * Lbox**3
    lcv_dict['Pk_tr_tr_ell'] = pk_tt_proj * Lbox**3
    lcv_dict['Nk_tr_tr_ell'] = nmodes
    lcv_dict['Pk_tr_tr_ell_lcv'] = pk_tt_betasmooth * Lbox**3
    lcv_dict['Pk_lf_lf_ell_CLASS'] = p_m_lin_input * Lbox**3
    lcv_dict['bias'] = bias
    return lcv_dict
