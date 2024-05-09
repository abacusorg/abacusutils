import os
import gc
from pathlib import Path

import asdf
import numpy as np
from scipy.fft import rfftn

from abacusnbody.analysis.power_spectrum import (
    calc_pk_from_deltak,
    get_field_fft,
    get_k_mu_edges,
    get_delta_mu2,
    get_W_compensated,
)
from abacusnbody.metadata import get_meta

from .ic_fields import compress_asdf

try:
    from classy import Class
except ImportError as e:
    raise ImportError(
        'Could not import classy. Install abacusutils with '
        '"pip install abacusutils[all]" to install zcv dependencies.'
    ) from e


def get_tracer_power(tracer_pos, want_rsd, config, want_save=True, save_3D_power=False):
    """
    Compute the auto- and cross-correlation between a galaxy catalog (`tracer_pos`)
    and the advected Zel'dovich fields.

    Parameters
    ----------
    tracer_pos : array_like
        galaxy positions with shape (N, 3)
    want_rsd : bool
        compute the power spectra in redshift space?
    config : str
        name of the yaml containing parameter specifications.
    save_3D_power : bool, optional
        save the 3D power spectra in individual ASDF files.
        Default is False.

    Returns
    -------
    pk_tr_dict : dict
        dictionary containing the auto- and cross-power spectra of
        the tracer with the 5 fields.
    """
    # read zcv parameters
    advected_dir = config['zcv_params']['zcv_dir']  # input of advected fields
    tracer_dir = config['zcv_params']['zcv_dir']  # output of tracers
    # ic_dir = config['zcv_params']['ic_dir']
    nmesh = config['zcv_params']['nmesh']
    kcut = config['zcv_params']['kcut']
    keynames = config['zcv_params']['fields']

    # power params
    sim_name = config['sim_params']['sim_name']
    z_this = config['sim_params']['z_mock']
    k_hMpc_max = config['power_params']['k_hMpc_max']
    logk = config['power_params']['logk']
    n_k_bins = config['power_params']['nbins_k']
    n_mu_bins = config['power_params']['nbins_mu']
    poles = config['power_params']['poles']
    paste = config['power_params']['paste']
    compensated = config['power_params']['compensated']
    interlaced = config['power_params']['interlaced']
    # want_rsd = config['HOD_params']['want_rsd']
    rsd_str = '_rsd' if want_rsd else ''

    # get a few parameters for the simulation
    meta = get_meta(sim_name, redshift=z_this)
    Lbox = meta['BoxSize']
    z_ic = meta['InitialRedshift']
    # k_Ny = np.pi*nmesh/Lbox

    # define k, mu bins
    k_bin_edges, mu_bin_edges = get_k_mu_edges(
        Lbox, k_hMpc_max, n_k_bins, n_mu_bins, logk
    )
    k_binc = (k_bin_edges[1:] + k_bin_edges[:-1]) * 0.5
    mu_binc = (mu_bin_edges[1:] + mu_bin_edges[:-1]) * 0.5

    # start dictionary with power spectra
    pk_tr_dict = {}
    pk_tr_dict['k_binc'] = k_binc
    pk_tr_dict['mu_binc'] = mu_binc

    # create save directory
    save_dir = Path(tracer_dir) / sim_name
    save_z_dir = save_dir / f'z{z_this:.3f}'
    save_z_dir.mkdir(exist_ok=True, parents=True)

    # get path to input directory
    advected_dir_z_dir = Path(advected_dir) / sim_name / f'z{z_this:.3f}'

    # get the window function of TSC/CIC
    if compensated:
        W = get_W_compensated(Lbox, nmesh, paste, interlaced)
    else:
        W = None

    # set up cosmology
    boltz = Class()
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
    boltz.set(cosmo)
    boltz.compute()

    # file to save to
    fields_fft_fn = []
    for i in range(len(keynames)):
        fields_fft_fn.append(
            advected_dir_z_dir
            / f'advected_{keynames[i]}_field{rsd_str}_fft_nmesh{nmesh:d}.asdf'
        )
    tr_field_fft_fn = Path(save_z_dir) / f'tr_field{rsd_str}_fft_nmesh{nmesh:d}.asdf'
    if not logk:
        dk = k_bin_edges[1] - k_bin_edges[0]
    else:
        dk = np.log(k_bin_edges[1] / k_bin_edges[0])
    if n_k_bins == nmesh // 2:
        power_tr_fn = Path(save_z_dir) / f'power{rsd_str}_tr_nmesh{nmesh:d}.asdf'
    else:
        power_tr_fn = (
            Path(save_z_dir) / f'power{rsd_str}_tr_nmesh{nmesh:d}_dk{dk:.3f}.asdf'
        )

    # compute growth factor
    D = boltz.scale_independent_growth_factor(z_this)
    D /= boltz.scale_independent_growth_factor(z_ic)
    print('D = ', D)

    # field names and growths
    field_D = [1, D, D**2, D**2, D]

    # create fft field for the tracer
    w = None
    tracer_pos += Lbox / 2.0  # I think necessary for cross correlations
    tracer_pos %= Lbox
    print('min/max tracer pos', tracer_pos.min(), tracer_pos.max(), tracer_pos.shape)
    tr_field_fft = get_field_fft(
        tracer_pos, Lbox, nmesh, paste, w, W, compensated, interlaced
    )
    del tracer_pos
    gc.collect()

    if want_save:
        # save the tracer field
        header = {}
        header['sim_name'] = sim_name
        header['Lbox'] = Lbox
        header['nmesh'] = nmesh
        header['compensated'] = compensated
        header['interlaced'] = interlaced
        header['paste'] = paste
        table = {}
        table['tr_field_fft_Re'] = np.array(tr_field_fft.real, dtype=np.float32)
        table['tr_field_fft_Im'] = np.array(tr_field_fft.imag, dtype=np.float32)
        compress_asdf(tr_field_fft_fn, table, header)
        del table
        gc.collect()

    print('Computing auto-correlation of tracer')
    if save_3D_power:
        power_tr_fns = []

        # compute
        pk3d = np.array((tr_field_fft * np.conj(tr_field_fft)).real, dtype=np.float32)
        # pk3d *= Lbox**3

        # record
        pk_tr_dict = {}
        pk_tr_dict['P_k3D_tr_tr'] = pk3d
        header = {}
        header['sim_name'] = sim_name
        header['Lbox'] = Lbox
        header['nmesh'] = nmesh
        header['kcut'] = kcut
        power_tr_fn = Path(save_z_dir) / f'power{rsd_str}_tr_tr_nmesh{nmesh:d}.asdf'
        power_tr_fns.append(power_tr_fn)
        compress_asdf(str(power_tr_fn), pk_tr_dict, header)
    else:
        # get the box k and mu modes
        k_bin_edges, mu_bin_edges = get_k_mu_edges(
            Lbox, k_hMpc_max, n_k_bins, n_mu_bins, logk
        )

        # compute the galaxy auto rsd poles
        P = calc_pk_from_deltak(
            tr_field_fft,
            Lbox,
            k_bin_edges,
            mu_bin_edges,
            field2_fft=None,
            poles=np.asarray(poles),
        )
        pk_tr_dict['P_kmu_tr_tr'] = P['power']
        pk_tr_dict['N_kmu_tr_tr'] = P['N_mode']
        pk_tr_dict['P_ell_tr_tr'] = P['binned_poles']
        pk_tr_dict['N_ell_tr_tr'] = P['N_mode_poles']

    # loop over fields
    for i in range(len(keynames)):
        print('Computing cross-correlation of tracer and ', keynames[i])

        # load field
        field_fft_i = asdf.open(fields_fft_fn[i])['data']

        if save_3D_power:
            # compute
            field_fft_i = (
                field_fft_i[f'{keynames[i]}_Re'] + 1j * field_fft_i[f'{keynames[i]}_Im']
            )
            pk3d = np.array(
                (field_fft_i * np.conj(tr_field_fft)).real, dtype=np.float32
            )
            pk3d *= field_D[i]
            # pk3d *= Lbox**3

            # record
            pk_tr_dict = {}
            pk_tr_dict[f'P_k3D_{keynames[i]}_tr'] = pk3d
            header = {}
            header['sim_name'] = sim_name
            header['Lbox'] = Lbox
            header['nmesh'] = nmesh
            header['kcut'] = kcut
            power_tr_fn = (
                Path(save_z_dir)
                / f'power{rsd_str}_{keynames[i]}_tr_nmesh{nmesh:d}.asdf'
            )
            power_tr_fns.append(power_tr_fn)
            compress_asdf(str(power_tr_fn), pk_tr_dict, header)
            del field_fft_i
            gc.collect()

        else:
            # compute power spectrum
            P = calc_pk_from_deltak(
                field_fft_i[f'{keynames[i]}_Re']
                + 1j * field_fft_i[f'{keynames[i]}_Im'],
                Lbox,
                k_bin_edges,
                mu_bin_edges,
                field2_fft=tr_field_fft,
                poles=np.asarray(poles),
            )
            P['power'] *= field_D[i]
            P['binned_poles'] *= field_D[i]
            pk_tr_dict[f'P_kmu_{keynames[i]}_tr'] = P['power']
            pk_tr_dict[f'N_kmu_{keynames[i]}_tr'] = P['N_mode']
            pk_tr_dict[f'P_ell_{keynames[i]}_tr'] = P['binned_poles']
            pk_tr_dict[f'N_ell_{keynames[i]}_tr'] = P['N_mode_poles']
            del field_fft_i
            gc.collect()

    if save_3D_power:
        return power_tr_fns

    header = {}
    header['sim_name'] = sim_name
    header['Lbox'] = Lbox
    header['nmesh'] = nmesh
    header['kcut'] = kcut
    if want_save:
        compress_asdf(str(power_tr_fn), pk_tr_dict, header)
    return pk_tr_dict


def get_recon_power(
    tracer_pos,
    random_pos,
    want_rsd,
    config,
    want_save=True,
    save_3D_power=False,
    want_load_tr_fft=False,
):
    """
    Compute the auto- and cross-correlation between a galaxy catalog (`tracer_pos`)
    and the advected initial conditions fields (delta, delta*mu^2).

    Parameters
    ----------
    tracer_pos : array_like
        galaxy positions with shape (N, 3)
    random_pos : array_like
        randoms positions with shape (M, 3)
    want_rsd : bool
        compute the power spectra in redshift space?
    config : str
        name of the yaml containing parameter specifications.
    save_3D_power : bool, optional
        save the 3D power spectra in individual ASDF files.
        Default is False.
    want_load_tr_fft : bool, optional
        want to load provided 3D Fourier tracer field? Default is False.

    Returns
    -------
    pk_tr_dict : dict
        dictionary containing the auto- and cross-power spectra of
        the tracer with the 2 fields.
    """
    # field names
    keynames = ['delta', 'deltamu2']

    # read lcv parameters
    lcv_dir = config['lcv_params']['lcv_dir']
    config['lcv_params']['ic_dir']
    nmesh = config['lcv_params']['nmesh']
    kcut = config['lcv_params']['kcut']
    rec_algo = config['HOD_params']['rec_algo']

    # power params
    sim_name = config['sim_params']['sim_name']
    z_this = config['sim_params']['z_mock']
    k_hMpc_max = config['power_params']['k_hMpc_max']
    logk = config['power_params']['logk']
    n_k_bins = config['power_params']['nbins_k']
    n_mu_bins = config['power_params']['nbins_mu']
    poles = config['power_params']['poles']
    paste = config['power_params']['paste']
    compensated = config['power_params']['compensated']
    interlaced = config['power_params']['interlaced']
    # want_rsd = config['HOD_params']['want_rsd']
    rsd_str = '_rsd' if want_rsd else ''

    # get a few parameters for the simulation
    meta = get_meta(sim_name, redshift=z_this)
    Lbox = meta['BoxSize']
    meta['InitialRedshift']
    np.pi * nmesh / Lbox

    # define k, mu bins
    k_bin_edges, mu_bin_edges = get_k_mu_edges(
        Lbox, k_hMpc_max, n_k_bins, n_mu_bins, logk
    )
    k_binc = (k_bin_edges[1:] + k_bin_edges[:-1]) * 0.5
    mu_binc = (mu_bin_edges[1:] + mu_bin_edges[:-1]) * 0.5

    # start dictionary with power spectra
    pk_tr_dict = {}
    pk_tr_dict['k_binc'] = k_binc
    pk_tr_dict['mu_binc'] = mu_binc

    # create save directory
    save_dir = Path(lcv_dir) / sim_name
    save_z_dir = save_dir / f'z{z_this:.3f}'
    os.makedirs(save_z_dir, exist_ok=True)

    # get the window function of TSC/CIC
    if compensated:
        W = get_W_compensated(Lbox, nmesh, paste, interlaced)
    else:
        W = None

    # file to save to
    ic_fn = Path(save_dir) / f'ic_filt_nmesh{nmesh:d}.asdf'
    tr_field_fft_fn = (
        Path(save_z_dir) / f'tr_field{rsd_str}_fft_nmesh{nmesh:d}.asdf'
    )  # overwrites
    if not logk:
        dk = k_bin_edges[1] - k_bin_edges[0]
    else:
        dk = np.log(k_bin_edges[1] / k_bin_edges[0])
    if n_k_bins == nmesh // 2:
        power_tr_fn = (
            Path(save_z_dir) / f'power{rsd_str}_tr_{rec_algo}_lin_nmesh{nmesh:d}.asdf'
        )
    else:
        power_tr_fn = (
            Path(save_z_dir)
            / f'power{rsd_str}_tr_{rec_algo}_lin_nmesh{nmesh:d}_dk{dk:.3f}.asdf'
        )

    # create fft field for the tracer
    if want_load_tr_fft:
        tr_field_fft = (
            asdf.open(tr_field_fft_fn)['data']['tr_field_fft_Re']
            + 1j * asdf.open(tr_field_fft_fn)['data']['tr_field_fft_Im']
        )
    else:
        w = None
        print(
            'min/max tracer pos', tracer_pos.min(), tracer_pos.max(), tracer_pos.shape
        )
        tr_field_fft = get_field_fft(
            tracer_pos, Lbox, nmesh, paste, w, W, compensated, interlaced
        )
        if random_pos is not None:
            rn_field_fft = get_field_fft(
                random_pos, Lbox, nmesh, paste, w, W, compensated, interlaced
            )
            tr_field_fft -= rn_field_fft
            del random_pos, rn_field_fft
            gc.collect()
        del tracer_pos
        gc.collect()

        if want_save:
            # save the tracer field
            header = {}
            header['sim_name'] = sim_name
            header['Lbox'] = Lbox
            header['nmesh'] = nmesh
            header['compensated'] = compensated
            header['interlaced'] = interlaced
            header['paste'] = paste
            table = {}
            table['tr_field_fft_Re'] = np.array(tr_field_fft.real, dtype=np.float32)
            table['tr_field_fft_Im'] = np.array(tr_field_fft.imag, dtype=np.float32)
            compress_asdf(tr_field_fft_fn, table, header)
            del table
            gc.collect()

    # You need to call this function twice because large files -- once to compute the tr_field_fft and save it and once to just load it and compute stuff
    if want_load_tr_fft == 0:
        return

    # load density field
    f = asdf.open(ic_fn)
    delta = f['data']['dens'][:, :, :]
    print('mean delta', np.mean(delta))

    # do fourier transform
    delta_fft = rfftn(delta, workers=-1) / nmesh**3
    del delta
    gc.collect()

    # get the box k and mu modes
    k_bin_edges, mu_bin_edges = get_k_mu_edges(
        Lbox, k_hMpc_max, n_k_bins, n_mu_bins, logk
    )

    # do mu**2 delta and get the three power spectra from this
    fields = {'delta': delta_fft, 'deltamu2': get_delta_mu2(delta_fft, nmesh)}

    # compute the galaxy auto rsd poles
    print('Computing auto-correlation of tracer')
    if save_3D_power:
        power_tr_fns = []

        # compute
        pk3d = np.array((tr_field_fft * np.conj(tr_field_fft)).real, dtype=np.float32)
        # pk3d *= Lbox**3

        # record
        pk_tr_dict = {}
        pk_tr_dict['P_k3D_tr_tr'] = pk3d
        header = {}
        header['sim_name'] = sim_name
        header['Lbox'] = Lbox
        header['nmesh'] = nmesh
        header['kcut'] = kcut
        power_tr_fn = (
            Path(save_z_dir)
            / f'power{rsd_str}_tr_tr_{rec_algo}_lin_nmesh{nmesh:d}.asdf'
        )
        power_tr_fns.append(power_tr_fn)
        compress_asdf(str(power_tr_fn), pk_tr_dict, header)
    else:
        P = calc_pk_from_deltak(
            tr_field_fft,
            Lbox,
            k_bin_edges,
            mu_bin_edges,
            field2_fft=None,
            poles=np.asarray(poles),
        )
        pk_tr_dict['P_kmu_tr_tr'] = P['power']
        pk_tr_dict['N_kmu_tr_tr'] = P['N_mode']
        pk_tr_dict['P_ell_tr_tr'] = P['binned_poles']
        pk_tr_dict['N_ell_tr_tr'] = P['N_mode_poles']

    # initiate final arrays
    for i in range(len(keynames)):
        print('Computing cross-correlation of tracer and ', keynames[i])

        if save_3D_power:
            # compute
            pk3d = np.array(
                (fields[keynames[i]] * np.conj(tr_field_fft)).real, dtype=np.float32
            )

            # record
            pk_tr_dict = {}
            pk_tr_dict[f'P_k3D_{keynames[i]}_tr'] = pk3d
            header = {}
            header['sim_name'] = sim_name
            header['Lbox'] = Lbox
            header['nmesh'] = nmesh
            header['kcut'] = kcut
            power_tr_fn = (
                Path(save_z_dir)
                / f'power{rsd_str}_{keynames[i]}_tr_{rec_algo}_lin_nmesh{nmesh:d}.asdf'
            )
            power_tr_fns.append(power_tr_fn)
            compress_asdf(str(power_tr_fn), pk_tr_dict, header)
        else:
            # compute power spectrum
            P = calc_pk_from_deltak(
                fields[keynames[i]],
                Lbox,
                k_bin_edges,
                mu_bin_edges,
                field2_fft=tr_field_fft,
                poles=np.asarray(poles),
            )
            pk_tr_dict[f'P_kmu_{keynames[i]}_tr'] = P['power']
            pk_tr_dict[f'N_kmu_{keynames[i]}_tr'] = P['N_mode']
            pk_tr_dict[f'P_ell_{keynames[i]}_tr'] = P['binned_poles']
            pk_tr_dict[f'N_ell_{keynames[i]}_tr'] = P['N_mode_poles']

    if save_3D_power:
        return power_tr_fns
    header = {}
    header['sim_name'] = sim_name
    header['Lbox'] = Lbox
    header['nmesh'] = nmesh
    header['kcut'] = kcut

    if want_save:
        compress_asdf(str(power_tr_fn), pk_tr_dict, header)
    return pk_tr_dict
