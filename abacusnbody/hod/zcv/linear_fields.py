"""
Script for saving the linear density field power spectrum and cross correlations needed for Kaiser effects
TODO: change to pyfftw
"""

import argparse
import gc
import os
from pathlib import Path

import asdf
import numpy as np
import yaml
from scipy.fft import rfftn

from abacusnbody.analysis.power_spectrum import (
    calc_pk_from_deltak,
    get_k_mu_edges,
    get_delta_mu2,
    get_W_compensated,
)
from abacusnbody.metadata import get_meta

from .ic_fields import compress_asdf

DEFAULTS = {'path2config': 'config/abacus_hod.yaml'}


def main(path2config, alt_simname=None, save_3D_power=False):
    r"""
    Advect the initial conditions density field to some desired redshift
    and saving the 3D Fourier fields (delta, delta*mu^2) and power spectra in
    ASDF files along the way.

    Parameters
    ----------
    path2config : str
        name of the yaml containing parameter specifications.
    alt_simname : str, optional
        specify simulation name if different from yaml file.
    save_3D_power : bool, optional
        save the 3D power spectra in individual ASDF files.
        Default is False.

    Returns
    -------
    pk_lin_dict : dict
        dictionary containing the auto- and cross-power spectra of the 2 fields.
    """
    # field names
    keynames = ['delta', 'deltamu2']

    # read lcv parameters
    config = yaml.safe_load(open(path2config))
    lcv_dir = config['lcv_params']['lcv_dir']
    config['lcv_params']['ic_dir']
    nmesh = config['lcv_params']['nmesh']
    kcut = config['lcv_params']['kcut']

    # power params
    if alt_simname is not None:
        sim_name = alt_simname
    else:
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

    # get a few parameters for the simulation
    meta = get_meta(sim_name, redshift=z_this)
    Lbox = meta['BoxSize']

    # define k, mu bins
    k_bin_edges, mu_bin_edges = get_k_mu_edges(
        Lbox, k_hMpc_max, n_k_bins, n_mu_bins, logk
    )
    k_binc = (k_bin_edges[1:] + k_bin_edges[:-1]) * 0.5
    mu_binc = (mu_bin_edges[1:] + mu_bin_edges[:-1]) * 0.5

    # create save directory
    save_dir = Path(lcv_dir) / sim_name
    save_z_dir = save_dir / f'z{z_this:.3f}'
    os.makedirs(save_z_dir, exist_ok=True)

    # get the window function of TSC/CIC
    if compensated:
        get_W_compensated(Lbox, nmesh, paste, interlaced)
    else:
        pass

    # file to save to
    ic_fn = Path(save_dir) / f'ic_filt_nmesh{nmesh:d}.asdf'
    if not logk:
        dk = k_bin_edges[1] - k_bin_edges[0]
    else:
        dk = np.log(k_bin_edges[1] / k_bin_edges[0])
    if n_k_bins == nmesh // 2:
        power_lin_fn = Path(save_dir) / f'power_lin_nmesh{nmesh:d}.asdf'
    else:
        power_lin_fn = Path(save_dir) / f'power_lin_nmesh{nmesh:d}_dk{dk:.3f}.asdf'

    # load density field
    f = asdf.open(ic_fn)
    delta = f['data']['dens'][:, :, :]
    print('mean delta', np.mean(delta))

    # do fourier transform
    delta_fft = rfftn(delta, workers=-1) / np.float32(nmesh**3)
    del delta
    gc.collect()

    # get the box k and mu modes
    k_bin_edges, mu_bin_edges = get_k_mu_edges(
        Lbox, k_hMpc_max, n_k_bins, n_mu_bins, logk
    )

    # do mu**2 delta and get the three power spectra from this
    fields_fft = {'delta': delta_fft, 'deltamu2': get_delta_mu2(delta_fft, nmesh)}

    # save the power spectra
    pk_lin_dict = {}
    pk_lin_dict['k_binc'] = k_binc
    pk_lin_dict['mu_binc'] = mu_binc
    for i in range(len(keynames)):
        for j in range(len(keynames)):
            if i < j:
                continue
            print('Computing cross-correlation of', keynames[i], keynames[j])

            if save_3D_power:
                # compute
                pk3d = np.array(
                    (fields_fft[keynames[i]] * np.conj(fields_fft[keynames[j]])).real,
                    dtype=np.float32,
                )

                # record
                pk_ij_dict = {}
                pk_ij_dict[f'P_k3D_{keynames[i]}_{keynames[j]}'] = pk3d
                header = {}
                header['sim_name'] = sim_name
                header['Lbox'] = Lbox
                header['nmesh'] = nmesh
                header['kcut'] = kcut
                power_ij_fn = (
                    Path(save_z_dir)
                    / f'power_{keynames[i]}_{keynames[j]}_lin_nmesh{nmesh:d}.asdf'
                )
                compress_asdf(str(power_ij_fn), pk_ij_dict, header)

            else:
                # compute power spectrum
                P = calc_pk_from_deltak(
                    fields_fft[keynames[i]],
                    Lbox,
                    k_bin_edges,
                    mu_bin_edges,
                    field2_fft=fields_fft[keynames[j]],
                    poles=np.asarray(poles),
                )
                pk_lin_dict[f'P_kmu_{keynames[i]}_{keynames[j]}'] = P['power']
                pk_lin_dict[f'N_kmu_{keynames[i]}_{keynames[j]}'] = P['N_mode']
                pk_lin_dict[f'P_ell_{keynames[i]}_{keynames[j]}'] = P['binned_poles']
                pk_lin_dict[f'N_ell_{keynames[i]}_{keynames[j]}'] = P['N_mode_poles']

    # record power spectra
    header = {}
    header['sim_name'] = sim_name
    header['Lbox'] = Lbox
    header['nmesh'] = nmesh
    header['kcut'] = kcut
    compress_asdf(str(power_lin_fn), pk_lin_dict, header)
    return pk_lin_dict


class ArgParseFormatter(
    argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter
):
    pass


if __name__ == '__main__':
    # parsing arguments
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=ArgParseFormatter
    )
    parser.add_argument(
        '--path2config', help='Path to the config file', default=DEFAULTS['path2config']
    )
    parser.add_argument('--alt_simname', help='Alternative simulation name')
    parser.add_argument(
        '--save_3D_power', help='Record full 3D power spectrum', action='store_true'
    )
    args = vars(parser.parse_args())
    main(**args)
