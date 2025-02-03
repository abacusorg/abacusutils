import argparse
import gc
import os
from pathlib import Path
import warnings

import asdf
import numpy as np
import yaml

from abacusnbody.analysis.power_spectrum import (
    calc_pk_from_deltak,
    get_field_fft,
    get_k_mu_edges,
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

from asdf.exceptions import AsdfWarning

warnings.filterwarnings('ignore', category=AsdfWarning)

DEFAULTS = {'path2config': 'config/abacus_hod.yaml'}


def main(
    path2config,
    want_rsd=False,
    alt_simname=None,
    save_3D_power=False,
    only_requested_fields=False,
):
    r"""
    Advect the initial conditions fields (1cb, delta, delta^2, s^2, nabla^2) to
    some desired redshift and saving the 3D Fourier fields and power spectra in
    ASDF files along the way.

    Parameters
    ----------
    path2config : str
        name of the yaml containing parameter specifications.
    want_rsd : bool, optional
        compute the advected fields and power spectra in redshift space?
        Default is False.
    alt_simname : str, optional
        specify simulation name if different from yaml file.
    save_3D_power : bool, optional
        save the 3D power spectra in individual ASDF files.
        Default is False.
    only_requested_fields : bool, optional
        instead of all 5 fields, use only the `fields` specified in `zcv_params`.
        Default is False.

    Returns
    -------
    pk_ij_dict : dict
        dictionary containing the auto- and cross-power spectra of the 5 fields.
    """
    # read zcv parameters
    config = yaml.safe_load(open(path2config))
    zcv_dir = config['zcv_params']['zcv_dir']
    nmesh = config['zcv_params']['nmesh']
    kcut = config['zcv_params']['kcut']
    if only_requested_fields:
        keynames = config['zcv_params']['fields']
        warnings.warn(
            'Saving only requested fields. Delete pre-saved files and run again if changing `fields`.'
        )
    else:
        keynames = ['1cb', 'delta', 'delta2', 'tidal2', 'nabla2']

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

    # create save directory
    save_dir = Path(zcv_dir) / sim_name
    save_z_dir = save_dir / f'z{z_this:.3f}'
    os.makedirs(save_z_dir, exist_ok=True)

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
    ic_fn = Path(save_dir) / f'ic_filt_nmesh{nmesh:d}.asdf'
    fields_fn = Path(save_dir) / f'fields_nmesh{nmesh:d}.asdf'
    fields_fft_fn = []
    for i in range(len(keynames)):
        fields_fft_fn.append(
            Path(save_z_dir)
            / f'advected_{keynames[i]}_field{rsd_str}_fft_nmesh{nmesh:d}.asdf'
        )
    if not logk:
        dk = k_bin_edges[1] - k_bin_edges[0]
    else:
        dk = np.log(k_bin_edges[1] / k_bin_edges[0])
    if n_k_bins == nmesh // 2:
        power_ij_fn = Path(save_z_dir) / f'power{rsd_str}_ij_nmesh{nmesh:d}.asdf'
    else:
        power_ij_fn = (
            Path(save_z_dir) / f'power{rsd_str}_ij_nmesh{nmesh:d}_dk{dk:.3f}.asdf'
        )
        print('power file', str(power_ij_fn))

    # compute growth factor
    D = boltz.scale_independent_growth_factor(z_this)
    D /= boltz.scale_independent_growth_factor(z_ic)
    # Ha = boltz.Hubble(z_this) * 299792.458
    if want_rsd:
        f_growth = boltz.scale_independent_growth_factor_f(z_this)
    else:
        f_growth = 0.0
    print('D = ', D)

    # field names and growths
    field_D = [1, D, D**2, D**2, D]

    # save the advected fields
    if all(os.path.exists(fn) for fn in fields_fft_fn):
        fields_fft = []
        for i in range(len(keynames)):
            f = asdf.open(fields_fft_fn[i])
            fields_fft.append(f['data'])
            header = f['header']
            assert header['sim_name'] == sim_name, (
                f'Mismatch in the files: {str(fields_fft_fn[i])}'
            )
            assert np.isclose(header['Lbox'], Lbox), (
                f'Mismatch in the file: {str(fields_fft_fn[i])}'
            )
            assert header['nmesh'] == nmesh, (
                f'Mismatch in the file: {str(fields_fft_fn[i])}'
            )
            assert np.isclose(header['kcut'], kcut), (
                f'Mismatch in the file: {str(fields_fft_fn[i])}'
            )
            assert header['compensated'] == compensated, (
                f'Mismatch in the file: {str(fields_fft_fn[i])}'
            )
            assert header['interlaced'] == interlaced, (
                f'Mismatch in the file: {str(fields_fft_fn[i])}'
            )
            assert header['paste'] == paste, (
                f'Mismatch in the file: {str(fields_fft_fn[i])}'
            )
    else:
        # load density field and displacements
        f = asdf.open(ic_fn)
        header = f['header']
        assert header['nmesh'] == nmesh, 'Mismatch in the file: {str(ic_fn)}'
        assert np.isclose(header['kcut'], kcut), 'Mismatch in the file: {str(ic_fn)}'
        disp_pos = np.zeros((nmesh**3, 3), np.float32)
        disp_pos[:, 0] = f['data']['disp_x'][:, :, :].flatten() * D
        disp_pos[:, 1] = f['data']['disp_y'][:, :, :].flatten() * D
        disp_pos[:, 2] = f['data']['disp_z'][:, :, :].flatten() * D * (1 + f_growth)
        print('loaded displacements', disp_pos.dtype)
        f.close()
        gc.collect()

        # read in displacements, rescale by D=D(z_this)/D(z_ini)
        grid_x, grid_y, grid_z = np.meshgrid(
            np.arange(nmesh, dtype=np.float32) / nmesh,
            np.arange(nmesh, dtype=np.float32) / nmesh,
            np.arange(nmesh, dtype=np.float32) / nmesh,
            indexing='ij',
        )
        grid_x = grid_x.flatten()
        grid_y = grid_y.flatten()
        grid_z = grid_z.flatten()

        disp_pos[:, 0] += grid_x
        disp_pos[:, 1] += grid_y
        disp_pos[:, 2] += grid_z
        del grid_x, grid_y, grid_z
        gc.collect()

        disp_pos *= Lbox
        disp_pos %= Lbox
        print('box coordinates of the displacements', disp_pos.dtype)

        # Initiate fields
        for i in range(len(keynames)):
            if os.path.exists(fields_fft_fn[i]):
                continue
            print(keynames[i])
            if i == 0:
                w = None
            else:
                f = asdf.open(fields_fn)
                header = f['header']
                assert header['nmesh'] == nmesh, (
                    'Mismatch in the file: {str(fields_fn)}'
                )
                assert np.isclose(header['kcut'], kcut), (
                    'Mismatch in the file: {str(fields_fn)}'
                )
                w = f['data'][keynames[i]][:, :, :].flatten()
                f.close()
            field_fft = get_field_fft(
                disp_pos, Lbox, nmesh, paste, w, W, compensated, interlaced
            )
            del w
            gc.collect()
            table = {}
            table[f'{keynames[i]}_Re'] = np.array(field_fft.real, dtype=np.float32)
            table[f'{keynames[i]}_Im'] = np.array(field_fft.imag, dtype=np.float32)
            del field_fft
            gc.collect()

            # write out the advected fields
            header = {}
            header['sim_name'] = sim_name
            header['Lbox'] = Lbox
            header['nmesh'] = nmesh
            header['kcut'] = kcut
            header['compensated'] = compensated
            header['interlaced'] = interlaced
            header['paste'] = paste
            compress_asdf(fields_fft_fn[i], table, header)
            del table
            gc.collect()
        del disp_pos
        gc.collect()
    del W
    gc.collect()

    # check if pk_ij exists
    if os.path.exists(power_ij_fn) and not save_3D_power:
        pk_ij_dict = asdf.open(power_ij_fn)['data']
        return pk_ij_dict
    else:
        pk_ij_dict = {}
        pk_ij_dict['k_binc'] = k_binc
        pk_ij_dict['mu_binc'] = mu_binc

    # get the box k and mu modes
    k_bin_edges, mu_bin_edges = get_k_mu_edges(
        Lbox, k_hMpc_max, n_k_bins, n_mu_bins, logk
    )

    # initiate final arrays
    pk_auto = []
    # pk_cross = []
    for i in range(len(keynames)):
        for j in range(len(keynames)):
            if i < j:
                continue
            print('Computing cross-correlation of', keynames[i], keynames[j])

            # load field
            field_fft_i = asdf.open(fields_fft_fn[i], lazy_load=False)['data']
            field_fft_j = asdf.open(fields_fft_fn[j], lazy_load=False)['data']

            if save_3D_power:
                power_ij_fn = (
                    Path(save_z_dir)
                    / f'power{rsd_str}_{keynames[i]}_{keynames[j]}_nmesh{nmesh:d}.asdf'
                )
                if os.path.exists(power_ij_fn):
                    continue
                # construct
                field_fft_i = (
                    field_fft_i[f'{keynames[i]}_Re']
                    + 1j * field_fft_i[f'{keynames[i]}_Im']
                )
                field_fft_j = (
                    field_fft_j[f'{keynames[j]}_Re']
                    + 1j * field_fft_j[f'{keynames[j]}_Im']
                )

                # compute
                pk3d = np.array(
                    (field_fft_i * np.conj(field_fft_j)).real, dtype=np.float32
                )
                pk3d *= field_D[i] * field_D[j]
                # pk3d *= Lbox**3 # seems unnecessary

                # record
                pk_ij_dict = {}
                pk_ij_dict[f'P_k3D_{keynames[i]}_{keynames[j]}'] = pk3d
                header = {}
                header['sim_name'] = sim_name
                header['Lbox'] = Lbox
                header['nmesh'] = nmesh
                header['kcut'] = kcut
                compress_asdf(str(power_ij_fn), pk_ij_dict, header)
                del field_fft_i, field_fft_j
                gc.collect()
            else:
                # compute power spectrum
                P = calc_pk_from_deltak(
                    field_fft_i[f'{keynames[i]}_Re']
                    + 1j * field_fft_i[f'{keynames[i]}_Im'],
                    Lbox,
                    k_bin_edges,
                    mu_bin_edges,
                    field2_fft=field_fft_j[f'{keynames[j]}_Re']
                    + 1j * field_fft_j[f'{keynames[j]}_Im'],
                    poles=np.asarray(poles),
                )
                P['power'] *= field_D[i] * field_D[j]
                P['binned_poles'] *= field_D[i] * field_D[j]
                pk_auto.append(P['power'])
                pk_ij_dict[f'P_kmu_{keynames[i]}_{keynames[j]}'] = P['power']
                pk_ij_dict[f'N_kmu_{keynames[i]}_{keynames[j]}'] = P['N_mode']
                pk_ij_dict[f'P_ell_{keynames[i]}_{keynames[j]}'] = P['binned_poles']
                pk_ij_dict[f'N_ell_{keynames[i]}_{keynames[j]}'] = P['N_mode_poles']
                del field_fft_i, field_fft_j
                gc.collect()

    if not save_3D_power:
        # record power spectra
        header = {}
        header['sim_name'] = sim_name
        header['Lbox'] = Lbox
        header['nmesh'] = nmesh
        header['kcut'] = kcut
        compress_asdf(str(power_ij_fn), pk_ij_dict, header)
    return pk_ij_dict


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
    parser.add_argument('--want_rsd', help='Include RSD effects?', action='store_true')
    parser.add_argument('--alt_simname', help='Alternative simulation name')
    parser.add_argument(
        '--save_3D_power', help='Record full 3D power spectrum', action='store_true'
    )
    parser.add_argument(
        '--only_requested_fields',
        help='Save only the requested fields in the yaml file (not recommended)',
        action='store_true',
    )
    args = vars(parser.parse_args())
    if args['want_rsd']:
        for want_rsd in [True, False]:
            args['want_rsd'] = want_rsd
            main(**args)
            gc.collect()
    else:
        main(**args)
