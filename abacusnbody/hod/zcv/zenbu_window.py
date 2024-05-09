"""
Script for pre-saving zenbu for a given redshift and simulation.
"""

import argparse
import os
import warnings
from pathlib import Path

import numpy as np
import yaml
from numba import jit
from scipy.interpolate import interp1d

from abacusnbody.analysis.power_spectrum import get_k_mu_edges
from abacusnbody.metadata import get_meta

try:
    from classy import Class
    from ZeNBu.zenbu import Zenbu
    from ZeNBu.zenbu_rsd import Zenbu_RSD
except ImportError as e:
    raise ImportError(
        'Missing imports for zcv. Install abacusutils with '
        '"pip install abacusutils[all]" to install zcv dependencies.'
    ) from e

DEFAULTS = {'path2config': 'config/abacus_hod.yaml'}


@jit(nopython=True)
def meshgrid(x, y, z):
    """
    Create a 3D mesh given x, y and z.
    """
    xx = np.empty(shape=(y.size, x.size, z.size), dtype=x.dtype)
    yy = np.empty(shape=(y.size, x.size, z.size), dtype=y.dtype)
    zz = np.empty(shape=(y.size, x.size, z.size), dtype=z.dtype)
    for i in range(y.size):
        for j in range(x.size):
            for k in range(z.size):
                xx[i, j, k] = x[i]  # change to x[k] if indexing xy
                yy[i, j, k] = y[j]  # change to y[j] if indexing xy
                zz[i, j, k] = z[k]  # change to z[i] if indexing xy
    return zz, yy, xx


@jit(nopython=True)
def periodic_window_function(nmesh, lbox, kout, kin, k2weight=True):
    """
    Compute matrix appropriate for convolving a finely evaluated
    theory prediction with the mode coupling matrix.

    Parameters
    ----------
    nmesh : int
        size of the mesh used for power spectrum measurement.
    lbox : float
        box size of the simulation.
    kout : array_like
        k bins used for power spectrum measurement.
    kin : array_like
        Defaults to None.
    k2weight : bool
        Defaults to True.

    Returns
    -------
    window : array_like
        np.dot(window, pell_th) gives convovled theory
    keff : array_like
        effective k value of each output k bin.
    """

    kvals = np.zeros(nmesh, dtype=np.float32)
    kvals[: nmesh // 2] = np.arange(
        0, 2 * np.pi * nmesh / lbox / 2, 2 * np.pi / lbox, dtype=np.float32
    )
    kvals[nmesh // 2 :] = np.arange(
        -2 * np.pi * nmesh / lbox / 2, 0, 2 * np.pi / lbox, dtype=np.float32
    )

    kvalsr = np.arange(
        0, 2 * np.pi * nmesh / lbox / 2, 2 * np.pi / lbox, dtype=np.float32
    )
    kx, ky, kz = meshgrid(kvals, kvals, kvalsr)
    knorm = np.sqrt(kx**2 + ky**2 + kz**2)
    mu = kz / knorm
    mu[0, 0, 0] = 0

    ellmax = 3

    nkin = len(kin)

    if k2weight:
        dk = np.zeros_like(kin)
        dk[:-1] = kin[1:] - kin[:-1]
        dk[-1] = dk[-2]

    nkout = len(kout) - 1
    (kin[1:] - kin[:-1])[0]

    idx_o = np.digitize(knorm, kout) - 1
    nmodes_out = np.zeros(nkout * 3)

    idx_i = np.digitize(kin, kout) - 1
    nmodes_in = np.zeros(nkout, dtype=np.float32)

    for i in range(len(kout)):
        idx = i == idx_i
        if k2weight:
            nmodes_in[i] = np.sum(kin[idx] ** 2 * dk[idx])
        else:
            nmodes_in[i] = np.sum(idx)

    norm_in = 1 / nmodes_in
    norm_in[nmodes_in == 0] = 0
    norm_in_allell = np.zeros(3 * len(norm_in))
    norm_in_allell[:nkout] = norm_in
    norm_in_allell[nkout : 2 * nkout] = norm_in
    norm_in_allell[2 * nkout : 3 * nkout] = norm_in

    window = np.zeros((nkout * 3, nkin * 3), dtype=np.float32)
    keff = np.zeros(nkout, dtype=np.float32)

    L0 = np.ones_like(mu, dtype=np.float32)
    L2 = (3 * mu**2 - 1) / 2
    L4 = (35 * mu**4 - 30 * mu**2 + 3) / 8

    legs = [L0, L2, L4]
    pref = [1, (2 * 2 + 1), (2 * 4 + 1)]

    for i in range(kx.shape[0]):
        for j in range(kx.shape[1]):
            for k in range(kx.shape[2]):
                if idx_o[i, j, k] >= nkout:
                    pass
                else:
                    if k == 0:
                        nmodes_out[idx_o[i, j, k] :: nkout] += 1
                        keff[idx_o[i, j, k]] += knorm[i, j, k]
                    else:
                        nmodes_out[idx_o[i, j, k] :: nkout] += 2
                        keff[idx_o[i, j, k]] += 2 * knorm[i, j, k]

                    for beta in range(nkin):
                        if k2weight:
                            w = kin[beta] ** 2 * dk[beta]
                        else:
                            w = 1
                        if idx_i[beta] == idx_o[i, j, k]:
                            for ell in range(ellmax):
                                for ellp in range(ellmax):
                                    if k != 0:
                                        window[
                                            int(ell * nkout) + int(idx_o[i, j, k]),
                                            int(ellp * nkin) + int(beta),
                                        ] += (
                                            2
                                            * pref[ell]
                                            * legs[ell][i, j, k]
                                            * legs[ellp][i, j, k]
                                            * w
                                        )  # * norm_in[idx_o[i,j,k]]
                                    else:
                                        window[
                                            int(ell * nkout) + int(idx_o[i, j, k]),
                                            int(ellp * nkin) + int(beta),
                                        ] += (
                                            pref[ell]
                                            * legs[ell][i, j, k]
                                            * legs[ellp][i, j, k]
                                            * w
                                        )  # * norm_in[idx_o[i,j,k]]

    norm_out = 1 / nmodes_out
    norm_out[nmodes_out == 0] = 0
    window = window * norm_out.reshape(-1, 1) * norm_in_allell.reshape(-1, 1)
    keff = keff * norm_out[:nkout]

    return window, keff


def zenbu_spectra(
    k, z, cfg, kin, pin, pkclass=None, N=2700, jn=15, rsd=True, nmax=6, ngauss=6
):
    """
    Compute the ZeNBu power spectra.
    """

    if pkclass is None:
        pkclass = Class()
        pkclass.set(cfg['Cosmology'])
        pkclass.compute()

    cutoff = cfg['surrogate_gaussian_cutoff']
    cutoff = float(cfg['surrogate_gaussian_cutoff'])

    Dthis = pkclass.scale_independent_growth_factor(z)
    Dic = pkclass.scale_independent_growth_factor(cfg['z_ic'])
    f = pkclass.scale_independent_growth_factor_f(z)

    if rsd:
        lptobj, p0spline, p2spline, p4spline, pspline = _lpt_pk(
            kin,
            pin * (Dthis / Dic) ** 2,
            f,
            cutoff=cutoff,
            third_order=False,
            one_loop=False,
            jn=jn,
            N=N,
            nmax=nmax,
            ngauss=ngauss,
        )
        pk_zenbu = pspline(k)

    else:
        pspline, lptobj = _realspace_lpt_pk(
            kin, pin * (Dthis / Dic) ** 2, cutoff=cutoff
        )
        pk_zenbu = pspline(k)[1:]

    return pk_zenbu[:11], lptobj


def _lpt_pk(
    k,
    p_lin,
    f,
    cleftobj=None,
    third_order=True,
    one_loop=True,
    cutoff=np.pi * 700 / 525.0,
    jn=15,
    N=2700,
    nmax=8,
    ngauss=3,
):
    r"""
    LPT helper function for creating a bunch of splines.
    """

    lpt = Zenbu_RSD(k, p_lin, jn=jn, N=N, cutoff=cutoff)
    lpt.make_pltable(f, kv=k, nmax=nmax, ngauss=ngauss)

    p0table = lpt.p0ktable
    p2table = lpt.p2ktable
    p4table = lpt.p4ktable

    pktable = np.zeros((len(p0table), 3, p0table.shape[-1]))
    pktable[:, 0, :] = p0table
    pktable[:, 1, :] = p2table
    pktable[:, 2, :] = p4table

    pellspline = interp1d(k, pktable.T, fill_value='extrapolate')  # , kind='cubic')
    p0spline = interp1d(k, p0table.T, fill_value='extrapolate')  # , kind='cubic')
    p2spline = interp1d(k, p2table.T, fill_value='extrapolate')  # , kind='cubic')
    p4spline = interp1d(k, p4table.T, fill_value='extrapolate')  # , kind='cubic')

    return lpt, p0spline, p2spline, p4spline, pellspline


def _realspace_lpt_pk(k, p_lin, D=None, cleftobj=None, cutoff=np.pi * 700 / 525.0):
    r"""
    Returns a spline object which computes the cleft component spectra.
    Computed either in "full" CLEFT or in "k-expanded" CLEFT (kecleft)
    which allows for faster redshift dependence.

    Parameters
    ----------
    k : array-like
        array of wavevectors to compute power spectra at (in h/Mpc).
    p_lin : array-like
        linear power spectrum to produce velocileptors predictions for.
        If kecleft==True, then should be for z=0, and redshift evolution is
        handled by passing the appropriate linear growth factor to D.
    D : float
        linear growth factor. Only required if kecleft==True.
    kecleft : bool
        allows for faster computation of spectra keeping cosmology
        fixed and varying redshift if the cleftobj from the
        previous calculation at the same cosmology is provided to
        the cleftobj keyword.
    cutoff : float
        Gaussian cutoff scale.

    Returns
    -------
    cleftspline : InterpolatedUnivariateSpline
        Spline that computes basis spectra as a function of k.
    cleftobj: CLEFT object
        CLEFT object used to compute basis spectra.
    """

    zobj = Zenbu(k, p_lin, cutoff=cutoff, N=3000, jn=15)
    zobj.make_ptable(kvec=k)
    cleftpk = zobj.pktable.T
    cleftobj = zobj
    cleftspline = interp1d(cleftpk[0], cleftpk, fill_value='extrapolate')

    return cleftspline, cleftobj


def main(path2config, alt_simname=None, want_xi=False):
    """
    Save the mode-coupling window function and the ZeNBu power spectra
    as `npz` files given ZCV specs.

    Parameters
    ----------
    path2config : str
        name of the yaml containing parameter specifications.
    alt_simname : str, optional
        specify simulation name if different from yaml file.
    """
    # read zcv parameters
    config = yaml.safe_load(open(path2config))
    zcv_dir = config['zcv_params']['zcv_dir']
    # ic_dir = config['zcv_params']['ic_dir']
    nmesh = config['zcv_params']['nmesh']
    kcut = config['zcv_params']['kcut']

    # power params
    if alt_simname is not None:
        sim_name = alt_simname
    else:
        sim_name = config['sim_params']['sim_name']
    z_this = config['sim_params']['z_mock']

    # create save directory
    save_dir = Path(zcv_dir) / sim_name
    save_z_dir = save_dir / f'z{z_this:.3f}'
    os.makedirs(save_z_dir, exist_ok=True)

    # read meta data
    meta = get_meta(sim_name, redshift=z_this)
    Lbox = meta['BoxSize']
    z_ic = meta['InitialRedshift']
    D_ratio = meta['GrowthTable'][z_ic] / meta['GrowthTable'][1.0]
    # k_Ny = np.pi*nmesh/Lbox
    cosmo = {}
    cosmo['output'] = 'mPk mTk'
    cosmo['P_k_max_h/Mpc'] = 20.0
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
    rsd = config['HOD_params']['want_rsd']
    rsd_str = '_rsd' if rsd else ''

    # make sure that the parameters are set correctly
    if want_xi:
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

    # define k bins
    k_bins, mu_bins = get_k_mu_edges(Lbox, k_hMpc_max, n_k_bins, n_mu_bins, logk)
    k_binc = (k_bins[1:] + k_bins[:-1]) * 0.5

    # name of file to save to
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
    pk_lin_fn = save_dir / 'abacus_pk_lin_ic.dat'

    # load the power spectrum at z = 1
    if os.path.exists(pk_lin_fn):
        # load linear power spectrum
        p_in = np.loadtxt(pk_lin_fn)
        kth, p_m_lin = p_in[:, 0], p_in[:, 1]
    else:
        # TODO: this code path maybe not tested since addition of CLASS_power_spectrum
        kth = meta['CLASS_power_spectrum']['k (h/Mpc)']
        pk_z1 = meta['CLASS_power_spectrum']['P (Mpc/h)^3']
        p_m_lin = D_ratio**2 * pk_z1

        # to match the lowest k-modes in zenbu (make pretty)
        choice = kth > 1.0e-05
        kth = kth[choice]
        p_m_lin = p_m_lin[choice]
        np.savetxt(pk_lin_fn, np.vstack((kth, p_m_lin)).T)

    # create a dict with everything you would ever need
    cfg = {
        'lbox': Lbox,
        'nmesh_in': nmesh,
        'p_lin_ic_file': pk_lin_fn,
        'Cosmology': cosmo,
        'surrogate_gaussian_cutoff': kcut,
        'z_ic': z_ic,
    }

    # presave the zenbu power spectra
    print('Generating ZeNBu output')
    if os.path.exists(zenbu_fn):
        print('Already saved zenbu for this simulation, redshift and RSD choice.')
    else:
        pk_ij_zenbu, lptobj = zenbu_spectra(
            k_binc, z_this, cfg, kth, p_m_lin, pkclass=None, rsd=rsd
        )
        if rsd:
            p0table = lptobj.p0ktable
            p2table = lptobj.p2ktable
            p4table = lptobj.p4ktable
            lptobj = np.array([p0table, p2table, p4table])
        else:
            lptobj = lptobj.pktable
        np.savez(
            zenbu_fn, pk_ij_zenbu=pk_ij_zenbu, lptobj=lptobj, k_binc=k_binc, kcut=kcut
        )
        print('Saved zenbu for this simulation, redshift and RSD choice.')

    # presave the window function
    print('Generating window function')
    if os.path.exists(window_fn):
        print('Already saved window for this choice of box and nmesh')
    else:
        window, keff = periodic_window_function(
            nmesh, Lbox, k_bins, k_binc, k2weight=True
        )
        np.savez(window_fn, window=window, keff=keff)
        print('Saved window for this choice of box and nmesh')


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
        '--want_xi', help='Set up parameters for Xi computation', action='store_true'
    )
    args = vars(parser.parse_args())
    main(**args)
