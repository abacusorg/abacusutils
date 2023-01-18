"""
TODO: Fix cosmology here too
Just run twice for rsd and no rsd
CHECK: Syntax for load pk_ij
Might not be necessary to save
"""
import gc
import os
from pathlib import Path

import asdf
import numpy as np

from abacusnbody.hod.power_spectrum import (calc_pk3d, get_field_fft,
                                            get_k_mu_box_edges, get_k_mu_edges,
                                            get_W_compensated)
from abacusnbody.metadata import get_meta

from .ic_fields import compress_asdf

try:
    from classy import Class
except ImportError as e:
    raise ImportError('Could not import classy. Install abacusutils with '
        '"pip install abacusutils[zcv]" to install zcv dependencies.')


def get_tracer_power(tracer_pos, want_rsd, config, want_save=True):
    # field names
    keynames = ["1cb", "delta", "delta2", "tidal2", "nabla2"]

    # read zcv parameters
    advected_dir = config['zcv_params']['zcv_dir']  # input of advected fields
    tracer_dir = config['zcv_params']['tracer_dir']  # output of tracers
    # ic_dir = config['zcv_params']['ic_dir']
    nmesh = config['zcv_params']['nmesh']
    kcut = config['zcv_params']['kcut']

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
    #want_rsd = config['HOD_params']['want_rsd']
    rsd_str = "_rsd" if want_rsd else ""

    # get a few parameters for the simulation
    meta = get_meta(sim_name, redshift=z_this)
    Lbox = meta['BoxSize']
    z_ic = meta['InitialRedshift']
    # k_Ny = np.pi*nmesh/Lbox
    
    # define k, mu bins
    n_perp = n_los = nmesh
    k_bin_edges, mu_bin_edges = get_k_mu_edges(Lbox, k_hMpc_max, n_k_bins, n_mu_bins, logk)
    k_binc = (k_bin_edges[1:]+k_bin_edges[:-1])*.5
    mu_binc = (mu_bin_edges[1:]+mu_bin_edges[:-1])*.5

    # start dictionary with power spectra
    pk_tr_dict = {}
    pk_tr_dict['k_binc'] = k_binc
    pk_tr_dict['mu_binc'] = mu_binc
    
    # create save directory
    save_dir = Path(tracer_dir) / sim_name
    save_z_dir = save_dir / f"z{z_this:.3f}"
    save_z_dir.mkdir(exist_ok=True, parents=True)

    # get path to input directory
    advected_dir_z_dir = Path(advected_dir) / sim_name / f"z{z_this:.3f}"

    # get the window function of TSC/CIC
    if compensated:
        W = get_W_compensated(Lbox, nmesh, paste, interlaced)
    else:
        W = None
    
    # set up cosmology
    boltz = Class()
    cosmo = {}
    cosmo['output'] = 'mPk mTk'
    cosmo['P_k_max_h/Mpc'] = 20.
    for k in ('H0', 'omega_b', 'omega_cdm',
              'omega_ncdm', 'N_ncdm', 'N_ur',
              'n_s', #'A_s', 'alpha_s',
              #'wa', 'w0',
              ):
        cosmo[k] = meta[k]
    boltz.set(cosmo)
    boltz.compute()
    
    # file to save to
    # ic_fn = Path(save_dir) / f"ic_filt_nmesh{nmesh:d}.asdf"
    # fields_fn = Path(save_dir) / f"fields_nmesh{nmesh:d}.asdf"
    fields_fft_fn = []
    for i in range(len(keynames)):
        fields_fft_fn.append(advected_dir_z_dir / f"advected_{keynames[i]}_field{rsd_str}_fft_nmesh{nmesh:d}.asdf")
    tr_field_fft_fn = Path(save_z_dir) / f"tr_field{rsd_str}_fft_nmesh{nmesh:d}.asdf"
    power_tr_fn = Path(save_z_dir) / f"power{rsd_str}_tr_nmesh{nmesh:d}.asdf"

    # compute growth factor
    D = boltz.scale_independent_growth_factor(z_this)
    D /= boltz.scale_independent_growth_factor(z_ic)
    # Ha = boltz.Hubble(z_this) * 299792.458
    # if want_rsd:
    #     f_growth = boltz.scale_independent_growth_factor_f(z_this)
    # else:
    #     f_growth = 0.
    print("D = ", D)
    
    # field names and growths
    field_D = [1, D, D**2, D**2, D]

    # create fft field for the tracer
    w = None
    tracer_pos += Lbox/2. # I think necessary for cross correlations
    tracer_pos %= Lbox
    """
    # TESTING
    import h5py
    tracer_fn = "/global/cscratch1/sd/boryanah/zcv/test_base.h5" #test_base_z3.000.h5" ##test_base.h5" # test.h5
    if want_rsd:
        tracer_pos = h5py.File(tracer_fn)["pos_zspace"][:, :]
    else:
        tracer_pos = h5py.File(tracer_fn)["pos_rspace"][:, :]
    """
    print("min/max tracer pos", tracer_pos.min(), tracer_pos.max(), tracer_pos.shape)
    tr_field_fft = get_field_fft(tracer_pos, Lbox, nmesh, paste, w, W, compensated, interlaced)
    del tracer_pos; gc.collect()

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
        del table; gc.collect()

    # TESTING
    #tr_field_fft = asdf.open(tr_field_fft_fn)['data']['tr_field_fft_Re'] + 1j * asdf.open(tr_field_fft_fn)['data']['tr_field_fft_Im']
    
    # get the box k and mu modes
    k_box, mu_box, k_bin_edges, mu_bin_edges = get_k_mu_box_edges(Lbox, n_perp, n_los, n_k_bins, n_mu_bins, k_hMpc_max, logk)
    
    # compute the galaxy auto rsd poles
    print("Computing auto-correlation of tracer")
    pk3d, N3d, binned_poles, Npoles = calc_pk3d(tr_field_fft, Lbox, k_box, mu_box, k_bin_edges, mu_bin_edges, logk, field2_fft=None, poles=poles)
    pk_tr_dict['P_kmu_tr_tr'] = pk3d
    pk_tr_dict['N_kmu_tr_tr'] = N3d
    pk_tr_dict['P_ell_tr_tr'] = binned_poles
    pk_tr_dict['N_ell_tr_tr'] = Npoles
    #np.savez("/global/homes/b/boryanah/zcv/power_tr.npz", pk3d=pk3d, N3d=N3d, binned_poles=binned_poles, Npoles=Npoles)
    
    # initiate final arrays
    # pk_auto = []
    pk_cross = []
    # pkcounter = 0
    for i in range(len(keynames)):
        print("Computing cross-correlation of tracer and ", keynames[i])

        # load field
        field_fft_i = asdf.open(fields_fft_fn[i])['data']
        
        # compute power spectrum
        pk3d, N3d, binned_poles, Npoles = calc_pk3d(field_fft_i[f'{keynames[i]}_Re']+1j*field_fft_i[f'{keynames[i]}_Im'], Lbox, k_box, mu_box, k_bin_edges, mu_bin_edges, logk, field2_fft=tr_field_fft, poles=poles)
        pk3d *= field_D[i]
        binned_poles *= field_D[i]
        pk_cross.append(pk3d)
        pk_tr_dict[f'P_kmu_{keynames[i]}_tr'] = pk3d
        pk_tr_dict[f'N_kmu_{keynames[i]}_tr'] = N3d
        pk_tr_dict[f'P_ell_{keynames[i]}_tr'] = binned_poles
        pk_tr_dict[f'N_ell_{keynames[i]}_tr'] = Npoles
        del field_fft_i; gc.collect()
                        
    header = {}
    header['sim_name'] = sim_name
    header['Lbox'] = Lbox
    header['nmesh'] = nmesh
    header['kcut'] = kcut
    if want_save:
        compress_asdf(str(power_tr_fn), pk_tr_dict, header)
    return pk_tr_dict
