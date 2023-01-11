"""Script for pre-saving zenbu for a given redshift and simulation

TODO: Make the kbins nicer (and maybe read as a function here)
Make the cosmology more correct (A_s and sigma8 are not specified)
Change format of zenbu_fn to something nicer
Test no RSD
Note that the linear power from abacus is cb and not m
Make c000 reading nicer
Fix the k-cut that matches zenbu to make prety

"""
import os
from pathlib import Path

import yaml
import argparse
import numpy as np

from .tools_jdr import zenbu_spectra, periodic_window_function
from abacusnbody.metadata import get_meta
from abacusnbody.hod.power_spectrum import get_k_mu_edges

DEFAULTS = {'path2config': 'config/abacus_hod.yaml'}

def main(path2config):

    # read zcv parameters
    config = yaml.safe_load(open(path2config))
    zcv_dir = config['zcv_params']['zcv_dir']
    ic_dir = config['zcv_params']['ic_dir']
    cosmo_dir = config['zcv_params']['cosmo_dir']
    nmesh = config['zcv_params']['nmesh']
    kcut = config['zcv_params']['kcut']

    # power params
    sim_name = config['sim_params']['sim_name']
    z_this = config['sim_params']['z_mock']
    k_hMpc_max = config['power_params']['k_hMpc_max']
    logk = config['power_params']['logk']
    n_k_bins = config['power_params']['nbins_k']
    n_mu_bins = config['power_params']['nbins_mu']
    rsd = config['HOD_params']['want_rsd']
    rsd_str = "_rsd" if rsd else ""
        
    # create save directory
    save_dir = Path(zcv_dir) / sim_name
    save_z_dir = save_dir / f"z{z_this:.3f}"
    os.makedirs(save_z_dir, exist_ok=True)
    
    # read meta data
    meta = get_meta(sim_name, redshift=z_this)
    Lbox = meta['BoxSize']
    z_ic = meta['InitialRedshift']
    D_ratio = meta['GrowthTable'][z_ic]/meta['GrowthTable'][1.0]
    k_Ny = np.pi*nmesh/Lbox
    cosmo = {}
    cosmo['output'] = 'mPk mTk'
    cosmo['P_k_max_h/Mpc'] = 20.
    cosmo['H0'] = meta['H0']
    cosmo['omega_b'] = meta['omega_b']
    cosmo['omega_cdm'] = meta['omega_cdm']
    cosmo['omega_ncdm'] = meta['omega_ncdm']
    cosmo['N_ncdm'] = meta['N_ncdm']
    cosmo['N_ur'] = meta['N_ur']
    cosmo['n_s'] = meta['n_s']
    #cosmo['wa'] = meta['wa']
    #cosmo['w0'] = meta['w0']

    # define k bins
    k_bins, mu_bins = get_k_mu_edges(Lbox, k_hMpc_max, n_k_bins, n_mu_bins, logk)
    k_binc = (k_bins[1:] + k_bins[:-1])*.5
    
    # name of file to save to
    zenbu_fn = save_z_dir / f"zenbu_pk{rsd_str}_ij_lpt_nmesh{nmesh:d}.npz"
    pk_lin_fn = save_dir / "abacus_pk_lin_ic.dat"
    window_fn = save_dir / f"window_nmesh{nmesh:d}.npz"
    
    # load the power spectrum at z = 1
    if os.path.exists(pk_lin_fn):
        # load linear power spectrum
        p_in = np.loadtxt(pk_lin_fn)
        kth, p_m_lin = p_in[:, 0], p_in[:, 1]
    else:
        c = int((sim_name.split('_c')[-1]).split('_ph')[0])
        kth, pk_1 = np.loadtxt(Path(cosmo_dir) / f"abacus_cosm{c:03d}" / "CLASS_power", unpack=True)
        p_m_lin = D_ratio**2*pk_1
        
        # to match the lowest k-modes in zenbu (make pretty)
        choice = kth > 1.e-05
        kth = kth[choice]
        p_m_lin = p_m_lin[choice]
        np.savetxt(pk_lin_fn, np.vstack((kth, p_m_lin)).T)
        
    # create a dict with everything you would ever need
    cfg = {'lbox': Lbox, 'nmesh_in': nmesh, 'p_lin_ic_file': pk_lin_fn, 'Cosmology': cosmo, 'surrogate_gaussian_cutoff': kcut, 'z_ic': z_ic}
                
    # presave the zenbu power spectra
    print("Generating ZeNBu output")
    if os.path.exists(zenbu_fn):
        print("Already saved zenbu for this simulation, redshift and RSD choice.")
    else:
        pk_ij_zenbu, lptobj = zenbu_spectra(k_binc, z_this, cfg, kth, p_m_lin, pkclass=None, rsd=rsd)
        if rsd:
            p0table = lptobj.p0ktable
            p2table = lptobj.p2ktable
            p4table = lptobj.p4ktable
            lptobj = np.array([p0table, p2table, p4table])
        else:
            lptobj = lptobj.pktable
        np.savez(zenbu_fn, pk_ij_zenbu=pk_ij_zenbu, lptobj=lptobj)
        print("Saved zenbu for this simulation, redshift and RSD choice.")
        
    # presave the window function
    print("Generating window function")
    if os.path.exists(window_fn):
        print("Already saved window for this choice of box and nmesh")
    else:
        window, keff = periodic_window_function(nmesh, Lbox, k_bins, k_binc, k2weight=True)
        np.savez(window_fn, window=window, keff=keff)
        print("Saved window for this choice of box and nmesh")

        
class ArgParseFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    pass

if __name__ == "__main__":

    # parsing arguments
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=ArgParseFormatter)
    parser.add_argument('--path2config', help='Path to the config file', default=DEFAULTS['path2config'])
    args = vars(parser.parse_args())
    main(**args)