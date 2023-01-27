"""
TODO: Fix cosmology here too
Just run twice for rsd and no rsd
CHECK: Syntax for load pk_ij
Might not be necessary to save
"""
import argparse
import gc
import os
from pathlib import Path

import asdf
import numpy as np
import yaml
from classy import Class

from abacusnbody.hod.power_spectrum import (calc_pk3d, get_field_fft,
                                            get_k_mu_box_edges, get_k_mu_edges,
                                            get_W_compensated)
from abacusnbody.metadata import get_meta

from .ic_fields import compress_asdf

DEFAULTS = {'path2config': 'config/abacus_hod.yaml'}

def main(path2config, want_rsd=False, alt_simname=None):

    # read zcv parameters
    config = yaml.safe_load(open(path2config))
    zcv_dir = config['zcv_params']['zcv_dir']
    ic_dir = config['zcv_params']['ic_dir']
    nmesh = config['zcv_params']['nmesh']
    kcut = config['zcv_params']['kcut']
    keynames = config['zcv_params']['fields']

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
    #want_rsd = config['HOD_params']['want_rsd']
    rsd_str = "_rsd" if want_rsd else ""

    # get a few parameters for the simulation
    meta = get_meta(sim_name, redshift=z_this)
    Lbox = meta['BoxSize']
    z_ic = meta['InitialRedshift']
    k_Ny = np.pi*nmesh/Lbox
    
    # define k, mu bins
    n_perp = n_los = nmesh
    k_bin_edges, mu_bin_edges = get_k_mu_edges(Lbox, k_hMpc_max, n_k_bins, n_mu_bins, logk)
    k_binc = (k_bin_edges[1:]+k_bin_edges[:-1])*.5
    mu_binc = (mu_bin_edges[1:]+mu_bin_edges[:-1])*.5
    
    # create save directory
    save_dir = Path(zcv_dir) / sim_name
    save_z_dir = save_dir / f"z{z_this:.3f}"
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
    cosmo['P_k_max_h/Mpc'] = 20.
    # TESTING!!!!!!!!!!!!!!!
    phase = int(sim_name.split('ph')[-1])
    if phase <= 6 and z_this == 0.8: # case old convention:
        for k in ('H0', 'omega_b', 'omega_cdm',
                  'omega_ncdm', 'N_ncdm', 'N_ur',
                  'n_s', #'A_s', 'alpha_s',
                  #'wa', 'w0',
        ):
            cosmo[k] = meta[k]
    else:
        for k in ('H0', 'omega_b', 'omega_cdm',
                  'omega_ncdm', 'N_ncdm', 'N_ur',
                  'n_s', 'A_s', 'alpha_s',
                  #'wa', 'w0',
        ):
            cosmo[k] = meta[k]
    boltz.set(cosmo)
    boltz.compute()
    
    # file to save to
    ic_fn = Path(save_dir) / f"ic_filt_nmesh{nmesh:d}.asdf"
    fields_fn = Path(save_dir) / f"fields_nmesh{nmesh:d}.asdf"
    fields_fft_fn = []
    for i in range(len(keynames)):
        fields_fft_fn.append(Path(save_z_dir) / f"advected_{keynames[i]}_field{rsd_str}_fft_nmesh{nmesh:d}.asdf")
    if not logk:
        dk = k_bin_edges[1]-k_bin_edges[0]
    else:
        dk = np.log(k_bin_edges[1]/k_bin_edges[0])
    if n_k_bins == nmesh//2:
        power_ij_fn = Path(save_z_dir) / f"power{rsd_str}_ij_nmesh{nmesh:d}.asdf"
    else:
        power_ij_fn = Path(save_z_dir) / f"power{rsd_str}_ij_nmesh{nmesh:d}_dk{dk:.3f}.asdf"
        print("power file", str(power_ij_fn))
        
    # compute growth factor
    D = boltz.scale_independent_growth_factor(z_this)
    D /= boltz.scale_independent_growth_factor(z_ic)
    Ha = boltz.Hubble(z_this) * 299792.458
    if want_rsd:
        f_growth = boltz.scale_independent_growth_factor_f(z_this)
    else:
        f_growth = 0.
    print("D = ", D)
    
    # field names and growths
    field_D = [1, D, D**2, D**2, D]

    # save the advected fields
    if np.product(np.array([os.path.exists(fn) for fn in fields_fft_fn])):
        fields_fft = []
        for i in range(len(keynames)):
            fields_fft.append(asdf.open(fields_fft_fn[i])['data'])
    else:
        # load density field and displacements
        f = asdf.open(ic_fn)
        disp_pos = np.zeros((nmesh**3, 3), np.float32)
        disp_pos[:, 0] = f['data']['disp_x'][:, :, :].flatten() * D
        disp_pos[:, 1] = f['data']['disp_y'][:, :, :].flatten() * D
        disp_pos[:, 2] = f['data']['disp_z'][:, :, :].flatten() * D * (1 + f_growth)
        print("loaded displacements", disp_pos.dtype)
        f.close(); gc.collect()

        # read in displacements, rescale by D=D(z_this)/D(z_ini)
        grid_x, grid_y, grid_z = np.meshgrid(
            np.arange(nmesh, dtype=np.float32) / nmesh,
            np.arange(nmesh, dtype=np.float32) / nmesh,
            np.arange(nmesh, dtype=np.float32) / nmesh,
            indexing="ij",
        )
        grid_x = grid_x.flatten()
        grid_y = grid_y.flatten()
        grid_z = grid_z.flatten()
        
        disp_pos[:, 0] += grid_x
        disp_pos[:, 1] += grid_y
        disp_pos[:, 2] += grid_z
        del grid_x, grid_y, grid_z; gc.collect()
        
        disp_pos *= Lbox
        disp_pos %= Lbox
        print("box coordinates of the displacements", disp_pos.dtype)
        
        # Initiate fields
        for i in range(len(keynames)):
            
            print(keynames[i])
            if i == 0:
                w = None
            else:
                f = asdf.open(fields_fn)
                w = f['data'][keynames[i]][:, :, :].flatten()
                f.close()
            field_fft = (get_field_fft(disp_pos, Lbox, nmesh, paste, w, W, compensated, interlaced)) 
            del w; gc.collect()
            table = {}
            table[f'{keynames[i]}_Re'] = np.array(field_fft.real, dtype=np.float32)
            table[f'{keynames[i]}_Im'] = np.array(field_fft.imag, dtype=np.float32)
            del field_fft; gc.collect()

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
            del table; gc.collect()
        del disp_pos; gc.collect()
    del W; gc.collect()

    # check if pk_ij exists
    if os.path.exists(power_ij_fn):
        pk_ij_dict = asdf.open(power_ij_fn)['data']
        return pk_ij_dict
    else:
        pk_ij_dict = {}
        pk_ij_dict['k_binc'] = k_binc
        pk_ij_dict['mu_binc'] = mu_binc
    
    # get the box k and mu modes
    k_box, mu_box, k_bin_edges, mu_bin_edges = get_k_mu_box_edges(Lbox, n_perp, n_los, n_k_bins, n_mu_bins, k_hMpc_max, logk)
    
    # initiate final arrays
    pk_auto = []
    pk_cross = []
    for i in range(len(keynames)):        
        for j in range(len(keynames)):
            if i < j: continue
            print("Computing cross-correlation of", keynames[i], keynames[j])

            # load field
            field_fft_i = asdf.open(fields_fft_fn[i])['data']
            field_fft_j = asdf.open(fields_fft_fn[j])['data']

            # compute power spectrum
            pk3d, N3d, binned_poles, Npoles = calc_pk3d(field_fft_i[f'{keynames[i]}_Re']+1j*field_fft_i[f'{keynames[i]}_Im'], Lbox, k_box, mu_box, k_bin_edges, mu_bin_edges, logk, field2_fft=field_fft_j[f'{keynames[j]}_Re']+1j*field_fft_j[f'{keynames[j]}_Im'], poles=poles)
            pk3d *= field_D[i]*field_D[j]
            binned_poles *= field_D[i]*field_D[j]
            pk_auto.append(pk3d)
            pk_ij_dict[f'P_kmu_{keynames[i]}_{keynames[j]}'] = pk3d
            pk_ij_dict[f'N_kmu_{keynames[i]}_{keynames[j]}'] = N3d
            pk_ij_dict[f'P_ell_{keynames[i]}_{keynames[j]}'] = binned_poles
            pk_ij_dict[f'N_ell_{keynames[i]}_{keynames[j]}'] = Npoles
            del field_fft_i, field_fft_j; gc.collect()

    # record power spectra
    header = {}
    header['sim_name'] = sim_name
    header['Lbox'] = Lbox
    header['nmesh'] = nmesh
    header['kcut'] = kcut
    compress_asdf(str(power_ij_fn), pk_ij_dict, header)
    return pk_ij_dict

class ArgParseFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    pass

if __name__ == "__main__":

    # parsing arguments
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=ArgParseFormatter)
    parser.add_argument('--path2config', help='Path to the config file', default=DEFAULTS['path2config'])
    parser.add_argument('--want_rsd', help='Include RSD effects?', action='store_true')
    parser.add_argument('--alt_simname', help='Alternative simulation name')
    args = vars(parser.parse_args())
    main(**args)
