"""
TODO: Fix cosmology here too
Just run twice for rsd and no rsd
CHECK: Syntax for load pk_ij
Might not be necessary to save
"""
import os, gc, sys, time
from pathlib import Path

import numpy as np
import asdf
import h5py
from fast_cksum.cksum_io import CksumWriter
from abacusnbody.metadata import get_meta
from classy import Class

from save_fields import compress_asdf
from abacusnbody.hod.tpcf_corrfunc import get_k_mu_box_edges, get_field_fft, calc_pk3d, get_W_compensated

def advect(tracer_pos, want_rsd, config):

    # read zcv parameters
    zcv_dir = config['zcv_params']['zcv_dir']
    ic_dir = config['zcv_params']['ic_dir']
    cosmo_dir = config['zcv_params']['cosmo_dir']
    nmesh = config['zcv_params']['nmesh']

    # power params
    sim_name = config['sim_params']['sim_name']
    z_this = config['sim_params']['z_mock']
    k_hMpc_max = config['power_params']['k_hMpc_max']
    logk = config['power_params']['logk']
    n_k_bins = config['power_params']['n_k_bins']
    n_mu_bins = config['power_params']['n_mu_bins']
    poles = config['power_params']['poles']
    paste = config['power_params']['paste']
    compensated = config['power_params']['compensated']
    interlaced = config['power_params']['interlaced']
    rsd = config['HOD_params']['want_rsd']
    rsd_str = "_rsd" if rsd else ""
    if compensated:
        W = get_W_compensated(Lbox, nmesh, paste, interlaced)
    else:
        W = None
        
    # define k, mu bins
    n_perp = n_los = nmesh
    k_box, mu_box, k_bin_edges, mu_bin_edges = get_k_mu_box_edges(Lbox, n_perp, n_los, n_k_bins, n_mu_bins, k_hMpc_max, logk)
    k_binc = (k_bin_edges[1:]+k_bin_edges[:-1])*.5
    mu_binc = (mu_bin_edges[1:]+mu_bin_edges[:-1])*.5
    
    # create save directory
    save_dir = Path(zcv_dir) / sim_name
    save_z_dir = save_dir / f"z{z_this:.3f}"
    os.makedirs(save_z_dir, exist_ok=True)
    
    # get a few parameters for the simulation
    meta = get_meta(sim_name, redshift=z_this)
    Lbox = meta['BoxSize']
    z_ic = meta['InitialRedshift']
    k_Ny = np.pi*nmesh/Lbox
    kcut = 0.5*k_Ny
    
    # set up cosmology
    boltz = Class()
    cosmo = {}
    cosmo['output'] = 'mPk mTk'
    cosmo['P_k_max_h/Mpc'] = 20.
    #cosmo['fluid_equation_of_state'] = "CLP"
    cosmo['H0'] = meta['H0']
    cosmo['omega_b'] = meta['omega_b']
    cosmo['omega_cdm'] = meta['omega_cdm']
    cosmo['omega_ncdm'] = meta['omega_ncdm']
    cosmo['N_ncdm'] = meta['N_ncdm']
    cosmo['N_ur'] = meta['N_ur']
    cosmo['n_s'] = meta['n_s']
    #cosmo['wa'] = meta['wa']
    #cosmo['w0'] = meta['w0']
    boltz.set(cosmo)
    boltz.compute()
    
    # file to save to
    ic_fn = Path(save_dir) / f"ic_filt_{sim_name}_nmesh{nmesh:d}.asdf"
    fields_fn = Path(save_dir) / f"fields_{sim_name}_nmesh{nmesh:d}.asdf"
    power_tr_fn = Path(save_z_dir) / f"power{rsd_str}_tr_{sim_name}_nmesh{nmesh:d}.asdf"
    power_ij_fn = Path(save_z_dir) / f"power{rsd_str}_ij_{sim_name}_nmesh{nmesh:d}.asdf"
    
    # create fft field for the tracer
    w = None
    tr_field_fft = get_field_fft(tracer_pos, Lbox, nmesh, paste, w, W, compensated, interlaced)
    del tracer_pos; gc.collect()
    
    # load density field and displacements
    f = asdf.open(ic_fn)
    disp_x = f['data']['disp_x'][:, :, :]
    disp_y = f['data']['disp_y'][:, :, :]
    disp_z = f['data']['disp_z'][:, :, :]
    f.close()

    # compute growth factor
    D = boltz.scale_independent_growth_factor(z_this)
    D /= boltz.scale_independent_growth_factor(z_ic)
    Ha = boltz.Hubble(z_this) * 299792.458
    if want_rsd:
        f = boltz.scale_independent_growth_factor_f(z_this)
    else:
        f = 0.
    print("D", D)
    
    # compute the galaxy auto rsd poles
    pk3d, N3d, binned_poles, Npoles = calc_pk3d(tr_field_fft, Lbox, k_box, mu_box, k_bin_edges, mu_bin_edges, logk, field2_fft=None, poles=poles)
    pk_tr_dict = {}
    pk_tr_dict['k_binc'] = k_binc
    pk_tr_dict['mu_binc'] = mu_binc
    pk_tr_dict['P_kmu_tr_tr'] = pk3d
    pk_tr_dict['N_kmu_tr_tr'] = N3d
    pk_tr_dict['P_ell_tr_tr'] = binned_poles
    pk_tr_dict['N_ell_tr_tr'] = Npoles

    # check if pk_ij exists
    if os.path.exists(power_ij_fn):
        pk_ij_dict = asdf.open(power_ij_fn)['data'][:]
    else:
        pk_ij_dict = {}
        pk_ij_dict['k_binc'] = k_binc
        pk_ij_dict['mu_binc'] = mu_binc
    
    # field names and growths
    field_D = [1, D, D**2, D**2, D]
    keynames = ["1cb", "delta", "delta2", "tidal2", "nabla2"]
    
    # read in displacements, rescale by D=D(z_this)/D(z_ini)
    grid = np.meshgrid(
        np.arange(nmesh),
        np.arange(nmesh),
        np.arange(nmesh),
        indexing="ij",
    )
    disp_x = ((grid[0] / nmesh + D * disp_x) % 1) * Lbox
    disp_y = ((grid[1] / nmesh + D * disp_y) % 1) * Lbox
    disp_z = ((grid[2] / nmesh + D * (1 + f) * disp_z) % 1) * Lbox       
    del grid
    gc.collect()

    # combine into single array
    disp_x = disp_x.flatten()
    disp_y = disp_y.flatten()
    disp_z = disp_z.flatten()
    disp_pos = np.vstack((disp_x, disp_y, disp_z)).T
    del disp_x, disp_y, disp_z; gc.collect()
    
    # Initiate fields
    fields_fft = []
    f = asdf.open(fields_fn)
    for i in range(len(keynames)):
        if i == 0:
            w = None
        else:
            w = f['data'][keynames[i]][:, :, :].flatten()
        fields_fft.append(get_field_fft(disp_pos, Lbox, nmesh, paste, w, W, compensated, interlaced))
        del w; gc.collect()
    del disp_pos; gc.collect()
    f.close()
    
    # initiate final arrays
    pk_auto = []
    pk_cross = []
    pkcounter = 0
    for i in range(len(keynames)):
        print(keynames[i])
        
        # compute power
        pk3d, N3d, binned_poles, Npoles = calc_pk3d(fields_fft[i], Lbox, k_box, mu_box, k_bin_edges, mu_bin_edges, logk, field2_fft=tr_field_fft, poles=poles)
        pk3d *= field_D[i]
        binned_poles *= field_D[i]
        pk_cross.append(pk3d)
        pk_tr_dict[f'P_kmu_{keynames[i]}_tr'] = pk3d
        pk_tr_dict[f'N_kmu_{keynames[i]}_tr'] = N3d
        pk_tr_dict[f'P_ell_{keynames[i]}_tr'] = binned_poles
        pk_tr_dict[f'N_ell_{keynames[i]}_tr'] = Npoles
        
        for j in range(len(keynames)):
            if i < j: continue
            print(keynames[i], keynames[j])

            if not os.path.exists(power_ij_fn):
                pk3d, N3d, binned_poles, Npoles = calc_pk3d(fields_fft[i], Lbox, k_box, mu_box, k_bin_edges, mu_bin_edges, logk, field2_fft=fields_fft[j], poles=poles)
                pk3d *= field_D[i]*field_D[j]
                binned_poles *= field_D[i]*field_D[j]
                pk_auto.append(pk3d)
                pk_ij_dict[f'P_kmu_{keynames[i]}_{keynames[j]}'] = pk3d
                pk_ij_dict[f'N_kmu_{keynames[i]}_{keynames[j]}'] = N3d
                pk_ij_dict[f'P_ell_{keynames[i]}_{keynames[j]}'] = binned_poles
                pk_ij_dict[f'N_ell_{keynames[i]}_{keynames[j]}'] = Npoles
            else:
                print("Skip since already saved")
                
    print(pk_auto, pk_cross)
    header = {}
    header['sim_name'] = sim_name
    header['Lbox'] = Lbox
    header['nmesh'] = nmesh
    header['kcut'] = kcut
    if not os.path.exists(power_ij_fn):
        compress_asdf(str(power_ij_fn), pk_ij_dict, header)
    compress_asdf(str(power_tr_fn), pk_tr_dict, header)
    
    return pk_tr_dict, pk_ij_dict
