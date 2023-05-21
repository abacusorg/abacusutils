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

from scipy.fft import ifftn, fftfreq
from scipy.special import legendre
from ..power_spectrum import mean2d_numba_seq

DEFAULTS = {'path2config': 'config/abacus_hod.yaml'}

def get_r_mu_box(L_hMpc, n_xy, n_z):
    """
    Compute the size of the k vector and mu for each mode. Assumes z direction is LOS
    """
    # cell width in (x,y) directions (Mpc/h)
    d_xy = L_hMpc / n_xy
    #k_xy = (fftfreq(n_xy, d=d_xy) * 2. * np.pi).astype(np.float32)
    r_xy = (np.arange(n_xy) * d_xy).astype(np.float32)
    
    # cell width in z direction (Mpc/h)
    d_z = L_hMpc / n_z
    #r_z = (fftfreq(n_z, d=d_z) * 2. * np.pi).astype(np.float32)
    r_z = (np.arange(n_z) * d_z).astype(np.float32)
    
    # h/Mpc
    x = r_xy[:, np.newaxis, np.newaxis]
    y = r_xy[np.newaxis, :, np.newaxis]
    z = r_z[np.newaxis, np.newaxis, :]
    x[x > L_hMpc/2.] -= L_hMpc
    y[y > L_hMpc/2.] -= L_hMpc
    z[z > L_hMpc/2.] -= L_hMpc
    r_box = np.sqrt(x**2 + y**2 + z**2)

    # define mu box
    mu_box = z/np.ones_like(r_box)
    mu_box[r_box > 0.] /= r_box[r_box > 0.]
    mu_box[r_box == 0.] = 0.
    
    # I believe the definition is that and it allows you to count all modes (agrees with nbodykit)
    mu_box = np.abs(mu_box)
    print("r box, mu box", r_box.dtype, mu_box.dtype)

    r_box = r_box.flatten()
    mu_box = mu_box.flatten()    
    return r_box, mu_box

def pk_to_xi(power_tr_fn, r_bins, poles=[0, 2, 4], key='P_k3D_tr_tr'):
    
    # apply fourier transform to get 3D correlation
    #power_tr_fn = Path(save_z_dir) / f"power{rsd_str}_tr_tr_nmesh{nmesh:d}.asdf"
    f = asdf.open(power_tr_fn)
    Lbox = f['header']['Lbox']
    nmesh = f['header']['nmesh']
    n_perp = n_los = nmesh
    Pk = f['data'][key]
    Xi = ifftn(Pk).real.flatten()
    del Pk; gc.collect()

    # define r bins
    #r_bins = np.linspace(0., 200., 201)
    r_binc = (r_bins[1:]+r_bins[:-1])*.5

    # get r and mu box
    r_box, mu_box = get_r_mu_box(Lbox, n_perp, n_los)
    r_box = r_box.astype(np.float32)
    mu_box = mu_box.astype(np.float32)
    
    # bin into xi_ell(r)
    binned_poles = []
    Npoles = []
    ranges = ((r_bins[0], r_bins[-1]), (0., 1.+1.e-6)) # not doing this misses the +/- 1 mu modes in the first few bins
    nbins2d = (len(r_bins)-1, 1)
    nbins2d = np.asarray(nbins2d).astype(np.int64)
    ranges = np.asarray(ranges).astype(np.float64)
    for i in range(len(poles)):
        Ln = legendre(poles[i])
        print(r_box.dtype, Xi.dtype)
        binned_pole, Npole = mean2d_numba_seq(np.array([r_box, mu_box]), bins=nbins2d, ranges=ranges, logk=False, weights=Xi*Ln(mu_box)*(2.*poles[i]+1.))
        binned_poles.append(binned_pole)
    Npoles.append(Npole)
    Npoles = np.array(Npoles)
    binned_poles = np.array(binned_poles)

    xi_dict = {}
    xi_dict['r_binc'] = r_binc
    xi_dict['Npoles'] = Npoles
    xi_dict['binned_poles'] = binned_poles
    return xi_dict

def main(path2config, want_rsd=False, alt_simname=None, want_zcv=False, want_lcv=False):

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

    if want_lcv or want_zcv:
        assert want_zcv and want_lcv == False
        if want_zcv:
            power_cv_tr_fn = Path(save_z_dir) / f"power{rsd_str}_ZCV_tr_nmesh{nmesh:d}.asdf"
        elif want_lcv:
            power_cv_tr_fn = Path(save_z_dir) / f"power{rsd_str}_LCV_tr_nmesh{nmesh:d}.asdf"
        f = asdf.open(power_cv_tr_fn)
        assert f['header']['nmesh'] == nmesh
        assert f['header']['kcut'] == kcut
        if want_zcv:
            Pk = f['data'][f'P_k3D_tr_tr_zcv']
        elif want_lcv:
            Pk = f['data'][f'P_k3D_tr_tr_lcv']
        
    else:
        # load tracer field and compute power
        tr_field_fft_fn = Path(save_z_dir) / f"tr_field{rsd_str}_fft_nmesh{nmesh:d}.asdf"

        if False: #os.path.exists(tr_field_fft_fn): # TESTING
            # load field
            tr_field_fft = asdf.open(tr_field_fft_fn)['data']
            tr_field_fft = tr_field_fft[f'tr_field_fft_Re']+1j*tr_field_fft['tr_field_fft_Im']
        else:
            print("brat tf")
            if want_rsd:
                #tracer_pos = np.load("/global/homes/b/boryanah/zcv/tracer_pos_rsd_10.npy")
                tracer_pos = np.load("/global/homes/b/boryanah/zcv/LRG_rsd_pos_1.npy") # TESTING
                print("tuk li sme")
            else:
                tracer_pos = np.load("/global/homes/b/boryanah/zcv/tracer_pos_10.npy")
            print("min/max tracer pos", tracer_pos.min(), tracer_pos.max(), tracer_pos.shape)

            # get the window function of TSC/CIC
            if compensated:
                W = get_W_compensated(Lbox, nmesh, paste, interlaced)
            else:
                W = None
            w = None

            tr_field_fft = get_field_fft(tracer_pos, Lbox, nmesh, paste, w, W, compensated, interlaced)
            del tracer_pos; gc.collect()

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

        # get 3D power spectrum
        Pk = (tr_field_fft*np.conj(tr_field_fft)).real

        # normalize since we do that for the tracer
        #Pk *= nmesh**6
        #Pk *= Lbox**3
        Pk *= nmesh**3

        apply_apod = True
        if apply_apod:
            # apply apodization
            k = (fftfreq(nmesh, d=Lbox/nmesh) * 2. * np.pi).astype(np.float32)# not flattened
            #R = 4. # Mpc/h
            #W = np.exp(-(k*R)**2/2.)
            k0 = 0.6 #0.5*np.pi*nmesh/Lbox #0.618
            dk = 0.1 #0.167
            W = 0.5 * (1. - np.tanh((k - k0)/dk))
            Pk *= W[:, np.newaxis, np.newaxis]
            Pk *= W[np.newaxis, :, np.newaxis]
            Pk *= W[np.newaxis, np.newaxis, :]
            del k, W; gc.collect()

        
    # apply fourier transform to get 3D correlation
    Xi = ifftn(Pk).real.flatten()
    del Pk; gc.collect()

    # define r bins
    r_bins = np.linspace(0., 200., 201)
    #r_bins = np.linspace(0., 200., 101)
    r_binc = (r_bins[1:]+r_bins[:-1])*.5

    # get r and mu box
    r_box, mu_box = get_r_mu_box(Lbox, n_perp, n_los)
    r_box = r_box.astype(np.float32)
    mu_box = mu_box.astype(np.float32)
    
    # bin into xi_ell(r)
    binned_poles = []
    Npoles = []
    ranges = ((r_bins[0], r_bins[-1]), (0., 1.+1.e-6)) # not doing this misses the +/- 1 mu modes in the first few bins
    nbins2d = (len(r_bins)-1, 1)
    nbins2d = np.asarray(nbins2d).astype(np.int64)
    ranges = np.asarray(ranges).astype(np.float64)
    for i in range(len(poles)):
        Ln = legendre(poles[i])
        print(r_box.dtype, Xi.dtype)
        binned_pole, Npole = mean2d_numba_seq(np.array([r_box, mu_box]), bins=nbins2d, ranges=ranges, logk=False, weights=Xi*Ln(mu_box)*(2.*poles[i]+1.))
        binned_poles.append(binned_pole)
    Npoles.append(Npole)
    Npoles = np.array(Npoles)
    binned_poles = np.array(binned_poles)

    # save
    np.savez(f"/global/homes/b/boryanah/zcv/xi_tr_tr_{sim_name}_{len(r_binc):d}_nmesh{nmesh:d}.npz", r_binc=r_binc, Npoles=Npoles, binned_poles=binned_poles)    

    # TESTING!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # apply fourier transform to get 3D correlation
    power_tr_fn = Path(save_z_dir) / f"power{rsd_str}_tr_tr_nmesh{nmesh:d}.asdf"
    Pk_raw = asdf.open(power_tr_fn)['data'][f'P_k3D_tr_tr']
    Xi = ifftn(Pk_raw).real.flatten()
    del Pk_raw; gc.collect()

    # define r bins
    r_bins = np.linspace(0., 200., 201)
    #r_bins = np.linspace(0., 200., 101)
    r_binc = (r_bins[1:]+r_bins[:-1])*.5

    # get r and mu box
    r_box, mu_box = get_r_mu_box(Lbox, n_perp, n_los)
    r_box = r_box.astype(np.float32)
    mu_box = mu_box.astype(np.float32)
    
    # bin into xi_ell(r)
    binned_poles = []
    Npoles = []
    ranges = ((r_bins[0], r_bins[-1]), (0., 1.+1.e-6)) # not doing this misses the +/- 1 mu modes in the first few bins
    nbins2d = (len(r_bins)-1, 1)
    nbins2d = np.asarray(nbins2d).astype(np.int64)
    ranges = np.asarray(ranges).astype(np.float64)
    for i in range(len(poles)):
        Ln = legendre(poles[i])
        print(r_box.dtype, Xi.dtype)
        binned_pole, Npole = mean2d_numba_seq(np.array([r_box, mu_box]), bins=nbins2d, ranges=ranges, logk=False, weights=Xi*Ln(mu_box)*(2.*poles[i]+1.))
        binned_poles.append(binned_pole)
    Npoles.append(Npole)
    Npoles = np.array(Npoles)
    binned_poles = np.array(binned_poles)

    # save
    np.savez(f"/global/homes/b/boryanah/zcv/xi_tr_tr_{len(r_binc):d}_nmesh{nmesh:d}_raw.npz", r_binc=r_binc, Npoles=Npoles, binned_poles=binned_poles)    

    
class ArgParseFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    pass

if __name__ == "__main__":

    # parsing arguments
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=ArgParseFormatter)
    parser.add_argument('--path2config', help='Path to the config file', default=DEFAULTS['path2config'])
    parser.add_argument('--want_rsd', help='Include RSD effects?', action='store_true')
    parser.add_argument('--want_zcv', help='Load reduced ZCV field?', action='store_true')
    parser.add_argument('--want_lcv', help='Load reduced LCV field?', action='store_true')
    parser.add_argument('--alt_simname', help='Alternative simulation name')
    args = vars(parser.parse_args())
    main(**args)
