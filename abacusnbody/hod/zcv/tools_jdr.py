"""
Tools for applying variance reduction (ZCV) by (mostly) Joe DeRose.
"""
import sys
import os
from pathlib import Path
sys.path.append('/global/homes/b/boryanah/repos/ZeNBu/') # tuks remove

import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from scipy.interpolate import splrep, splev
from scipy.signal import savgol_filter
import asdf

from classy import Class
from zenbu_rsd import Zenbu_RSD
from zenbu import Zenbu
from abacusnbody.metadata import get_meta
from abacusnbody.hod.power_spectrum import get_k_mu_edges

from numba import jit

def get_spectra_from_fields(fields1, fields2, neutrinos=True):
    spectra = []
    for i, fi in enumerate(fields1):
        for j, fj in enumerate(fields2):
            if (i<j) | (neutrinos & (i==1) & (j==0)): continue
            spectra.append((fi, fj))
                      
    return spectra

def combine_real_space_spectra(k, spectra, bias_params, cross=False, numerical_nabla=False):

    pkvec = np.zeros((14, spectra.shape[1]))
    if numerical_nabla:
        pkvec[...] = spectra[:14]
    else:
        pkvec[:10, ...] = spectra[:10]
        # IDs for the  ~ -k^2 <1, X> approximation.
        nabla_idx = [0, 1, 3, 6]

        # Higher derivative terms
        pkvec[10:, ...] = -k[np.newaxis, :]**2 * pkvec[nabla_idx, ...]

    b1, b2, bs, bk2, sn = bias_params
    if not cross:
        bterms = [1,
                  2*b1, b1**2,
                  b2, b2*b1, 0.25*b2**2,
                  2*bs, 2*bs*b1, bs*b2, bs**2,
                  2*bk2, 2*bk2*b1, bk2*b2, 2*bk2*bs]
    else:
        # hm correlations only have one kind of <1,delta_i> correlation
        bterms = [1,
                  b1, 0,
                  b2/2, 0, 0,
                  bs, 0, 0, 0,
                  bk2, 0, 0, 0]
    p = np.einsum('b, bk->k', bterms, pkvec)
    if not cross:
        p += sn
    return p    

def combine_rsd_spectra(k, spectra_poles, bias_params, ngauss=3):

    pkvec = np.zeros((17, spectra_poles.shape[1], spectra_poles.shape[2]))
    pkvec[:10, ...] = spectra_poles[:10,...]
      
    b1,b2,bs,alpha0,alpha2,alpha4,alpha6,sn,sn2,sn4 = bias_params
    bias_monomials = np.array([1, 2*b1, b1**2, b2, b1*b2, 0.25*b2**2, 2*bs, 2*b1*bs, b2*bs, bs**2, alpha0, alpha2, alpha4,alpha6,sn,sn2,sn4])
    
    nus, ws = np.polynomial.legendre.leggauss(2*ngauss)
    nus_calc = nus[0:ngauss]
    nus = nus[0:ngauss]
    ws = ws[:ngauss]
    n_nu = ngauss
    leggauss = True
        
    L0 = np.polynomial.legendre.Legendre((1))(nus)
    L2 = np.polynomial.legendre.Legendre((0,0,1))(nus)
    L4 = np.polynomial.legendre.Legendre((0,0,0,0,1))(nus)        
        
    pk_stoch = np.zeros((3, n_nu, spectra_poles.shape[2]))    
    
    pk_stoch[0,:,:] = 1
    pk_stoch[1,:,:] = k[np.newaxis,:]**2 * nus[:, np.newaxis]**2 
    pk_stoch[2,:,:] = k[np.newaxis,:]**4 * nus[:, np.newaxis]**4
    
    pkvec[14:,0,...] = 0.5 * np.sum((ws*L0)[np.newaxis,:ngauss,np.newaxis]*pk_stoch,axis=1)
    pkvec[14:,1,...] = 2.5 * np.sum((ws*L2)[np.newaxis,:ngauss,np.newaxis]*pk_stoch,axis=1)
    pkvec[14:,2,...] = 4.5 * np.sum((ws*L4)[np.newaxis,:ngauss,np.newaxis]*pk_stoch,axis=1) 
    
    p0 = np.sum(bias_monomials[:,np.newaxis] * pkvec[:,0,:], axis=0)
    p2 = np.sum(bias_monomials[:,np.newaxis] * pkvec[:,1,:], axis=0)
    p4 = np.sum(bias_monomials[:,np.newaxis] * pkvec[:,2,:], axis=0)
    return p0, p2, p4

def combine_rsd_cross_spectra(k, spectra_poles, bias_params, ngauss=3):
    pkvec = np.zeros((5, spectra_poles.shape[1], spectra_poles.shape[2]))
    pkvec[:5, ...] = spectra_poles[:5,...]
    b1,b2,bs,alpha0,alpha2,alpha4,alpha6,sn,sn2,sn4 = bias_params
    bias_monomials = np.array([1, b1, 0.5 * b2, bs, alpha0])
    p0 = np.sum(bias_monomials[:,np.newaxis] * pkvec[:,0,:], axis=0) #+ sn
    p2 = np.sum(bias_monomials[:,np.newaxis] * pkvec[:,1,:], axis=0) #+ sn2
    p4 = np.sum(bias_monomials[:,np.newaxis] * pkvec[:,2,:], axis=0) #+ sn4
    return p0, p2, p4

def combine_real_space_cross_spectra(k, spectra_poles, bias_params):
    pkvec = np.zeros((5, spectra_poles.shape[1]))
    pkvec[:5, ...] = spectra_poles[:5,...]
    b1,b2,bs,bk,sn = bias_params
    bias_monomials = np.array([1, b1, 0.5 * b2, bs, bk])
    pk = np.sum(bias_monomials[:,np.newaxis] * pkvec[:,:], axis=0) + sn
    return pk


def combine_spectra(k, spectra, bias_params, rsd=False):
    if rsd:
        p0, p2, p4 = combine_rsd_spectra(k, spectra, bias_params)
        pk = np.stack([p0, p2, p4])
    else:
        pk = combine_real_space_spectra(k, spectra, bias_params)
    return pk

def combine_cross_spectra(k, spectra, bias_params, rsd=False):
    if rsd:
        p0, p2, p4 = combine_rsd_cross_spectra(k, spectra, bias_params)
        pk = np.stack([p0, p2, p4])
    else:
        pk = combine_real_space_cross_spectra(k, spectra, bias_params)
    return pk

@jit(nopython=True)
def meshgrid(x, y, z):
    xx = np.empty(shape=(y.size, x.size, z.size), dtype=x.dtype)
    yy = np.empty(shape=(y.size, x.size, z.size), dtype=y.dtype)
    zz = np.empty(shape=(y.size, x.size, z.size), dtype=z.dtype)
    for i in range(y.size):
        for j in range(x.size):
            for k in range(z.size):
                xx[i,j,k] = x[i]  # change to x[k] if indexing xy
                yy[i,j,k] = y[j]  # change to y[j] if indexing xy
                zz[i,j,k] = z[k]  # change to z[i] if indexing xy
    return zz, yy, xx

def zenbu_spectra(k, z, cfg, kin, pin, pkclass=None, N=2700, jn=15, rsd=True, nmax=6, ngauss=6):
    
    if pkclass==None:
        pkclass = Class()
        pkclass.set(cfg["Cosmology"])
        pkclass.compute()
        
    cutoff = cfg['surrogate_gaussian_cutoff']
    cutoff = float(cfg['surrogate_gaussian_cutoff'])
        
    Dthis = pkclass.scale_independent_growth_factor(z)
    Dic = pkclass.scale_independent_growth_factor(cfg['z_ic'])
    f = pkclass.scale_independent_growth_factor_f(z)
    
    if rsd:
        lptobj, p0spline, p2spline, p4spline, pspline = _lpt_pk(kin, pin*(Dthis/Dic)**2, 
                                                               f, cutoff=cutoff, 
                                                               third_order=False, one_loop=False,
                                                               jn=jn, N=N, nmax=nmax, ngauss=ngauss)
        pk_zenbu = pspline(k)
        
    else:
        pspline, lptobj = _realspace_lpt_pk(kin, pin*(Dthis/Dic)**2)  
        pk_zenbu = pspline(k)[1:]

    return pk_zenbu[:11], lptobj


def _lpt_pk(k, p_lin, f, cleftobj=None,
            third_order=True, one_loop=True, cutoff=np.pi*700/525.,
            jn=15, N=2700, nmax=8, ngauss=3):
    '''
    Returns a spline object which computes the cleft component spectra.
    Computed either in "full" CLEFT or in "k-expanded" CLEFT (kecleft)
    which allows for faster redshift dependence.
    Args:
        k: array-like
            Array of wavevectors to compute power spectra at (in h/Mpc).
        p_lin: array-like
            Linear power spectrum to produce velocileptors predictions for.
            If kecleft==True, then should be for z=0, and redshift evolution is
            handled by passing the appropriate linear growth factor to D.
        D: float
            Linear growth factor. Only required if kecleft==True.
        kecleft: bool
            Whether to use kecleft or not. Setting kecleft==True
            allows for faster computation of spectra keeping cosmology
            fixed and varying redshift if the cleftobj from the
            previous calculation at the same cosmology is provided to
            the cleftobj keyword.
    Returns:
        cleft_aem : InterpolatedUnivariateSpline
            Spline that computes basis spectra as a function of k.
        cleftobt: CLEFT object
            CLEFT object used to compute basis spectra.
    '''
    
    
    lpt  = Zenbu_RSD(k, p_lin, jn=jn, N=N, cutoff=cutoff)
    lpt.make_pltable(f, kv=k, nmax=nmax, ngauss=ngauss)
        
    p0table = lpt.p0ktable
    p2table = lpt.p2ktable
    p4table = lpt.p4ktable
    
    pktable = np.zeros((len(p0table), 3, p0table.shape[-1]))
    pktable[:,0,:] = p0table
    pktable[:,1,:] = p2table
    pktable[:,2,:] = p4table

    pellspline = interp1d(k, pktable.T, fill_value='extrapolate')#, kind='cubic')
    p0spline = interp1d(k, p0table.T, fill_value='extrapolate')#, kind='cubic')
    p2spline = interp1d(k, p2table.T, fill_value='extrapolate')#, kind='cubic')
    p4spline = interp1d(k, p4table.T, fill_value='extrapolate')#, kind='cubic')


    return lpt, p0spline, p2spline, p4spline, pellspline 


def _realspace_lpt_pk(k, p_lin, D=None, cleftobj=None, cutoff=np.pi*700/525.):
    '''
    Returns a spline object which computes the cleft component spectra.
    Computed either in "full" CLEFT or in "k-expanded" CLEFT (kecleft)
    which allows for faster redshift dependence.
    Args:
        k: array-like
            Array of wavevectors to compute power spectra at (in h/Mpc).
        p_lin: array-like
            Linear power spectrum to produce velocileptors predictions for.
            If kecleft==True, then should be for z=0, and redshift evolution is
            handled by passing the appropriate linear growth factor to D.
        D: float
            Linear growth factor. Only required if kecleft==True.
        kecleft: bool
            Whether to use kecleft or not. Setting kecleft==True
            allows for faster computation of spectra keeping cosmology
            fixed and varying redshift if the cleftobj from the
            previous calculation at the same cosmology is provided to
            the cleftobj keyword.
    Returns:
        cleft_aem : InterpolatedUnivariateSpline
            Spline that computes basis spectra as a function of k.
        cleftobt: CLEFT object
            CLEFT object used to compute basis spectra.
    '''        

    zobj = Zenbu(k, p_lin, cutoff=cutoff, N=3000, jn=15)
    zobj.make_ptable(kvec=k)
    cleftpk = zobj.pktable.T
    cleftobj = zobj
    cleftspline = interp1d(cleftpk[0], cleftpk, fill_value='extrapolate')#, kind='cubic')

    return cleftspline, cleftobj#, cleftpk

def reduce_variance_tt(k, pk_nn, pk_ij_zn, pk_ij_zz, cfg, z, bias_vec, kth, p_m_lin, window=None,
                       kin=None, kout=None, s_ell=[0.1, 10, 100],
                       sg_window=21, rsd=True, neutrinos=True,
                       pkclass=None, pk_ij_zenbu=None, lptobj=None,
                       exact_window=True, win_fac=1):

    if pk_ij_zenbu is None:
        if kin is not None:
            pk_ij_zenbu, lptobj = zenbu_spectra(kin, z, cfg, kth, p_m_lin, pkclass=pkclass, rsd=rsd)
        else:
            pk_ij_zenbu, lptobj = zenbu_spectra(k, z, cfg, kth, p_m_lin, pkclass=pkclass, rsd=rsd)
        p0table = lptobj.p0ktable  # [0, 2, 4]
        p2table = lptobj.p2ktable
        p4table = lptobj.p4ktable
        
    if exact_window & rsd:
        print('Convolving theory with exact window')
        
        lbox = cfg['lbox']
        nmesh_win = int(cfg['nmesh_in'] / win_fac)
        
        if kout is None:
            dk = 2 * np.pi / lbox
            kmax = np.pi*cfg['nmesh_in']/lbox + dk/2
            kout = np.arange(dk, np.pi*cfg['nmesh_in']/lbox + dk/2, dk)        
            assert(len(k) == (len(kout)-1))

        window = None
        pell_conv_list = []

        for i in range(pk_ij_zenbu.shape[0]):
            pell_in = [p0table[:,i], p2table[:,i], p4table[:,i]]  # [0, 2, 4]
            pell_conv_i, keff = conv_theory_window_function(nmesh_win, lbox, kout, pell_in, kth)
            pell_conv_list.append(pell_conv_i.reshape(3,-1))

        pk_ij_zenbu_conv = np.stack(pell_conv_list)
        temp = np.zeros_like(pk_ij_zenbu) 
        pk_ij_zenbu[:,:,:pk_ij_zenbu_conv.shape[-1]] = pk_ij_zenbu_conv # if win fac is > 1 get rid of nans
        pk_ij_zenbu[:,:,pk_ij_zenbu_conv.shape[-1]:] = 0
        
    pk_nn_hat, pk_nn_betasmooth, pk_nn_betasmooth_nohex, pk_nn_beta1, beta, beta_damp, beta_smooth, beta_smooth_nohex, pk_zz, pk_zenbu, r_zt, r_zt_smooth, r_zt_smooth_nohex, pk_zn = compute_beta_and_reduce_variance_tt(k, pk_nn, pk_ij_zn, pk_ij_zz, pk_ij_zenbu, bias_vec, window=window, kin=kin, kout=kout, s_ell=s_ell, rsd=rsd, k0=0.618, dk=0.167, sg_window=sg_window, poles=cfg['poles'])
    return pk_nn_hat, pk_nn_betasmooth, pk_nn_betasmooth_nohex, pk_nn_beta1, beta, beta_damp, beta_smooth, beta_smooth_nohex, pk_zz, pk_zenbu, r_zt, r_zt_smooth, r_zt_smooth_nohex, pk_zn, pk_ij_zenbu, pkclass

def compute_beta_and_reduce_variance_tt(k, pk_nn, pk_ij_zn, pk_ij_zz, pk_ij_zb, 
                                        bias_vec, window=None, kin=None, kout=None, 
                                        s_ell=[0.1, 10, 100], sg_window=21, rsd=False,
                                        k0=0.618, dk=0.167, beta1_k=0.05, poles=[0, 2, 4]):
    
    fields_z = ['1', 'd', 'd2', 's', 'n2']
    fields_zenbu = ['1', 'd', 'd2', 's']

    # parsing
    component_spectra_zz = get_spectra_from_fields(fields_z, fields_z, neutrinos=False)
    pk_ij_zz_dict = dict(zip(component_spectra_zz, pk_ij_zz))
    nspec = len(bias_vec)

    if rsd:
        if len(bias_vec)<11:
            bias_vec = np.hstack([bias_vec, np.zeros(11-len(bias_vec))])
            
    else:
        if len(bias_vec)<6:
            bias_vec = np.hstack([bias_vec, np.zeros(6-len(bias_vec))])        
    
    # first element of bias vec should always be one, so don't pass this 
    # to our usual component spectra summation functions

    pk_zz = combine_spectra(k, pk_ij_zz, bias_vec[1:], rsd=rsd)
    if kin is not None:
        pk_zenbu = combine_spectra(kin, pk_ij_zb, bias_vec[1:], rsd=rsd)
    else:
        pk_zenbu = combine_spectra(k, pk_ij_zb, bias_vec[1:], rsd=rsd)
        
    pk_zn = combine_cross_spectra(k, pk_ij_zn, bias_vec[1:], rsd=rsd)

    if rsd:
        cov_zn = np.stack([multipole_cov(pk_zn, ell) for ell in poles])
        var_zz = np.stack([multipole_cov(pk_zz, ell) for ell in poles])
        var_nn = np.stack([multipole_cov(pk_nn, ell) for ell in poles])
    else:
        cov_zn = 2 * pk_zn ** 2
        var_zz = 2 * pk_zz ** 2
        var_nn = 2 * pk_nn ** 2
    
    beta = cov_zn / var_zz
    beta_damp = 1/2 * (1 - np.tanh((k - k0)/dk)) * beta

    r_zt = cov_zn / np.sqrt(var_zz * var_nn)
    beta_damp = np.atleast_2d(beta_damp)
    beta_damp[beta_damp != beta_damp] = 0
    beta_damp[:,:k.searchsorted(beta1_k)] = 1
    
    r_zt = np.atleast_2d(r_zt)
    r_zt[r_zt != r_zt] = 0

    beta_smooth = np.zeros_like(beta_damp)
    beta_smooth_nohex = np.zeros_like(beta_damp)
    
    for i in range(beta_smooth.shape[0]):
        # TESTING kinda ugly
        try:
            beta_smooth[i,:] = savgol_filter(beta_damp.T[:,i], sg_window, 3) #splev(k, betaspl)
        except: # only when testing
            beta_smooth[i,:] = savgol_filter(beta_damp.T[:,i], 3, 2)
        if i!=2:
            try:
                beta_smooth_nohex[i,:] = savgol_filter(beta_damp.T[:,i], sg_window, 3)#splev(k, betaspl)
            except:
                beta_smooth_nohex[i,:] = savgol_filter(beta_damp.T[:,i], 3, 2)
        else:
            beta_smooth_nohex[i,:] = 1
            
    r_zt_smooth = np.zeros_like(r_zt)
    r_zt_smooth_nohex = np.zeros_like(r_zt)

    for i in range(r_zt.shape[0]):
        spl = splrep(k, r_zt.T[:,i], s=s_ell[i])
        r_zt_smooth[i,:] = splev(k, spl) 
        if i<1:
            r_zt_smooth_nohex[i,:] = splev(k, spl)
        else:
            r_zt_smooth_nohex[i,:] = 1
            
    if (window is not None):

        pk_zenbu = np.hstack(pk_zenbu)
        if rsd:
            pk_zenbu = np.dot(window.T, pk_zenbu).reshape(len(poles),-1)
        else:
            pk_zenbu = np.dot(window[:window.shape[0]//len(poles), :window.shape[1]//len(poles)].T, pk_zenbu).reshape(pk_zz.shape)
    
    pk_nn_hat = pk_nn - beta_damp * (pk_zz - pk_zenbu) # pk_zz has shape of measurement nmesh/2
    pk_nn_betasmooth = pk_nn - beta_smooth * (pk_zz - pk_zenbu) # joe says beta needs to be smooth
    pk_nn_betasmooth_nohex = pk_nn - beta_smooth * (pk_zz - pk_zenbu) # same as above
    pk_nn_beta1 = pk_nn - (pk_zz - pk_zenbu)
        
    return pk_nn_hat, pk_nn_betasmooth, pk_nn_betasmooth_nohex, pk_nn_beta1, beta, beta_damp, beta_smooth, beta_smooth_nohex, pk_zz, pk_zenbu, r_zt, r_zt_smooth, r_zt_smooth_nohex, pk_zn


def multipole_cov(pell, ell):
    
    if ell==0:
        cov = 2 * pell[0,:]**2 + 2/5 * pell[1,:]**2 + 2/9 * pell[2,:]**2
        
    elif ell==2:
        cov = 2/5 * pell[0,:]**2 + 6/35 * pell[1,:]**2 + 3578/45045 * pell[2,:]**2 \
               + 8/35 * pell[0,:] * pell[1,:] + 8/35 * pell[0,:] * pell[2,:] + 48/385 * pell[1,:] * pell[2,:]

    elif ell==4:
        cov = 2/9 * pell[0,:]**2 + 3578/45045 * pell[1,:]**2 + 1058/17017 * pell[2,:]**2 \
               + 80/693 * pell[0,:] * pell[1,:] + 72/1001 * pell[0,:] * pell[2,:] + 80/1001 * pell[1,:] * pell[2,:]
        
    return cov

@jit(nopython=True)
def conv_theory_window_function(nmesh, lbox, kout, plist, kth):
    """Exactly convolve the periodic box window function, without any 
        bin averaging uncertainty by evaluating a theory power spectrum
        at the k modes in the box.

    Args:
        nmesh (int): Size of the mesh used for power spectrum measurement
        lbox (float): Box length
        kout (np.array): k bins used for power spectrum measurement
        plist (list of np.arrays): List of theory multipoles evaluated at kth
        kth (np.array): k values that theory is evaluated at
        
    Returns:
        pell_conv : window convolved theory prediction
        keff: Effective k value of each output k bin. 
        
    """
    
    kvals = np.zeros(nmesh, dtype=np.float32)
    kvals[:nmesh//2] = np.arange(0, 2 * np.pi * nmesh / lbox / 2, 2 * np.pi / lbox, dtype=np.float32)
    kvals[nmesh//2:] = np.arange(-2 * np.pi * nmesh / lbox / 2, 0, 2 * np.pi / lbox, dtype=np.float32)
    kvalsr = np.arange(0, 2 * np.pi * nmesh / lbox / 2, 2 * np.pi / lbox, dtype=np.float32) 
    kx, ky, kz = meshgrid(kvals, kvals, kvalsr)    
    knorm = np.sqrt(kx**2 + ky**2 + kz**2)
    mu = kz / knorm
    mu[0,0,0] = 0
    
    ellmax = 3
    
    nkout = len(kout) - 1
    idx_o = np.digitize(knorm, kout) - 1
    nmodes_out = np.zeros(nkout * 3)
  
    pell_conv = np.zeros((nkout * 3), dtype=np.float32)
    keff = np.zeros(nkout, dtype=np.float32)
    
    ellmax_in = len(plist)

    pells = []
    for i in range(ellmax_in):
        pells.append(np.interp(knorm, kth, plist[i]))
        pells[i][0,0,0] = 0
        
    L0 = np.ones_like(mu, dtype=np.float32)
    L2 = (3 * mu**2 - 1) / 2
    L4 = (35 * mu**4 - 30 * mu**2 + 3) / 8
    L6 = (231 * mu**6 - 315 * mu**4 + 105 * mu**2 - 5) / 16
    
#    pk = (p0 * L0 + p2 * L2 + p4 * L4)

    legs = [L0, L2, L4, L6]
    pref = [(2 * (2 * i) + 1) for i in range(ellmax_in)]

    for i in range(kx.shape[0]):
        for j in range(kx.shape[1]):
            for k in range(kx.shape[2]):
                if (idx_o[i,j,k]>=nkout): 
                    pass
                else:
                    if k==0:
                        nmodes_out[idx_o[i,j,k]::nkout] += 1
                        keff[idx_o[i,j,k]] += knorm[i,j,k]
                    else:
                        nmodes_out[idx_o[i,j,k]::nkout] += 2
                        keff[idx_o[i,j,k]] += 2 * knorm[i,j,k]      
                    for ell in range(ellmax):
                        for ellp in range(ellmax_in):
                            if k!=0:
                                pell_conv[int(ell * nkout) + int(idx_o[i,j,k])] += 2 * pref[ell] * legs[ell][i,j,k] * legs[ellp][i,j,k] * pells[ellp][i,j,k]
                            else:
                                pell_conv[int(ell * nkout) + int(idx_o[i,j,k])] += pref[ell] * legs[ell][i,j,k] * legs[ellp][i,j,k] * pells[ellp][i,j,k]

    norm_out = 1/nmodes_out
    norm_out[nmodes_out==0] = 0
    pell_conv = pell_conv * norm_out
    keff = keff * norm_out[:nkout]
    
    return pell_conv, keff 

@jit(nopython=True)
def periodic_window_function(nmesh, lbox, kout, kin, k2weight=True):
    """Returns matrix appropriate for convolving a finely evaluated
    theory prediction with the 

    Args:
        nmesh (int): Size of the mesh used for power spectrum measurement
        lbox (float): Box length
        kout (np.array): k bins used for power spectrum measurement
        kin (np.array): . Defaults to None.
        k2weight (bool, optional): _description_. Defaults to True.

    Returns:
        window : np.dot(window, pell_th) gives convovled theory
        keff: Effective k value of each output k bin.
    """
    
    kvals = np.zeros(nmesh, dtype=np.float32)
    kvals[:nmesh//2] = np.arange(0, 2 * np.pi * nmesh / lbox / 2, 2 * np.pi / lbox, dtype=np.float32)
    kvals[nmesh//2:] = np.arange(-2 * np.pi * nmesh / lbox / 2, 0, 2 * np.pi / lbox, dtype=np.float32)

    kvalsr = np.arange(0, 2 * np.pi * nmesh / lbox / 2, 2 * np.pi / lbox, dtype=np.float32) 
    kx, ky, kz = meshgrid(kvals, kvals, kvalsr)    
    knorm = np.sqrt(kx**2 + ky**2 + kz**2)
    mu = kz / knorm
    mu[0,0,0] = 0
    
    ellmax = 3
    
    nkin = len(kin)
        
    if k2weight:
        dk = np.zeros_like(kin)
        dk[:-1] = kin[1:] - kin[:-1]
        dk[-1] = dk[-2]
    
    nkout = len(kout) - 1
    dkin = (kin[1:] - kin[:-1])[0]
    
    idx_o = np.digitize(knorm, kout) - 1
    nmodes_out = np.zeros(nkout * 3)

    idx_i = np.digitize(kin, kout) - 1
    nmodes_in = np.zeros(nkout, dtype=np.float32)

    for i in range(len(kout)):
        idx = i==idx_i
        if k2weight:
            nmodes_in[i] = np.sum(kin[idx]**2 * dk[idx])
        else:
            nmodes_in[i] = np.sum(idx)
            
    norm_in = 1/nmodes_in
    norm_in[nmodes_in==0] = 0
    norm_in_allell = np.zeros(3 * len(norm_in))
    norm_in_allell[:nkout] = norm_in
    norm_in_allell[nkout:2*nkout] = norm_in
    norm_in_allell[2*nkout:3*nkout] = norm_in
    
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
                if (idx_o[i,j,k]>=nkout): 
                    pass
                else:
                    if k==0:
                        nmodes_out[idx_o[i,j,k]::nkout] += 1
                        keff[idx_o[i,j,k]] += knorm[i,j,k]
                    else:
                        nmodes_out[idx_o[i,j,k]::nkout] += 2
                        keff[idx_o[i,j,k]] += 2 * knorm[i,j,k]

                    for beta in range(nkin):
                        if k2weight:
                            w = kin[beta]**2 * dk[beta]
                        else:
                            w = 1
                        if (idx_i[beta] == idx_o[i,j,k]):               
                            for ell in range(ellmax):
                                for ellp in range(ellmax):
                                    if k!=0:
                                        window[int(ell * nkout) + int(idx_o[i,j,k]), int(ellp * nkin) + int(beta)] += 2 * pref[ell] * legs[ell][i,j,k] * legs[ellp][i,j,k] * w # * norm_in[idx_o[i,j,k]]
                                    else:
                                        window[int(ell * nkout) + int(idx_o[i,j,k]), int(ellp * nkin) + int(beta)] += pref[ell] * legs[ell][i,j,k] * legs[ellp][i,j,k] * w # * norm_in[idx_o[i,j,k]]

    norm_out = 1/nmodes_out
    norm_out[nmodes_out==0] = 0
    window = window * norm_out.reshape(-1, 1) * norm_in_allell.reshape(-1, 1)
    keff = keff * norm_out[:nkout]
    
    return window, keff

def measure_2pt_bias_rsd(k, pk_ij_heft, pk_tt, kmax, ellmax=2, nbias=3, kmin=0.0):
    
    kidx_max = k.searchsorted(kmax)
    # TESTING
    kmin = 0.01
    kidx_min = k.searchsorted(kmin)
    kcut = k[kidx_min:kidx_max]
    pk_tt_kcut = pk_tt[:ellmax,kidx_min:kidx_max]
    pk_ij_heft_kcut = pk_ij_heft[:,:,kidx_min:kidx_max]
    bvec0 = [1, 0, 0]#, 0, 0]
    # adding more realistic noise: error on P(k) is sqrt(2/Nk) P(k)
    #dk = kcut[1]-kcut[0]
    #Nk = kcut**2*dk # propto missing division by (2.pi/L^3)
    Nk = 1
    
    loss = lambda bvec : np.sum((pk_tt_kcut - combine_spectra(kcut, pk_ij_heft_kcut, np.hstack([bvec,np.zeros(10-len(bvec))]), rsd=True)[:ellmax,:])**2/(2 * pk_tt_kcut**2 / Nk))
    #b1,b2,bs,alpha0,alpha2,alpha4,alpha6,sn,sn2,sn4
    
    out = minimize(loss, bvec0)
    
    return out

def measure_2pt_bias(k, pk_ij_heft, pk_tt, kmax, nbias=3, kmin=0.0):
    
    kidx_max = k.searchsorted(kmax)
    # TESTING
    kmin = 0.01
    kidx_min = k.searchsorted(kmin)
    
    kcut = k[kidx_min:kidx_max]
    pk_tt_kcut = pk_tt[kidx_min:kidx_max]
    pk_ij_heft_kcut = pk_ij_heft[:,kidx_min:kidx_max]
    bvec0 = [1, 0, 0, 0, 5000]
    # adding more realistic noise: error on P(k) is sqrt(2/Nk) P(k)
    #dk = kcut[1]-kcut[0]
    #Nk = kcut**2*dk # propto missing division by (2.pi/L^3)
    Nk = 1
    
    loss = lambda bvec : np.sum((pk_tt_kcut - combine_spectra(kcut, pk_ij_heft_kcut, np.hstack([bvec,np.zeros(5-len(bvec))])))**2/(2 * pk_tt_kcut**2 / Nk))
    #b1,b2,bs,alpha0,alpha2,alpha4,alpha6,sn,sn2,sn4
    
    out = minimize(loss, bvec0)
    
    return out

def read_power(power_fn, keynames):
    f = asdf.open(str(power_fn))
    k = f['data']['k_binc'].flatten()
    mu = np.zeros((len(k), 1))
    if "rsd" in str(power_fn):
        pk_tt = np.zeros((1, 3, len(k)))
        pk_ij_zz = np.zeros((15, 3, len(k)))
        pk_ij_zt = np.zeros((5, 3, len(k)))
    else:
        pk_tt = np.zeros((1, len(k), 1))
        pk_ij_zz = np.zeros((15, len(k), 1))
        pk_ij_zt = np.zeros((5, len(k), 1))
    
    if "rsd" in str(power_fn):
        pk_tt[0, :, :] = f['data']['P_ell_tr_tr'].reshape(3, len(k))
    else:
        pk_tt[0, :, :] = f['data']['P_kmu_tr_tr'].reshape(len(k), 1)
    count = 0
    for i in range(len(keynames)):
        if "rsd" in str(power_fn):
            pk_ij_zt[i, :, :] = f['data'][f'P_ell_{keynames[i]}_tr'].reshape(3, len(k))
        else:
            pk_ij_zt[i, :, :] = f['data'][f'P_kmu_{keynames[i]}_tr'].reshape(len(k), 1)
        for j in range(len(keynames)):
            if i < j: continue
            if "rsd" in str(power_fn):
                pk_ij_zz[count, :, :] = f['data'][f'P_ell_{keynames[i]}_{keynames[j]}'].reshape(3, len(k))
            else:
                pk_ij_zz[count, :, :] = f['data'][f'P_kmu_{keynames[i]}_{keynames[j]}'].reshape(len(k), 1)
            count += 1
    f.close()
    #print(k, mu, pk_tt, pk_ij_zz, pk_ij_zt)
    print("zeros = ", np.sum(pk_tt == 0.), np.sum(pk_ij_zz == 0.), np.sum(pk_ij_zt == 0.))
    return k, mu, pk_tt, pk_ij_zz, pk_ij_zt

def read_power_dict(power_tr_dict, power_ij_dict, want_rsd, keynames, poles):
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
    else:
        pk_tt[0, :, :] = power_tr_dict['P_kmu_tr_tr'].reshape(len(k), 1)
    count = 0
    for i in range(len(keynames)):
        if want_rsd:
            pk_ij_zt[i, :, :] = power_tr_dict[f'P_ell_{keynames[i]}_tr'].reshape(len(poles), len(k))
        else:
            pk_ij_zt[i, :, :] = power_tr_dict[f'P_kmu_{keynames[i]}_tr'].reshape(len(k), 1)
        for j in range(len(keynames)):
            if i < j: continue
            if want_rsd:
                pk_ij_zz[count, :, :] = power_ij_dict[f'P_ell_{keynames[i]}_{keynames[j]}'].reshape(len(poles), len(k))
            else:
                pk_ij_zz[count, :, :] = power_ij_dict[f'P_kmu_{keynames[i]}_{keynames[j]}'].reshape(len(k), 1)
            count += 1
    
    #print(k, mu, pk_tt, pk_ij_zz, pk_ij_zt)
    print("zeros = ", np.sum(pk_tt == 0.), np.sum(pk_ij_zz == 0.), np.sum(pk_ij_zt == 0.))
    return k, mu, pk_tt, pk_ij_zz, pk_ij_zt

def get_cfg(sim_name, z_this, nmesh):
    meta = get_meta(sim_name, redshift=z_this)
    Lbox = meta['BoxSize']
    z_ic = meta['InitialRedshift']
    k_Ny = np.pi*nmesh/Lbox
    kcut = 0.5*k_Ny
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

    # create a dict with everything you would ever need
    cfg = {'lbox': Lbox, 'Cosmology': cosmo, 'surrogate_gaussian_cutoff': kcut, 'z_ic': z_ic}
    return cfg

def run_zcv(power_rsd_tr_dict, power_rsd_ij_dict, power_tr_dict, power_ij_dict, config):

    # read out some parameters from the config function
    sim_name = config['sim_params']['sim_name']
    z_this = config['sim_params']['z_mock']
    zcv_dir = config['zcv_params']['zcv_dir']
    nmesh = config['zcv_params']['nmesh']
    want_rsd = config['HOD_params']['want_rsd']
    rsd_str = "_rsd" if want_rsd else ""
    
    # power params
    k_hMpc_max = config['power_params']['k_hMpc_max']
    logk = config['power_params']['logk']
    n_k_bins = config['power_params']['nbins_k']
    n_mu_bins = config['power_params']['nbins_mu']
    poles = config['power_params']['poles']
    
    # create save directory
    save_dir = Path(zcv_dir) / sim_name
    save_z_dir = save_dir / f"z{z_this:.3f}"

    # name of files to read from
    zenbu_fn = save_z_dir / f"zenbu_pk{rsd_str}_ij_lpt.npz"
    pk_lin_fn = save_dir / "abacus_pk_lin_ic.dat"
    window_fn = save_dir / f"window_nmesh{nmesh:d}.npz"

    # read the config params
    cfg = get_cfg(sim_name, z_this, nmesh)
    cfg['p_lin_ic_file'] = str(pk_lin_fn)
    cfg['nmesh_in'] = nmesh
    cfg['poles'] = poles
    Lbox = cfg['lbox']
   
    # define k bins
    k_bins, mu_bins = get_k_mu_edges(Lbox, k_hMpc_max, n_k_bins, n_mu_bins, logk)
    k_binc = (k_bins[1:] + k_bins[:-1])*.5
    
    # field names
    keynames = ["1cb", "delta", "delta2", "tidal2", "nabla2"]

    # read out the dictionaries
    if want_rsd: # then we need the no-rsd version (names are misleading)
        k, mu, pk_tt, pk_ij_zz, pk_ij_zt = read_power_dict(power_tr_dict, power_ij_dict, want_rsd=False, keynames=keynames, poles=poles)
    k, mu, pk_tt_poles, pk_ij_zz_poles, pk_ij_zt_poles = read_power_dict(power_rsd_tr_dict, power_rsd_ij_dict, want_rsd=want_rsd, keynames=keynames, poles=poles)
    assert len(k) == len(k_binc)
    
    # load the linear power spectrum
    p_in = np.genfromtxt(cfg['p_lin_ic_file'])
    kth, p_m_lin = p_in[:,0], p_in[:,1]

    # fit to get the biases # ellmax = 1 fits to monopole only ; 0.15 might be too high; try going to 0.1 tuks (nbar P(k) for shotnoise) nabla get rid of;
    kmax = 0.1 #0.15 # og
    if want_rsd:
        bvec_opt = measure_2pt_bias_rsd(k, pk_ij_zz_poles[:,:,:], pk_tt_poles[0,:,:], kmax, ellmax=1)
        bvec_opt_rs = measure_2pt_bias(k, pk_ij_zz[:,:,0], pk_tt[0,:,0], kmax)
        print("bias zspace", bvec_opt['x'])
    else:
        # names are misleading
        bvec_opt_rs = measure_2pt_bias(k, pk_ij_zz_poles[:,:,0], pk_tt_poles[0,:,0], kmax)
    print("bias rspace", bvec_opt_rs['x'])

    # load the presaved window function
    data = np.load(window_fn)
    window = data['window']
    window_exact = False
    
    # load the presaved zenbu power spectra
    data = np.load(zenbu_fn)
    pk_ij_zenbu = data['pk_ij_zenbu']
    lptobj = data['lptobj']

    # reduce variance of measurement # give only first element, set rest to 0 (use b1, b1+b2) tuks (shotnoise is last; not used)
    bias_vec = np.array(bvec_opt_rs['x']) # b1, b2, bs, bn, shot
    bias_vec[-2] = 0. # set to 0 if not using nabla
    bias_vec = np.hstack(([1], bias_vec))
    #bias_vec = [1, *bvec_opt_rs['x']]

    # decide what to input depending on whether rsd requested or not
    if want_rsd:
        pk_tt_input = pk_tt_poles[0,...]
        pk_ij_zz_input = pk_ij_zz_poles
        pk_ij_zt_input = pk_ij_zt_poles
    else:
        pk_tt_input = pk_tt_poles[0, :, 0]
        pk_ij_zz_input = pk_ij_zz_poles[:,:,0]
        pk_ij_zt_input = pk_ij_zt_poles[:,:,0]
    
    pk_nn_hat, pk_nn_betasmooth, pk_nn_betasmooth_nohex,\
    pk_nn_beta1, beta, beta_damp, beta_smooth, \
    beta_smooth_nohex, pk_zz, pk_zenbu, r_zt, r_zt_smooth,\
    r_zt_smooth_nohex, pk_zn, pk_ij_zenbu_poles, pkclass = reduce_variance_tt(k_binc, pk_tt_input, pk_ij_zt_input,
                                                                              pk_ij_zz_input, cfg, z_this,
                                                                              bias_vec, kth, p_m_lin, rsd=want_rsd,
                                                                              win_fac=1, kout=k_bins,
                                                                              window=window, exact_window=window_exact,
                                                                              pk_ij_zenbu=pk_ij_zenbu, lptobj=lptobj)

    zcv_dict = {}
    zcv_dict['k_binc'] = k_binc
    zcv_dict['poles'] = poles
    zcv_dict['rho_tr_ZD'] = r_zt
    zcv_dict['Pk_ZD_ZD_ell'] = pk_zz
    zcv_dict['Pk_tr_tr_ell'] = pk_tt_poles
    zcv_dict['Pk_tr_tr_ell_zcv'] = pk_nn_betasmooth
    zcv_dict['Pk_ZD_ZD_ell_ZeNBu'] = pk_zenbu
    return zcv_dict
    
    
if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    from classy import Class    
    from abacusnbody.metadata import get_meta

    # get a few parameters for the simulation
    sim_name = "AbacusSummit_high_c000_ph100"
    z_this = 3.0
    cfg = get_cfg(sim_name, z_this)
    Lbox = cfg['lbox']
    nmesh = cfg['nmesh']
    
    keynames = ["1cb", "delta", "delta2", "tidal2", "nabla2"]
    save_dir = "/global/cscratch1/sd/boryanah/zcv/anew"
    
    # read power spectra
    power_fn = Path(save_dir) / f"power_{sim_name}_{nmesh:d}.asdf"
    power_rsd_fn = Path(save_dir) / f"power_rsd_{sim_name}_{nmesh:d}.asdf"
    k, mu, pk_tt, pk_ij_zz, pk_ij_zt = read_power(power_fn, keynames)
    k, mu, pk_tt_poles, pk_ij_zz_poles, pk_ij_zt_poles = read_power(power_rsd_fn, keynames)

    # load the linear power spectrum
    p_in = np.genfromtxt(cfg['p_lin_ic_file'])
    kth, p_m_lin = p_in[:,0], p_in[:,1]

    # fit to get the biases
    bvec_opt = measure_2pt_bias_rsd(k, pk_ij_zz_poles[:,:,:], pk_tt_poles[0,:,:], 0.15, ellmax=1)
    bvec_opt_rs = measure_2pt_bias(k, pk_ij_zz[:,:,0], pk_tt[0,:,0], 0.15)
    print("bias zspace", bvec_opt['x'])
    print("bias rspace", bvec_opt_rs['x'])

    # kout is kedges
    dk = np.diff(k)[0]
    kout = k-dk/2.
    kout = np.hstack((kout, kout[-1]+dk))

    # let's make a window
    fn_win = f"window_nmesh{nmesh:d}.npy"
    if os.path.exists(fn_win):
        window = np.load(fn_win)
    else:
        window, keff = periodic_window_function(nmesh, Lbox, kout, k, k2weight=True) # I think this should have the right shape now
        print("finished with window")
        np.save(fn_win, window)
    window_exact = False
    #window_exact = True

    # presave the zenbu power spectra
    zenbu_fn = "data/zenbu_data.npz"
    if os.path.exists(zenbu_fn):
        data = np.load(zenbu_fn)
        pk_ij_zenbu = data['pk_ij_zenbu']
        lptobj = data['lptobj']
    else:
        pk_ij_zenbu, lptobj = zenbu_spectra(k, z_this, cfg, kth, p_m_lin, pkclass=None, rsd=True)
        p0table = lptobj.p0ktable
        p2table = lptobj.p2ktable
        p4table = lptobj.p4ktable
        lptobj = np.array([p0table, p2table, p4table])
        np.savez(zenbu_fn, pk_ij_zenbu=pk_ij_zenbu, lptobj=lptobj)

    # to pass the pre-saved window window_exact=False (note that it overwrite window even if passed)
    # more accurate if you don't pass the window function
    bias_vec = [1, *bvec_opt_rs['x']]
    pk_nn_hat, pk_nn_betasmooth, pk_nn_betasmooth_nohex,\
    pk_nn_beta1, beta, beta_damp, beta_smooth, \
    beta_smooth_nohex, pk_zz, pk_zenbu, r_zt, r_zt_smooth,\
    r_zt_smooth_nohex, pk_zn, pk_ij_zenbu_poles, pkclass = reduce_variance_tt(k, pk_tt_poles[0,...], pk_ij_zt_poles,\
                                                                              pk_ij_zz_poles, cfg, z_this,\
                                                                              bias_vec, kth, p_m_lin, rsd=True, win_fac=1, kout=kout, window=window, exact_window=window_exact, pk_ij_zenbu=pk_ij_zenbu, lptobj=lptobj)

    # save these things
    print("correlation coefficient", r_zt)
    print("kout", kout)
    print(kout.shape, r_zt.shape)
    np.savez(f"corr_coeff_nmesh{nmesh:d}.npz", kout=kout, r_zt=r_zt)
    print("reduced variance tt")

    # plotting
    f, ax = plt.subplots(1, 2, sharex=True, sharey=True)
    sidx = 0
    for ell in range(2):
        #ax[ell].plot(k, pk_zz[ell,:]/pk_zenbu[ell,:] - 1)
        ax[ell].plot(k, k*pk_zz[ell,:])
        ax[ell].plot(k, k*pk_zenbu[ell,:])
    #ax[0].set_ylim([-0.2, 0.2])
    ax[0].set_xlim([0, 0.5])
    plt.savefig(f"first_nmesh{nmesh:d}.png")
    plt.close()

    f, ax = plt.subplots(1, 2, sharex=True, sharey=True)
    for ell in range(2):
        ax[ell].plot(k, k * pk_tt_poles[0,ell,:])   
        ax[ell].plot(k, k * pk_nn_betasmooth[ell,:])
        ax[ell].plot(k, k * pk_zenbu[ell,:])

    ax[0].set_xlim([0, 0.5])
    ax[0].set_ylim([-100, 3000])
    plt.savefig(f"second_nmesh{nmesh:d}.png")
    plt.close()

