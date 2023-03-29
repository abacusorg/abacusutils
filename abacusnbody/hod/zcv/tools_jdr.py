"""
Tools for applying variance reduction (ZCV) (base of script by Joe DeRose).
"""
import os, gc
from pathlib import Path

import asdf
import numpy as np
from numba import jit
from scipy.interpolate import interp1d, splev, splrep
from scipy.optimize import minimize
from scipy.signal import savgol_filter
from scipy.special import legendre

from abacusnbody.hod.power_spectrum import get_k_mu_edges, get_k_mu_box_edges, project_3d_to_poles
from abacusnbody.metadata import get_meta
from .ic_fields import compress_asdf


try:
    from classy import Class
    from ZeNBu.zenbu import Zenbu
    from ZeNBu.zenbu_rsd import Zenbu_RSD
except ImportError as e:
    raise ImportError('Missing imports for zcv. Install abacusutils with '
        '"pip install abacusutils[zcv]" to install zcv dependencies.')


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

def combine_cross_kaiser_spectra(k, spectra_dict, D, bias, f_growth, rec_algo, R, rsd=False):
    """
    kinda slow smoothing
    """    
    if rec_algo == "recsym":
        # < D (b delta + f mu2 delta, tr > = D * (b < delta, tr> + f < mu2 delta, tr >)
        if rsd:
            pk = D * (bias * spectra_dict['P_ell_delta_tr'] + f_growth * spectra_dict['P_ell_deltamu2_tr'])
        else:
            pk = D * (bias * spectra_dict['P_kmu_delta_tr'] + f_growth * spectra_dict['P_kmu_deltamu2_tr'])
    elif rec_algo == "reciso":
        # < D ((b+f mu^2)(1-S) + bS) delta, tr > = D * (b < delta, tr > + f (1-S) < delta, mu^2 >)
        assert R is not None
        S = get_smoothing(k, R)
        f_eff = f_growth * (1.-S)
        if rsd:
            f_eff = f_eff.reshape(1, len(k), 1)
            print(f_eff.shape, spectra_dict['P_ell_delta_tr'].shape)
            pk = D * (bias * spectra_dict['P_ell_delta_tr'] + f_eff * spectra_dict['P_ell_deltamu2_tr'])
        else:
            pk = D * (bias * spectra_dict['P_kmu_delta_tr'] + f_eff * spectra_dict['P_kmu_deltamu2_tr'])
    return pk

def combine_kaiser_spectra(k, spectra_dict, D, bias, f_growth, rec_algo, R, rsd=False):
    """
    kinda slow
    """
    if rec_algo == "recsym":
        # < D (b delta + f mu2 delta, D (b delta + f mu2 delta) > = D^2 (b^2 < delta, delta> + f^2 < mu2 delta, mu2 delta > + 2 b f < delta, mu2 delta >)
        if rsd:
            pk = D**2 * (2. * bias * f_growth * spectra_dict['P_ell_deltamu2_delta'] + f_growth**2 * spectra_dict['P_ell_deltamu2_deltamu2'] + bias**2 * spectra_dict['P_ell_delta_delta'])
        else:
            pk = D**2 * (2. * bias * f_growth * spectra_dict['P_kmu_deltamu2_delta'] + f_growth**2 * spectra_dict['P_kmu_deltamu2_deltamu2'] + bias**2 * spectra_dict['P_kmu_delta_delta'])
    elif rec_algo == "reciso":
        # ((b+f mu^2)(1-S) + bS) delta = (b + f (1-S) mu2) delta
        # < D ((b+f mu^2)(1-S) + bS) delta, D ((b+f mu^2)(1-S) + bS) delta > = D^2 (b^2 < delta, delta> + fefff^2 < mu2 delta, mu2 delta > + 2 b feff < delta, mu2 delta >)
        assert R is not None
        S = get_smoothing(k, R) 
        f_eff = f_growth * (1.-S)
        if rsd:
            f_eff = f_eff.reshape(1, len(k), 1)
            print(f_eff.shape, spectra_dict['P_ell_delta_delta'].shape)
            pk = D**2 * (2. * bias * f_eff * spectra_dict['P_ell_deltamu2_delta'] + f_eff**2 * spectra_dict['P_ell_deltamu2_deltamu2'] + bias**2 * spectra_dict['P_ell_delta_delta'])
        else:
            pk = D**2 * (2. * bias * f_eff * spectra_dict['P_kmu_deltamu2_delta'] + f_eff**2 * spectra_dict['P_kmu_deltamu2_deltamu2'] + bias**2 * spectra_dict['P_kmu_delta_delta'])
    return pk

def get_poles(k, pk, D, bias, f_growth, poles=[0, 2, 4]):
    beta = f_growth/bias
    p_ell = np.zeros((len(poles), len(k)))
    count = 0
    if 0 in poles:
        p_ell[count] = ((1. + 2./3.*beta + 1./5*beta**2)*pk)
        count += 1
    if 2 in poles:
        p_ell[count] = ((4./3.*beta + 4./7*beta**2)*pk)
        count += 1
    if 4 in poles:
        p_ell[count] = ((8./35*beta**2)*pk)
        count += 1
    p_ell *= bias**2 * D**2
    return k, p_ell

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
        pspline, lptobj = _realspace_lpt_pk(kin, pin*(Dthis/Dic)**2, cutoff=cutoff)
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

    # remove
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

        
    pk_nn_hat, pk_nn_betasmooth, pk_nn_betasmooth_nohex, pk_nn_beta1, beta, beta_damp, beta_smooth, beta_smooth_nohex, pk_zz, pk_zenbu, r_zt, r_zt_smooth, r_zt_smooth_nohex, pk_zn, r_zt_sn_lim = compute_beta_and_reduce_variance_tt(k, pk_nn, pk_ij_zn, pk_ij_zz, pk_ij_zenbu, bias_vec, window=window, kin=kin, kout=kout, s_ell=s_ell, rsd=rsd, k0=0.618, dk=0.167, sg_window=sg_window, poles=cfg['poles'])
    return pk_nn_hat, pk_nn_betasmooth, pk_nn_betasmooth_nohex, pk_nn_beta1, beta, beta_damp, beta_smooth, beta_smooth_nohex, pk_zz, pk_zenbu, r_zt, r_zt_smooth, r_zt_smooth_nohex, pk_zn, pk_ij_zenbu, pkclass, r_zt_sn_lim


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


    # TODO: sketchy stuff 
    # P_epseps = <(delta_t(k)-delta_z(k))(delta_t(-k)-delta_z(-k))> = Ptt-2Pzt+Pzz
    shotnoise = (pk_nn - 2. * pk_zn + pk_zz)[0]
    #shotnoise = 1.048156031502e+03
    pk_nn_nosn = pk_nn.copy()
    pk_nn_nosn[0] -= shotnoise
    #print("shotnoise estimate", shotnoise)
    if rsd:
        var_nn_nosn = np.stack([multipole_cov(pk_nn_nosn, ell) for i_ell, ell in enumerate(poles)])
    else:
        var_nn_nosn = 2. * (pk_nn_nosn)**2
    r_zt_sn_lim = var_nn_nosn / np.sqrt(var_nn * var_nn_nosn)
    
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
        # TODO: kinda ugly
        try:
            beta_smooth[i,:] = savgol_filter(beta_damp.T[:,i], sg_window, 3) #splev(k, betaspl)
        except: # only when doing the smoke test because we have few points
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

    if (window is not None) and rsd:

        pk_zenbu = np.hstack(pk_zenbu)
        if rsd:
            pk_zenbu = np.dot(window.T, pk_zenbu).reshape(len(poles),-1)
        else:
            print("no window")
            # no need to apply window in real space
            #else:
            #    pk_zenbu = np.dot(window[:window.shape[0]//len(poles), :window.shape[1]//len(poles)].T, pk_zenbu).reshape(pk_zz.shape)

    pk_nn_hat = pk_nn - beta_damp * (pk_zz - pk_zenbu) # pk_zz has shape of measurement nmesh/2
    pk_nn_betasmooth = pk_nn - beta_smooth * (pk_zz - pk_zenbu) # joe says beta needs to be smooth
    pk_nn_betasmooth_nohex = pk_nn - beta_smooth * (pk_zz - pk_zenbu) # same as above
    pk_nn_beta1 = pk_nn - (pk_zz - pk_zenbu)
        
    return pk_nn_hat, pk_nn_betasmooth, pk_nn_betasmooth_nohex, pk_nn_beta1, beta, beta_damp, beta_smooth, beta_smooth_nohex, pk_zz, pk_zenbu, r_zt, r_zt_smooth, r_zt_smooth_nohex, pk_zn, r_zt_sn_lim


# not used
def reduce_variance_field(k_box, pk_nn, pk_zn, pk_zz, pk_zb, cfg,
                          sg_window=21, rsd=False,
                          k0=0.618, dk=0.167, beta1_k=0.05, poles=[0, 2, 4]):

    # compute beta
    k = (np.fft.fftfreq(cfg['nmesh_in'], d=cfg['lbox']/cfg['nmesh_in'])*2.*np.pi).astype(np.float32)
    w = 1/2 * (1 - np.tanh((k - k0)/dk))
    beta_damp = pk_zn**2/pk_zz**2
    beta_damp *= w[:, np.newaxis, np.newaxis]
    beta_damp *= w[np.newaxis, :, np.newaxis]
    beta_damp *= w[np.newaxis, np.newaxis, :]
    beta_damp[k_box.reshape(beta_damp.shape) < beta1_k] = 1.
    beta_smooth = savgol_filter(beta_damp, sg_window, 3) # axis=-1 only
    del k, w; gc.collect()
    
    # cross-correlation
    r_zt = pk_zn**2 / (pk_zz * pk_nn)
    
    # get reduced fields
    pk_nn_hat = pk_nn - beta_damp * (pk_zz - pk_zb)
    pk_nn_betasmooth = pk_nn - beta_smooth * (pk_zz - pk_zb)
        
    return pk_nn_hat, pk_nn_betasmooth, beta_damp, beta_smooth, r_zt

def reduce_variance_lin(k, pk_tt, pk_lt, pk_ll, p_m_lin, window=None,
                        s_ell=[0.1, 10, 100], sg_window=21, rsd=False,
                        k0=0.618, dk=0.167, beta1_k=0.05, poles=[0, 2, 4]):
    
    if rsd:
        cov_lt = np.stack([multipole_cov(pk_lt, ell) for ell in poles])
        var_ll = np.stack([multipole_cov(pk_ll, ell) for ell in poles])
        var_tt = np.stack([multipole_cov(pk_tt, ell) for ell in poles])
    else:
        cov_lt = 2 * pk_lt ** 2
        var_ll = 2 * pk_ll ** 2
        var_tt = 2 * pk_tt ** 2

    # TODO: sketchy stuff is this really working?
    # P_epseps = <(delta_t(k)-delta_z(k))(delta_t(-k)-delta_z(-k))> = Ptt-2Pzt+Pzz
    shotnoise = (pk_tt - 2. * pk_lt + pk_ll)[0]
    pk_tt_nosn = pk_tt.copy()
    pk_tt_nosn[0] -= shotnoise
    #print("shotnoise estimate", shotnoise)
    #shotnoise = [1.048156031502e+03, 0, 0]
    #print("shotnoise theory", shotnoise)
    if rsd:
        var_tt_nosn = np.stack([multipole_cov(pk_tt_nosn, ell) for i_ell, ell in enumerate(poles)])
    else:
        var_tt_nosn = 2. * (pk_tt-shotnoise[0])**2
    r_lt_sn_lim = var_tt_nosn / np.sqrt(var_tt * var_tt_nosn)
        
    beta = cov_lt / var_ll
    beta_damp = 1/2 * (1 - np.tanh((k - k0)/dk)) * beta

    r_lt = cov_lt / np.sqrt(var_ll * var_tt)
    beta_damp = np.atleast_2d(beta_damp)
    beta_damp[beta_damp != beta_damp] = 0
    beta_damp[:, :k.searchsorted(beta1_k)] = 1
    
    r_lt = np.atleast_2d(r_lt)
    r_lt[r_lt != r_lt] = 0

    beta_smooth = np.zeros_like(beta_damp)
    beta_smooth_nohex = np.zeros_like(beta_damp)
    
    for i in range(beta_smooth.shape[0]):
        # ToDO: kinda ugly
        try:
            beta_smooth[i, :] = savgol_filter(beta_damp.T[:, i], sg_window, 3)
        except: # only when running the smoke test
            beta_smooth[i, :] = savgol_filter(beta_damp.T[:, i], 3, 2)
        if i != 2:
            try:
                beta_smooth_nohex[i, :] = savgol_filter(beta_damp.T[:, i], sg_window, 3)
            except:
                beta_smooth_nohex[i, :] = savgol_filter(beta_damp.T[:, i], 3, 2)
        else:
            beta_smooth_nohex[i, :] = 1
            
    r_lt_smooth = np.zeros_like(r_lt)
    r_lt_smooth_nohex = np.zeros_like(r_lt)

    for i in range(r_lt.shape[0]):
        spl = splrep(k, r_lt.T[:, i], s=s_ell[i])
        r_lt_smooth[i, :] = splev(k, spl) 
        if i<1:
            r_lt_smooth_nohex[i, :] = splev(k, spl)
        else:
            r_lt_smooth_nohex[i, :] = 1
            
    if (window is not None) and rsd:
        p_m_lin = np.hstack(p_m_lin)
        if rsd:
            p_m_lin = np.dot(window.T, p_m_lin).reshape(len(poles), -1)
    
    pk_tt_hat = pk_tt - beta_damp * (pk_ll - p_m_lin)
    pk_tt_betasmooth = pk_tt - beta_smooth * (pk_ll - p_m_lin)
    pk_tt_betasmooth_nohex = pk_tt - beta_smooth * (pk_ll - p_m_lin)
    pk_tt_beta1 = pk_tt - (pk_ll - p_m_lin)
    return pk_tt_hat, pk_tt_betasmooth, pk_tt_betasmooth_nohex, pk_tt_beta1, beta, beta_damp, beta_smooth, beta_smooth_nohex, pk_ll, p_m_lin, r_lt, r_lt_smooth, r_lt_smooth_nohex, pk_lt, r_lt_sn_lim


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
    kidx_min = k.searchsorted(kmin)
    kcut = k[kidx_min:kidx_max]
    pk_tt_kcut = pk_tt[kidx_min:kidx_max]
    pk_ij_heft_kcut = pk_ij_heft[:,kidx_min:kidx_max]
    bvec0 = [1, 0, 0, 0, 5000] # og b1, b2, bs, bn, sn
    # adding more realistic noise: error on P(k) is sqrt(2/Nk) P(k)
    #dk = kcut[1]-kcut[0]
    #Nk = kcut**2*dk # propto missing division by (2.pi/L^3)
    Nk = 1

    loss = lambda bvec : np.sum((pk_tt_kcut - combine_spectra(kcut, pk_ij_heft_kcut, np.hstack([bvec,np.zeros(5-len(bvec))])))**2)#/(2 * pk_tt_kcut**2 / Nk))
    #b1,b2,bs,alpha0,alpha2,alpha4,alpha6,sn,sn2,sn4

    out = minimize(loss, bvec0)
    return out

def combine_field_spectra_full(bias, power_ij_fns, keynames):
    counter = 0
    for i in range(len(keynames)):
        for j in range(len(keynames)):
            if i < j: continue
            if i == 0 and j == 0:
                power = np.zeros_like(asdf.open(power_ij_fns[counter])['data'][f'P_k3D_{keynames[i]}_{keynames[j]}'])
            if (i == j):
                power += bias[i]*bias[j]*asdf.open(power_ij_fns[counter])['data'][f'P_k3D_{keynames[i]}_{keynames[j]}']
            else:
                power += 2.*bias[i]*bias[j]*asdf.open(power_ij_fns[counter])['data'][f'P_k3D_{keynames[i]}_{keynames[j]}']
            counter += 1
    return power

def combine_field_cross_spectra_full(bias, power_tr_fns, keynames):
    counter = 1 # 0 is tr tr
    for i in range(len(keynames)):
        if i == 0:
            power = np.zeros_like(asdf.open(power_tr_fns[counter])['data'][f'P_k3D_{keynames[i]}_tr'])
        power += bias[i]*asdf.open(power_tr_fns[counter])['data'][f'P_k3D_{keynames[i]}_tr']
        counter += 1
    return power

def combine_field_spectra(bvec, power_ij, keynames):
    bias = np.hstack((1., bvec))
    for i in range(len(keynames)):
        for j in range(len(keynames)):
            if i < j: continue
            if i == 0 and j == 0:
                power = np.zeros_like(power_ij[f'{keynames[i]}_{keynames[j]}'])
            if (i == j):
                power += bias[i]*bias[j]*power_ij[f'{keynames[i]}_{keynames[j]}']
            else:
                power += 2.*bias[i]*bias[j]*power_ij[f'{keynames[i]}_{keynames[j]}']
            #print("bi, bj, p", bias[i], bias[j], power[:10])
    power += bias[-1]
    return power

def measure_field_bias(power_tr_fns, power_ij_fns, k_box, keynames, kmax, kmin=0.0):
    pk_tt_kcut = asdf.open(power_tr_fns[0])['data'][f'P_k3D_tr_tr'].flatten()[(k_box < kmax) & (k_box > kmin)]
    kcut = k_box[(k_box < kmax) & (k_box > kmin)]
    bvec0 = [1, 0, 0, 0, 0, 5000] # b1, b2, bs, bn, sn
    power_ij = {}
    counter = 0
    for i in range(len(keynames)):
        for j in range(len(keynames)):
            if i < j: continue
            power_ij[f'{keynames[i]}_{keynames[j]}'] = asdf.open(power_ij_fns[counter])['data'][f'P_k3D_{keynames[i]}_{keynames[j]}'].flatten()[(k_box < kmax) & (k_box > kmin)]
            counter += 1
    print(pk_tt_kcut[:10], power_ij['1cb_1cb'][:10], power_ij['delta_1cb'][:10])
    def loss(bvec): # I think you want something that divides by Nmodes
        l = (pk_tt_kcut - combine_field_spectra(bvec, power_ij, keynames))**2/pk_tt_kcut**2
        l[np.isnan(l)] = 0.
        l[np.isinf(l)] = 0.
        l = np.sum(l)
        return l
    # b1,b2,bs,alpha0,alpha2,alpha4,alpha6,sn,sn2,sn4
    out = minimize(loss, bvec0, method='Nelder-Mead')    
    return out

def get_smoothing(k, R):
    return np.exp(-k**2*R**2/2.)

def measure_2pt_bias_lin(k, power_dict, power_rsd_tr_dict, D, f_growth, kmax, rsd, rec_algo, R, ellmax=2, kmin=0.0):
    pk_tt = power_rsd_tr_dict['P_ell_tr_tr'][:ellmax, :, 0]
    kidx_max = k.searchsorted(kmax)
    kidx_min = k.searchsorted(kmin)
    kcut = k[kidx_min:kidx_max]
    pk_tt_kcut = pk_tt[:ellmax, kidx_min:kidx_max]
    power_lin_dict = power_dict.copy()
    for key in power_lin_dict.keys():
        if "P_ell" not in key: continue
        power_lin_dict[key] = power_lin_dict[key][:, kidx_min:kidx_max]
    loss = lambda bias: np.sum((pk_tt_kcut - combine_kaiser_spectra(kcut, power_lin_dict, D, bias, f_growth, rec_algo, R, rsd=rsd)[:ellmax, :, 0])**2/(2 * pk_tt_kcut**2))
    out = minimize(loss, 1.)
    return out

def measure_2pt_bias_cross_lin(k, power_dict, power_rsd_tr_dict, D, f_growth, kmax, rsd, rec_algo, R, ellmax=2, kmin=0.0): 
    kidx_max = k.searchsorted(kmax)
    kidx_min = k.searchsorted(kmin)
    kcut = k[kidx_min:kidx_max]
    power_rsd_dict = power_rsd_tr_dict.copy()
    power_lin_dict = power_dict.copy()
    for key in power_lin_dict.keys():
        if "P_ell" not in key: continue
        power_lin_dict[key] = power_lin_dict[key][:, kidx_min:kidx_max]
    for key in power_rsd_dict.keys():
        if "P_ell" not in key: continue
        power_rsd_dict[key] = power_rsd_dict[key][:, kidx_min:kidx_max]
    loss = lambda bias: np.sum((combine_kaiser_spectra(kcut, power_lin_dict, D, bias, f_growth, rec_algo, R, rsd=rsd)[:ellmax, :, 0] -
                                combine_cross_kaiser_spectra(kcut, power_rsd_dict, D, bias, f_growth, rec_algo, R, rsd=rsd)[:ellmax, :, 0])**2/
                               (2 * combine_kaiser_spectra(kcut, power_lin_dict, D, bias, f_growth, rec_algo, R, rsd=rsd)[:ellmax, :, 0]**2))
    out = minimize(loss, 1.)
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
        nmodes = power_tr_dict['N_ell_tr_tr'].flatten()
    else:
        pk_tt[0, :, :] = power_tr_dict['P_kmu_tr_tr'].reshape(len(k), 1)
        nmodes = power_tr_dict['N_kmu_tr_tr'].flatten()
    
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
    return k, mu, pk_tt, pk_ij_zz, pk_ij_zt, nmodes

def get_cfg(sim_name, z_this, nmesh):
    meta = get_meta(sim_name, redshift=z_this)
    Lbox = meta['BoxSize']
    z_ic = meta['InitialRedshift']
    k_Ny = np.pi*nmesh/Lbox
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
            

    # create a dict with everything you would ever need
    cfg = {'lbox': Lbox, 'Cosmology': cosmo, 'z_ic': z_ic}
    return cfg

def run_zcv(power_rsd_tr_dict, power_rsd_ij_dict, power_tr_dict, power_ij_dict, config):

    # read out some parameters from the config function
    sim_name = config['sim_params']['sim_name']
    z_this = config['sim_params']['z_mock']
    zcv_dir = config['zcv_params']['zcv_dir']
    nmesh = config['zcv_params']['nmesh']
    kcut = config['zcv_params']['kcut']
    keynames = config['zcv_params']['fields']
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

    # linear power
    pk_lin_fn = save_dir / "abacus_pk_lin_ic.dat"

    # read the config params
    cfg = get_cfg(sim_name, z_this, nmesh)
    cfg['p_lin_ic_file'] = str(pk_lin_fn)
    cfg['nmesh_in'] = nmesh
    cfg['poles'] = poles
    cfg['surrogate_gaussian_cutoff'] = kcut
    Lbox = cfg['lbox']

    # define k bins
    k_bins, mu_bins = get_k_mu_edges(Lbox, k_hMpc_max, n_k_bins, n_mu_bins, logk)
    k_binc = (k_bins[1:] + k_bins[:-1])*.5
    
    # name of files to read from
    if not logk:
        dk = k_bins[1]-k_bins[0]
    else:
        dk = np.log(k_bins[1]/k_bins[0])
    if n_k_bins == nmesh//2:
        zenbu_fn = save_z_dir / f"zenbu_pk{rsd_str}_ij_lpt_nmesh{nmesh:d}.npz"
        window_fn = save_dir / f"window_nmesh{nmesh:d}.npz"
    else:
        zenbu_fn = save_z_dir / f"zenbu_pk{rsd_str}_ij_lpt_nmesh{nmesh:d}_dk{dk:.3f}.npz"
        window_fn = save_dir / f"window_nmesh{nmesh:d}_dk{dk:.3f}.npz"
   
    
    # read out the dictionaries 
    if want_rsd: # then we need the no-rsd version (names are misleading)
        k, mu, pk_tt, pk_ij_zz, pk_ij_zt, nmodes = read_power_dict(power_tr_dict, power_ij_dict, want_rsd=False, keynames=keynames, poles=poles)
    k, mu, pk_tt_poles, pk_ij_zz_poles, pk_ij_zt_poles, nmodes = read_power_dict(power_rsd_tr_dict, power_rsd_ij_dict, want_rsd=want_rsd, keynames=keynames, poles=poles)
    print(len(k), len(k_binc))
    assert len(k) == len(k_binc)

    # load the linear power spectrum
    p_in = np.genfromtxt(cfg['p_lin_ic_file'])
    kth, p_m_lin = p_in[:,0], p_in[:,1]

    # fit to get the biases
    kmax = 0.15
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

    # reduce variance of measurement
    bias_vec = np.array(bvec_opt_rs['x']) # b1, b2, bs, bn, shot (not used)
    #bias_vec[1:] = 0. # set to 0 all but b1
    #bias_vec[2:] = 0. # set to 0 all but b1 and b2
    #bias_vec[3:] = 0. # set to 0 all but b1, b2, bs
    bias_vec = np.hstack(([1.], bias_vec))
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
    r_zt_smooth_nohex, pk_zn, pk_ij_zenbu_poles, pkclass, r_zt_sn_lim = reduce_variance_tt(k_binc, pk_tt_input, pk_ij_zt_input,
                                                                              pk_ij_zz_input, cfg, z_this,
                                                                              bias_vec, kth, p_m_lin, rsd=want_rsd,
                                                                              win_fac=1, kout=k_bins,
                                                                              window=window, exact_window=window_exact,
                                                                              pk_ij_zenbu=pk_ij_zenbu, lptobj=lptobj)

    zcv_dict = {}
    zcv_dict['k_binc'] = k_binc
    zcv_dict['poles'] = poles
    zcv_dict['rho_tr_ZD'] = r_zt
    zcv_dict['rho_tr_ZD_sn_lim'] = r_zt_sn_lim
    zcv_dict['Pk_ZD_ZD_ell'] = pk_zz
    zcv_dict['Pk_tr_ZD_ell'] = pk_zn
    zcv_dict['Pk_tr_tr_ell'] = pk_tt_poles
    zcv_dict['Nk_tr_tr_ell'] = nmodes
    zcv_dict['Pk_tr_tr_ell_zcv'] = pk_nn_betasmooth
    zcv_dict['Pk_ZD_ZD_ell_ZeNBu'] = pk_zenbu
    return zcv_dict

def expand_poles_to_3d(k_binc, pk_zenbu, k_box, mu_box, only_ell0=False):
    
    # kbox mu_box
    p0spline = interp1d(k_binc.astype(np.float32), pk_zenbu[0].astype(np.float32), fill_value='extrapolate')
    if not only_ell0:
        p2spline = interp1d(k_binc.astype(np.float32), pk_zenbu[1].astype(np.float32), fill_value='extrapolate')
        p4spline = interp1d(k_binc.astype(np.float32), pk_zenbu[2].astype(np.float32), fill_value='extrapolate')

    # interpolate zenbu on the box
    pk_zb = np.zeros(k_box.shape, dtype=np.float32)
    pk_zb[:] += p0spline(k_box) # *legendre(0)(mu_box)
    gc.collect()
    if not only_ell0:
        pk_zb[:] += p2spline(k_box)*legendre(2)(mu_box)
        gc.collect()
        pk_zb[:] += p4spline(k_box)*legendre(4)(mu_box)
        gc.collect()
    #pk_zb[:] += p0spline(k_box)*legendre(0)(mu_box) + p2spline(k_box)*legendre(2)(mu_box) + p4spline(k_box)*legendre(4)(mu_box)
    return pk_zb

def run_zcv_field(power_rsd_tr_fns, power_rsd_ij_fns, power_tr_fns, power_ij_fns, config):

    # read out some parameters from the config function
    sim_name = config['sim_params']['sim_name']
    z_this = config['sim_params']['z_mock']
    zcv_dir = config['zcv_params']['zcv_dir']
    nmesh = config['zcv_params']['nmesh']
    kcut = config['zcv_params']['kcut']
    keynames = config['zcv_params']['fields']
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

    # linear power
    pk_lin_fn = save_dir / "abacus_pk_lin_ic.dat"

    # read the config params
    cfg = get_cfg(sim_name, z_this, nmesh)
    cfg['p_lin_ic_file'] = str(pk_lin_fn)
    cfg['nmesh_in'] = nmesh
    cfg['poles'] = poles
    cfg['surrogate_gaussian_cutoff'] = kcut
    Lbox = cfg['lbox']

    # parameter for fitting bias
    kmax = 0.15

    # name of files to read from could add the _dk version
    zenbu_fn = save_z_dir / f"zenbu_pk{rsd_str}_ij_lpt_nmesh{nmesh:d}.npz"
    window_fn = save_dir / f"window_nmesh{nmesh:d}.npz"

    # file to save to
    power_cv_tr_fn = Path(save_z_dir) / f"power{rsd_str}_ZCV_tr_nmesh{nmesh:d}.asdf"
    
    # load the linear power spectrum (unused)
    #p_in = np.genfromtxt(cfg['p_lin_ic_file'])
    #kth, p_m_lin = p_in[:,0], p_in[:,1]

    # get the box k and mu modes
    n_perp = n_los = nmesh
    k_box, mu_box, k_bins, _ = get_k_mu_box_edges(Lbox, n_perp, n_los, n_k_bins, n_mu_bins, k_hMpc_max, logk) # could get rid of kbins
    k_binc = 0.5*(k_bins[1:]+k_bins[:-1])
    
    # fit in real space
    pk_nn = asdf.open(power_tr_fns[0])['data'][f'P_k3D_tr_tr']
    pk_nn = project_3d_to_poles(k_bins, k_box, mu_box, logk, pk_nn.flatten(), Lbox, poles=[0])[0].flatten()[k_binc < kmax]/Lbox**3
    pk_ij = {}
    counter = 0
    # TESTING!!!!!!!!!!!!
    keynames = ["1cb", "delta"]
    for i in range(len(keynames)):
        for j in range(len(keynames)):
            if i < j: continue
            print("projecting", i, j)
            pk = asdf.open(power_ij_fns[counter])['data'][f'P_k3D_{keynames[i]}_{keynames[j]}']
            pk = project_3d_to_poles(k_bins, k_box, mu_box, logk, pk.flatten(), Lbox, poles=[0])
            pk_ij[f"{keynames[i]}_{keynames[i]}"] = pk[0].flatten()[k_binc < kmax]/Lbox**3
            nmodes = pk[1].flatten()
            counter += 1

    # bias fitting
    def predict_power(bvec):
        bias = np.hstack((1, bvec))
        power = np.zeros_like(pk_ij[f"{keynames[0]}_{keynames[0]}"])
        for i in range(len(keynames)):
            for j in range(len(keynames)):
                if i < j: continue
                if i == j:
                    power += bias[i]*bias[j]*pk_ij[f"{keynames[i]}_{keynames[i]}"]
                else:
                    power += 2.*bias[i]*bias[j]*pk_ij[f"{keynames[i]}_{keynames[i]}"]
        power += bias[-1]
        return power
    sum_frac = lambda bvec: np.sum((pk_nn-predict_power(bvec))**2/pk_nn**2)
    bvec0 = np.zeros(len(keynames)) # because it's b1, b2, ..., shotnoise
    bvec_opt_rs = minimize(sum_frac, bvec0, method='Nelder-Mead')
    print(bvec_opt_rs)
    #np.savez("power_proj.npz", k_binc = 0.5*(k_bins[1:]+k_bins[:-1]), pk_ij=pk_ij, pk_nn=pk_nn, nmodes=nmodes)

    """
    # fit to get the biases
    if want_rsd:
        #bvec_opt = measure_field_bias_rsd(power_rsd_tr_fns, power_rsd_ij_fns, k_box, keynames, kmax)
        bvec_opt_rs = measure_field_bias(power_tr_fns, power_ij_fns, k_box, keynames, kmax)
        #print("bias zspace", bvec_opt['x'])
    else:
        # names are misleading
        bvec_opt_rs = measure_field_bias(power_tr_fns, power_ij_fns, k_box, keynames, kmax)
    print("bias rspace", bvec_opt_rs['x'])
    """
    
    # load the presaved window function
    data = np.load(window_fn)
    window = data['window']
    window_exact = False

    # load the presaved zenbu power spectra
    data = np.load(zenbu_fn)
    pk_ij_zenbu = data['pk_ij_zenbu']
    lptobj = data['lptobj']
    
    # define k bins
    k_bins, mu_bins = get_k_mu_edges(Lbox, k_hMpc_max, n_k_bins, n_mu_bins, logk)
    k_binc = (k_bins[1:] + k_bins[:-1])*.5

    # reduce variance of measurement
    bias_vec = np.array(bvec_opt_rs['x']) # b1, b2, bs, bn, shot (not used)
    bias_vec = np.hstack(([1.], bias_vec))
    #bias_vec[1] = 0.3 
    
    # combine zenbu in ell space and then convert to 3D power
    if want_rsd:
        if len(bias_vec)<11:
            bias_vec = np.hstack([bias_vec, np.zeros(11-len(bias_vec))])
    else:
        if len(bias_vec)<6:
            bias_vec = np.hstack([bias_vec, np.zeros(6-len(bias_vec))])
    pk_zenbu = combine_spectra(k_binc, pk_ij_zenbu, bias_vec[1:], rsd=want_rsd)
    print("should be trivial")

    # interpolate to 3D
    #pk_zb = expand_poles_to_3d(k_binc, pk_zenbu, k_box, mu_box)/Lbox**3
    #pk_zb = pk_zb.reshape(nmesh, nmesh, nmesh)
    #print("pk_zb", pk_zb[10:12, 10:12, 10:12])

    # combine
    assert want_rsd
    pk_nn = asdf.open(power_rsd_tr_fns[0])['data'][f'P_k3D_tr_tr']
    pk_zz = combine_field_spectra_full(bias_vec, power_rsd_ij_fns, keynames)
    pk_zn = combine_field_cross_spectra_full(bias_vec, power_rsd_tr_fns, keynames)
    print("got pk zz, zn zz")

    pk_nn_proj = project_3d_to_poles(k_bins, k_box, mu_box, logk, pk_nn.flatten(), Lbox, poles)[0].reshape(len(poles), len(k_binc))/Lbox**3
    pk_zn_proj = project_3d_to_poles(k_bins, k_box, mu_box, logk, pk_zn.flatten(), Lbox, poles)[0].reshape(len(poles), len(k_binc))/Lbox**3
    del pk_zn; gc.collect()
    pk_zz_proj = project_3d_to_poles(k_bins, k_box, mu_box, logk, pk_zz.flatten(), Lbox, poles)[0].reshape(len(poles), len(k_binc))/Lbox**3
    pk_zz[:, :, :] -= (expand_poles_to_3d(k_binc, pk_zenbu, k_box, mu_box)/Lbox**3).reshape(nmesh, nmesh, nmesh)
    
    
    # TESTING!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    sg_window = 21; k0 = 0.618; dk = 0.167; beta1_k = 0.05
    #beta_smooth = (1/2 * (1 - np.tanh((k_box - k0)/dk))).reshape(nmesh, nmesh, nmesh)
    #beta_smooth = 1. # it is possible this is better see elg 0 through 3 and 4 through 7 or 6 have some tanh and after we set to 1
    # from function moved here
    if want_rsd:
        cov_zn = np.stack([multipole_cov(pk_zn_proj, ell) for ell in poles])
        var_zz = np.stack([multipole_cov(pk_zz_proj, ell) for ell in poles])
        var_nn = np.stack([multipole_cov(pk_nn_proj, ell) for ell in poles])
    else:
        cov_zn = 2 * pk_zn_proj ** 2
        var_zz = 2 * pk_zz_proj ** 2
        var_nn = 2 * pk_nn_proj ** 2
    beta_proj = cov_zn / var_zz
    r_zt_proj = cov_zn / np.sqrt(var_zz * var_nn) # need to figure out why infinity is first value
    beta_damp = 1/2 * (1 - np.tanh((k_binc - k0)/dk)) * beta_proj
    beta_damp = np.atleast_2d(beta_damp)
    r_zt_proj = np.atleast_2d(r_zt_proj)
    beta_damp[:, :k_binc.searchsorted(beta1_k)] = 1.
    beta_smooth = np.zeros_like(beta_damp)
    for i in range(beta_smooth.shape[0]):
        beta_smooth[i, :] = savgol_filter(beta_damp.T[:, i], sg_window, 3)
    #np.savez(f"beta_smooth_z{z_this:.3f}.npz", beta_smooth=beta_smooth, beta_proj=beta_proj, beta_damp=beta_damp, k_binc=k_binc, pk_zn=pk_zn_proj, pk_nn=pk_nn_proj, pk_zz=pk_zz_proj, r_zt=r_zt_proj)
    #quit()
    
    #beta_damp = expand_poles_to_3d(k_binc, beta_damp, k_box, mu_box).reshape(pk_nn.shape).astype(np.float32)
    beta_smooth = expand_poles_to_3d(k_binc, beta_smooth, k_box, mu_box, only_ell0=True).reshape(pk_nn.shape)
    #p = np.poly1d(np.polyfit(k_binc, beta_smooth[0], 9)) # TODO: should porbably do for all k's
    #beta_smooth = p(k_box)
    #beta_smooth[k_box < beta1_k] = 1.
    
    # get reduced fields
    pk_nn -= beta_smooth * pk_zz
    del beta_smooth; gc.collect()
    del pk_zz; gc.collect()
    
    # save 3d
    pk_tr_dict = {}
    pk_tr_dict[f'P_k3D_tr_tr_zcv'] = pk_nn
    header = {}
    header['sim_name'] = sim_name
    header['Lbox'] = Lbox
    header['nmesh'] = nmesh
    header['kcut'] = kcut
    compress_asdf(str(power_cv_tr_fn), pk_tr_dict, header)
    print("compressed")
    
    # project to multipoles
    pk_nn_betasmooth, nmodes = project_3d_to_poles(k_bins, k_box, mu_box, logk, pk_nn.flatten(), Lbox, poles)
    pk_nn = pk_nn_proj.reshape(1, len(poles), len(k_binc))
    pk_zz = pk_zz_proj
    pk_zn = pk_zn_proj
    r_zt = r_zt_proj
    
    # changing format (note that projection multiplies by L^3, so we get rid of that)
    pk_nn_betasmooth = pk_nn_betasmooth.reshape(len(poles), len(k_binc))/Lbox**3
    pk_zenbu = pk_zenbu.reshape(len(poles), len(k_binc))/Lbox**3
    nmodes = nmodes.flatten()[:len(k_binc)]
    
    # save result
    zcv_dict = {}
    zcv_dict['k_binc'] = k_binc
    zcv_dict['poles'] = poles
    zcv_dict['rho_tr_ZD'] = r_zt
    zcv_dict['Pk_ZD_ZD_ell'] = pk_zz
    zcv_dict['Pk_tr_ZD_ell'] = pk_zn
    zcv_dict['Pk_tr_tr_ell'] = pk_nn
    zcv_dict['Nk_tr_tr_ell'] = nmodes
    zcv_dict['Pk_tr_tr_ell_zcv'] = pk_nn_betasmooth
    zcv_dict['Pk_ZD_ZD_ell_ZeNBu'] = pk_zenbu
    
    return zcv_dict

def run_lcv(power_rsd_tr_dict, power_lin_dict, config):

    # read out some parameters from the config function
    sim_name = config['sim_params']['sim_name']
    z_this = config['sim_params']['z_mock']
    lcv_dir = config['lcv_params']['lcv_dir']
    nmesh = config['lcv_params']['nmesh']
    kcut = config['lcv_params']['kcut']
    want_rsd = config['HOD_params']['want_rsd']
    rsd_str = "_rsd" if want_rsd else ""
    
    # power params
    k_hMpc_max = config['power_params']['k_hMpc_max']
    logk = config['power_params']['logk']
    n_k_bins = config['power_params']['nbins_k']
    n_mu_bins = config['power_params']['nbins_mu']
    poles = config['power_params']['poles']

    # reconstruction algorithm
    rec_algo = config['HOD_params']['rec_algo']
    if rec_algo == 'recsym':
        R = None
    elif rec_algo == "reciso":
        R = config['HOD_params']['smoothing']
    
    # create save directory
    save_dir = Path(lcv_dir) / sim_name
    save_z_dir = save_dir / f"z{z_this:.3f}"

    # read meta data
    meta = get_meta(sim_name, redshift=z_this)
    Lbox = meta['BoxSize']
    z_ic = meta['InitialRedshift']
    D_ratio = meta['GrowthTable'][z_ic]/meta['GrowthTable'][1.0]
    k_Ny = np.pi*nmesh/Lbox
    cosmo = {}
    cosmo['output'] = 'mPk mTk'
    cosmo['P_k_max_h/Mpc'] = 20.
    phase = int(sim_name.split('ph')[-1])
    for k in ('H0', 'omega_b', 'omega_cdm',
              'omega_ncdm', 'N_ncdm', 'N_ur',
              'n_s', 'A_s', 'alpha_s',
              #'wa', 'w0',
    ):
        cosmo[k] = meta[k]
    
    # load input linear power
    kth = meta['CLASS_power_spectrum']['k (h/Mpc)']
    pk_z1 = meta['CLASS_power_spectrum']['P (Mpc/h)^3']
    print("how many ks are saved", len(kth))
    p_m_lin = D_ratio**2*pk_z1

    # apply gaussian cutoff to linear power
    p_m_lin *= np.exp(-(kth/kcut)**2)
    
    # compute growth factor
    pkclass = Class()
    pkclass.set(cosmo)
    pkclass.compute()
    D = pkclass.scale_independent_growth_factor(z_this)
    D /= pkclass.scale_independent_growth_factor(z_ic)
    Ha = pkclass.Hubble(z_this) * 299792.458
    if want_rsd:
        f_growth = pkclass.scale_independent_growth_factor_f(z_this)
    else:
        f_growth = 0.
    print("D, f = ", D, f_growth)
   
    # define k bins
    k_bins, mu_bins = get_k_mu_edges(Lbox, k_hMpc_max, n_k_bins, n_mu_bins, logk)
    k_binc = (k_bins[1:] + k_bins[:-1])*.5
    if not logk:
        dk = k_bins[1]-k_bins[0]
    else:
        dk = np.log(k_bins[1]/k_bins[0])

    # name of files to read from
    if n_k_bins == nmesh//2:
        window_fn = save_dir / f"window_nmesh{nmesh:d}.npz"
    else:
        window_fn = save_dir / f"window_nmesh{nmesh:d}_dk{dk:.3f}.npz"
        
    # get the bias
    kmax = 0.08 
    bvec_opt = measure_2pt_bias_lin(k_binc, power_lin_dict, power_rsd_tr_dict, D, f_growth, kmax, want_rsd, rec_algo, R, ellmax=1)
    #bvec_opt = measure_2pt_bias_cross_lin(k_binc, power_lin_dict, power_rsd_tr_dict, D, f_growth, kmax, want_rsd, rec_algo, R, ellmax=1)
    bias = np.array(bvec_opt['x'])[0]
    print("bias", bias)
    print(bvec_opt)
    #bias = 2.05 # used for 009 cause gives weird bias and 023
    
    # get linear prediction
    if rec_algo == "reciso":
        S = get_smoothing(kth, R)
        f_eff = f_growth*(1.-S)
    elif rec_algo == "recsym":
        f_eff = f_growth
    kth, p_m_lin_poles = get_poles(kth, p_m_lin, D, bias, f_eff, poles=poles)
    if want_rsd:
        p_m_lin_input = []
        for i in range(len(poles)):
            p_m_lin_input.append(interp1d(kth, p_m_lin_poles[i], fill_value='extrapolate')(k_binc))
        p_m_lin_input = np.array(p_m_lin_input)
    else:
        print("not implemented"); quit()
    
    # convert into kaiser-corrected power spectra
    print(p_m_lin_input.shape)
    print(power_rsd_tr_dict['P_ell_tr_tr'].shape, len(k_binc))
    pk_ll_input = combine_kaiser_spectra(k_binc, power_lin_dict, D, bias, f_growth, rec_algo, R, rsd=want_rsd).reshape(len(poles), len(k_binc))
    pk_tl_input = combine_cross_kaiser_spectra(k_binc, power_rsd_tr_dict, D, bias, f_growth, rec_algo, R, rsd=want_rsd).reshape(len(poles), len(k_binc))
    pk_tt_input = power_rsd_tr_dict['P_ell_tr_tr'].reshape(len(poles), len(k_binc))
    nmodes = power_rsd_tr_dict['N_ell_tr_tr'].flatten()
    
    # load the presaved window function
    data = np.load(window_fn)
    window = data['window']

    # reduce variance
    pk_tt_hat, pk_tt_betasmooth, pk_tt_betasmooth_nohex,\
    pk_tt_beta1, beta, beta_damp, beta_smooth, beta_smooth_nohex,\
    pk_ll, p_m_lin, r_tl, r_tl_smooth, r_tl_smooth_nohex, pk_tl, r_tl_sn_lim = reduce_variance_lin(k_binc, pk_tt_input, pk_tl_input, pk_ll_input, p_m_lin_input, window=window,
                                                                                      s_ell=[0.1, 10, 100], sg_window=21, rsd=want_rsd,
                                                                                      k0=0.618, dk=0.167, beta1_k=0.05, poles=poles)
    
    # save output in a dictionary
    lcv_dict = {}
    lcv_dict['k_binc'] = k_binc
    lcv_dict['poles'] = poles
    lcv_dict['rho_tr_lf'] = r_tl
    lcv_dict['rho_tr_lf_sn_lim'] = r_tl_sn_lim
    lcv_dict['Pk_lf_lf_ell'] = pk_ll_input
    lcv_dict['Pk_tr_lf_ell'] = pk_tl_input
    lcv_dict['Pk_tr_tr_ell'] = pk_tt_input
    lcv_dict['Nk_tr_tr_ell'] = nmodes
    lcv_dict['Pk_tr_tr_ell_lcv'] = pk_tt_betasmooth
    lcv_dict['Pk_lf_lf_ell_CLASS'] = p_m_lin_input
    return lcv_dict

def run_lcv_field(power_rsd_tr_fns, power_lin_fns, config):
    
    # read out some parameters from the config function
    sim_name = config['sim_params']['sim_name']
    z_this = config['sim_params']['z_mock']
    lcv_dir = config['lcv_params']['lcv_dir']
    nmesh = config['lcv_params']['nmesh']
    kcut = config['lcv_params']['kcut']
    want_rsd = config['HOD_params']['want_rsd']
    rsd_str = "_rsd" if want_rsd else ""
    
    # power params
    k_hMpc_max = config['power_params']['k_hMpc_max']
    logk = config['power_params']['logk']
    n_k_bins = config['power_params']['nbins_k']
    n_mu_bins = config['power_params']['nbins_mu']
    poles = config['power_params']['poles']

    # reconstruction algorithm
    rec_algo = config['HOD_params']['rec_algo']
    if rec_algo == 'recsym':
        R = None
    elif rec_algo == "reciso":
        R = config['HOD_params']['smoothing']
    
    # create save directory
    save_dir = Path(lcv_dir) / sim_name
    save_z_dir = save_dir / f"z{z_this:.3f}"

    # read meta data
    meta = get_meta(sim_name, redshift=z_this)
    Lbox = meta['BoxSize']
    z_ic = meta['InitialRedshift']
    D_ratio = meta['GrowthTable'][z_ic]/meta['GrowthTable'][1.0]
    k_Ny = np.pi*nmesh/Lbox
    cosmo = {}
    cosmo['output'] = 'mPk mTk'
    cosmo['P_k_max_h/Mpc'] = 20.
    phase = int(sim_name.split('ph')[-1])
    for k in ('H0', 'omega_b', 'omega_cdm',
              'omega_ncdm', 'N_ncdm', 'N_ur',
              'n_s', 'A_s', 'alpha_s',
              #'wa', 'w0',
    ):
        cosmo[k] = meta[k]

    # fitting parameter
    kmax = 0.08
    
    # load input linear power
    kth = meta['CLASS_power_spectrum']['k (h/Mpc)']
    pk_z1 = meta['CLASS_power_spectrum']['P (Mpc/h)^3']
    print("how many ks are saved", len(kth))
    p_m_lin = D_ratio**2*pk_z1

    # apply gaussian cutoff to linear power
    p_m_lin *= np.exp(-(kth/kcut)**2)
    
    # compute growth factor
    pkclass = Class()
    pkclass.set(cosmo)
    pkclass.compute()
    D = pkclass.scale_independent_growth_factor(z_this)
    D /= pkclass.scale_independent_growth_factor(z_ic)
    Ha = pkclass.Hubble(z_this) * 299792.458
    if want_rsd:
        f_growth = pkclass.scale_independent_growth_factor_f(z_this)
    else:
        f_growth = 0.
    print("D, f = ", D, f_growth)
   
    # define k bins
    k_bins, mu_bins = get_k_mu_edges(Lbox, k_hMpc_max, n_k_bins, n_mu_bins, logk)
    k_binc = (k_bins[1:] + k_bins[:-1])*.5
    if not logk:
        dk = k_bins[1]-k_bins[0]
    else:
        dk = np.log(k_bins[1]/k_bins[0])

    # name of files to read from
    window_fn = save_dir / f"window_nmesh{nmesh:d}.npz"
    
    # file to save to
    power_cv_tr_fn = Path(save_z_dir) / f"power{rsd_str}_LCV_tr_{rec_algo}_nmesh{nmesh:d}.asdf"

    # get the box k and mu modes
    n_perp = n_los = nmesh
    k_box, mu_box, _, _ = get_k_mu_box_edges(Lbox, n_perp, n_los, n_k_bins, n_mu_bins, k_hMpc_max, logk) # could get rid of kbins
    
    # compute bias
    pk_tt = asdf.open(power_rsd_tr_fns[0])['data'][f'P_k3D_tr_tr']
    pk_tt = project_3d_to_poles(k_bins, k_box, mu_box, logk, pk_tt.flatten(), Lbox, poles=[0])[0].flatten()[k_binc < kmax]/Lbox**3
    keynames = ['delta', 'deltamu2']
    pk_ij = {}
    counter = 0
    for i in range(len(keynames)):
        for j in range(len(keynames)):
            if i < j: continue
            print("projecting", i, j)
            pk = asdf.open(power_lin_fns[counter])['data'][f'P_k3D_{keynames[i]}_{keynames[j]}']
            pk = project_3d_to_poles(k_bins, k_box, mu_box, logk, pk.flatten(), Lbox, poles=[0])
            pk_ij[f"{keynames[i]}_{keynames[j]}"] = pk[0].flatten()[k_binc < kmax]/Lbox**3
            nmodes = pk[1].flatten()
            counter += 1

    # minimize
    def predict_power(bias):
        if rec_algo == "recsym":
            f_eff = f_growth
        elif rec_algo == "reciso":
            assert R is not None
            S = get_smoothing(k_binc, R)
            f_eff = f_growth * (1.-S)
        pk = D**2 * (2. * bias * f_eff * pk_ij['deltamu2_delta'] + f_eff**2 * pk_ij['deltamu2_deltamu2'] + bias**2 * pk_ij['delta_delta'])
        return pk    
    sum_frac = lambda b: np.sum((pk_tt-predict_power(b))**2/pk_tt**2)
    bvec_opt = minimize(sum_frac, 1., method='Nelder-Mead')
    bias = np.array(bvec_opt['x'])[0]
    print(bvec_opt)
    
    # get linear prediction
    if rec_algo == "reciso":
        S = get_smoothing(kth, R)
        f_eff = f_growth*(1.-S)
    elif rec_algo == "recsym":
        f_eff = f_growth
    kth, p_m_lin_poles = get_poles(kth, p_m_lin, D, bias, f_eff, poles=poles)
    
    # combine
    # get linear prediction
    if rec_algo == "reciso":
        S = get_smoothing(k_box, R)
        f_eff = f_growth*(1.-S)
        f_eff = f_eff.reshape(nmesh, nmesh, nmesh)
    elif rec_algo == "recsym":
        f_eff = f_growth
    
    assert want_rsd
    pk_tt = asdf.open(power_rsd_tr_fns[0])['data'][f'P_k3D_tr_tr']
    pk_ll = D**2 * (2. * bias * f_eff * asdf.open(power_lin_fns[1])['data']['P_k3D_deltamu2_delta'] + f_eff**2 * asdf.open(power_lin_fns[2])['data']['P_k3D_deltamu2_deltamu2'] + bias**2 * asdf.open(power_lin_fns[0])['data']['P_k3D_delta_delta'])
    pk_lt = D * (bias * asdf.open(power_rsd_tr_fns[1])['data']['P_k3D_delta_tr'] + f_eff * asdf.open(power_rsd_tr_fns[2])['data']['P_k3D_deltamu2_tr'])
    print("got pk ll, pk tt pk lt")
    del f_eff; gc.collect()

    # project to poles
    pk_lt_proj = project_3d_to_poles(k_bins, k_box, mu_box, logk, pk_lt.flatten(), Lbox, poles)[0].reshape(len(poles), len(k_binc))/Lbox**3
    del pk_lt; gc.collect()
    pk_tt_proj = project_3d_to_poles(k_bins, k_box, mu_box, logk, pk_tt.flatten(), Lbox, poles)[0].reshape(len(poles), len(k_binc))/Lbox**3
    pk_ll_proj = project_3d_to_poles(k_bins, k_box, mu_box, logk, pk_ll.flatten(), Lbox, poles)[0].reshape(len(poles), len(k_binc))/Lbox**3
    pk_ll[:, :, :] -= (expand_poles_to_3d(kth, p_m_lin_poles, k_box, mu_box)/Lbox**3).reshape(nmesh, nmesh, nmesh)
            
    # simple beta function
    #sg_window = 21; k0 = 0.618; dk = 0.167; beta1_k=0.05
    sg_window = 21; k0 = 1.18; dk = 0.167; beta1_k=0.05
    #beta_smooth = (1/2 * (1 - np.tanh((k_box - k0)/dk))).reshape(nmesh, nmesh, nmesh)
    #beta_smooth = 1.
    if want_rsd:
        cov_lt = np.stack([multipole_cov(pk_lt_proj, ell) for ell in poles])
        var_ll = np.stack([multipole_cov(pk_ll_proj, ell) for ell in poles])
        var_tt = np.stack([multipole_cov(pk_tt_proj, ell) for ell in poles])
    else:
        cov_lt = 2 * pk_lt_proj ** 2
        var_ll = 2 * pk_ll_proj ** 2
        var_tt = 2 * pk_tt_proj ** 2
    beta_proj = cov_lt / var_ll
    r_lt_proj = cov_lt / np.sqrt(var_ll * var_tt) # need to figure out why infinity is first value
    beta_damp = 1/2 * (1 - np.tanh((k_binc - k0)/dk)) * beta_proj
    beta_damp = np.atleast_2d(beta_damp)
    r_lt_proj = np.atleast_2d(r_lt_proj)
    beta_damp[:, :k_binc.searchsorted(beta1_k)] = 1.
    beta_smooth = np.zeros_like(beta_damp)
    for i in range(beta_smooth.shape[0]):
        beta_smooth[i, :] = savgol_filter(beta_damp.T[:, i], sg_window, 3)
    #beta_damp = expand_poles_to_3d(k_binc, beta_damp, k_box, mu_box).reshape(pk_tt.shape).astype(np.float32)
    beta_smooth = expand_poles_to_3d(k_binc, beta_smooth, k_box, mu_box, only_ell0=True).reshape(pk_tt.shape)
    print("beta", beta_smooth[10:12, 10:12, 10:12])
    
    # get reduced fields
    pk_tt -= beta_smooth * pk_ll
    del beta_smooth; gc.collect()
    del pk_ll; gc.collect()

    # save 3d
    pk_tr_dict = {}
    pk_tr_dict[f'P_k3D_tr_tr_lcv'] = pk_tt
    header = {}
    header['sim_name'] = sim_name
    header['Lbox'] = Lbox
    header['nmesh'] = nmesh
    header['kcut'] = kcut
    compress_asdf(str(power_cv_tr_fn), pk_tr_dict, header)
    print("compressed")

    # project to multipoles
    pk_tt_betasmooth, nmodes = project_3d_to_poles(k_bins, k_box, mu_box, logk, pk_tt.flatten(), Lbox, poles)
    pk_tt = pk_tt_proj.reshape(1, len(poles), len(k_binc))
    pk_ll = pk_ll_proj
    pk_lt = pk_lt_proj

    # changing format (note that projection multiplies by L^3, so we get rid of that)
    pk_tt_betasmooth = pk_tt_betasmooth.reshape(len(poles), len(k_binc))/Lbox**3
    nmodes = nmodes.flatten()[:len(k_binc)]

    # interpolate
    p_m_lin_input = np.zeros((len(poles), len(k_binc)))
    for i in range(len(poles)):
        p_m_lin_input[i] = (interp1d(kth, p_m_lin_poles[i], fill_value='extrapolate')(k_binc))/Lbox**3
    
    # save output in a dictionary
    lcv_dict = {}
    lcv_dict['k_binc'] = k_binc
    lcv_dict['poles'] = poles
    lcv_dict['rho_tr_lf'] = r_lt_proj
    lcv_dict['Pk_lf_lf_ell'] = pk_ll_proj
    lcv_dict['Pk_tr_lf_ell'] = pk_lt_proj
    lcv_dict['Pk_tr_tr_ell'] = pk_tt_proj
    lcv_dict['Nk_tr_tr_ell'] = nmodes
    lcv_dict['Pk_tr_tr_ell_lcv'] = pk_tt_betasmooth
    lcv_dict['Pk_lf_lf_ell_CLASS'] = p_m_lin_input
    return lcv_dict
