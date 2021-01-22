#!/usr/bin/env python

"""
Script for calculating the HOD function for satellite
and central galaxies of the following three types:
 - luminous red galaxies (LRGs)
 - emission-line galaxies (ELGs)
 - quasi stellar objects (QSOs)

Usage:
------
$ ./tracer_fun --help
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.special import erf
import argparse


def N_sat(M_h, M_cut, kappa, M_1, alpha, A_s=1., **kwargs):
    """
    Standard Zheng et al. (2005) satellite HOD parametrization for all tracers with an optional amplitude parameter, A_s.
    """
    N = A_s*((M_h-kappa*M_cut)/M_1)**alpha
    return N

def N_cen_LRG(M_h, M_cut, sigma, **kwargs):
    """
    Standard Zheng et al. (2005) central HOD parametrization for LRGs.
    """
    N = 0.5*(1 + erf((np.log10(M_h)-np.log10(M_cut))/sigma))
    return N

def N_cen_ELG_v1(M_h, p_max, Q, M_cut, sigma, gamma, **kwargs):
    """
    HOD function for ELG centrals taken from arXiv:1910.05095.
    """
    phi = phi_fun(M_h, M_cut, sigma)
    Phi = Phi_fun(M_h, M_cut, sigma, gamma)
    A = A_fun(p_max, Q, phi, Phi)
    N = 2.*A*phi*Phi + 1./(2.*Q)*(1 + erf((np.log10(M_h)-np.log10(M_cut))/0.01))
    return N

def N_cen_ELG_v2(M_h, p_max, M_cut, sigma, gamma, **kwargs):
    """
    HOD function for ELG centrals taken from arXiv:2007.09012.
    """
    N = np.zeros(len(M_h))
    N[M_h <= M_cut] = p_max*Gaussian_fun(np.log10(M_h[M_h <= M_cut]), np.log10(M_cut), sigma)
    N[M_h > M_cut] = p_max*(M_h[M_h > M_cut]/M_cut)**gamma/(np.sqrt(2.*np.pi)*sigma)
    return N

def N_cen_QSO(M_h, p_max, M_cut, sigma, **kwargs):
    """
    HOD function (Zheng et al. (2005) with p_max) for QSO centrals taken from arXiv:2007.09012.
    """
    N = 0.5*p_max*(1 + erf((np.log10(M_h)-np.log10(M_cut))/sigma))
    return N


def phi_fun(M_h, M_cut, sigma):
    """
    Aiding function for N_cen_ELG_v1().
    """
    phi = Gaussian_fun(np.log10(M_h),np.log10(M_cut), sigma)
    return phi

def Phi_fun(M_h, M_cut, sigma, gamma):
    """
    Aiding function for N_cen_ELG_v1().
    """
    x = gamma*(np.log10(M_h)-np.log10(M_cut))/sigma
    Phi = 0.5*(1 + erf(x/np.sqrt(2)))
    return Phi
    
def A_fun(p_max, Q, phi, Phi):
    """
    Aiding function for N_cen_ELG_v1().
    """
    print(2.*phi*Phi, np.max(2.*phi*Phi))
    A = (p_max-1./Q)/np.max(2.*phi*Phi)
    return A
    
def Gaussian_fun(x, mean, sigma):
    """
    Gaussian function with centered at `mean' with standard deviation `sigma'.
    """
    return norm.pdf(x, loc=mean, scale=sigma)

def main(tracer, want_plot=False):
    # HOD design for each tracer:
    #LRG: M_cut, kappa, sigma, M_1, alpha
    #ELG_v1: p_max, Q, M_cut, kappa, sigma, M_1, alpha, gamma
    #ELG_v2: p_max, M_cut, kappa, sigma, M_1, alpha, A_s
    #QSO: p_max, M_cut, kappa, sigma, M_1, alpha, A_s

    # Example values
    if tracer == 'ELG_v1':
        p_max = 0.33;
        Q = 100.;
        M_cut = 10.**11.75;
        kappa = 1.;
        sigma = 0.58;
        M_1 = 10.**13.53;
        alpha = 1.;
        gamma = 4.12;
        A_s = 1.
        
    elif tracer == 'ELG_v2':
        p_max = 0.00537;
        Q = 100.;
        M_cut = 10.**11.515;
        kappa = 1.;
        sigma = 0.08;
        M_1 = 10.**13.53;
        alpha = 1.;
        gamma = -1.4;
        A_s = 1.

    else:
        p_max = 0.33;
        Q = 100.;
        M_cut = 10.**12.75;
        kappa = 1.;
        sigma = 0.58;
        M_1 = 10.**13.53;
        alpha = 1.;
        gamma = 4.12;
        A_s = 1.
        
    HOD_design = {
        'p_max': p_max,
        'Q': Q,
        'M_cut': M_cut,
        'kappa': kappa,
        'sigma': sigma,
        'M_1': M_1,
        'alpha': alpha,
        'gamma': gamma,
        'A_s': A_s
    }

    # select range for computing the HOD function
    M_h = np.logspace(11, 15, 1000)

    # calculate the HOD function for the satellites and centrals
    
    if tracer == 'LRG':
        HOD_cen = N_cen_LRG(M_h, **HOD_design)
    elif tracer == 'ELG_v1':
        HOD_cen = N_cen_ELG_v1(M_h, **HOD_design)
    elif tracer == 'ELG_v2':
        HOD_cen = N_cen_ELG_v2(M_h, **HOD_design)
    elif tracer == 'QSO':
        HOD_cen = N_cen_QSO(M_h, **HOD_design)
    HOD_sat = N_sat(M_h, **HOD_design)

    if want_plot:
        plt.plot(M_h, HOD_sat, ls='--', color='k')
        plt.plot(M_h, HOD_cen, ls='-', color='k', label=tracer)
        plt.legend()
        plt.xlim([1.e11, 1.e15])
        plt.ylim([1.e-3, 1.e1])
        plt.xscale("log")
        plt.yscale("log")
        plt.show()

class ArgParseFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    pass

if __name__ == "__main__":
    # parsing arguments
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=ArgParseFormatter)
    parser.add_argument('--tracer', help='Select tracer type', choices=["LRG", "ELG_v1", "ELG_v2", "QSO"], default='ELG_v1')
    parser.add_argument('--want_plot', help='Plot HOD distribution?', action='store_true')
    args = vars(parser.parse_args())
    main(**args)
