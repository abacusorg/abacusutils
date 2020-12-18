# allsims testhod rsd
import numpy as np
import os,sys
import random
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as pl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import matplotlib.colors as colors
from matplotlib import rc, rcParams
rcParams.update({'font.size': 12})
from astropy.table import Table
import astropy.io.fits as pf
from astropy.cosmology import WMAP9 as cosmo
import scipy.spatial as spatial
import multiprocessing
from multiprocessing import Pool
from scipy import signal
from matplotlib import gridspec


from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0= 67.26 , Om0=0.316)

# start timer
startclock = time.time()

import Corrfunc
# from Corrfunc.mocks.DDrppi_mocks import DDrppi_mocks
# from Corrfunc.utils import convert_3d_counts_to_cf, convert_rp_pi_counts_to_wp
from Corrfunc.theory.DDrppi import DDrppi
from Corrfunc.theory import wp, xi

# set the colormap and centre the colorbar
class MidpointNormalize(colors.Normalize):
    """
    Normalise the colorbar so that diverging bars work there way either side from a prescribed midpoint value)

    e.g. im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))
    """
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))

np.random.seed(100)

# constants
params = {}
params['z'] = 0.5
params['h'] = 0.6736
params['Lboxh'] = 2000
params['Lbox'] = params['Lboxh']/params['h'] # Mpc, box size
params['Mpart'] = 3131059264.330557  # Msun, mass of each particle
params['velz2kms'] = 176311.8953571892/params['Lbox']
params['numchunks'] = 34

# rsd?
rsd = True
params['rsd'] = rsd
# central design 

# fiducial hod
newdesign = {'M_cut': 10**13.3, 
             'M1': 10**14.4, 
             'sigma': 0.8, 
             'alpha': 1.0,
             'kappa': 0.4}    
newdecor = {'s': 0, 
            's_v': 0, 
            'alpha_c': 0, 
            's_p': 0, 
            's_r': 0,
            'A': 0,
            'Ae': 0} 
decorator = "_fiducial"

rp_bins_course = np.logspace(-1, np.log10(30), 11)
pimax = 30.0 # h-1 mpc
pi_bin_size = 5
ximin = 0.02
ximax = 100

def calc_xi_mock_natural(design, decorations, rpbins, params, newseed, rsd = rsd):

    M_cut, M1, sigma, alpha, kappa = map(design.get, ('M_cut', 'M1', 'sigma', 'alpha', 'kappa'))
    s, s_v, alpha_c, s_p, s_r, A, Ae = map(decorations.get, ('s', 's_v', 'alpha_c', 's_p', 's_r', 'A', 'Ae'))

    start = time.time()
    # data directory
    scratchdir = "/data_mocks_summit_new"
    cdatadir = "/mnt/marvin1/boryanah/scratch" + scratchdir
    #cdatadir = "/mnt/marvin1/syuan/scratch" + scratchdir
    if rsd:
        cdatadir = cdatadir+"_rsd"
    savedir = cdatadir+"/rockstar_"\
    +str(np.log10(M_cut))[0:10]+"_"+str(np.log10(M1))[0:10]+"_"+str(sigma)[0:6]+"_"+str(alpha)[0:6]+"_"+str(kappa)[0:6]\
    +"_decor_"+str(s)+"_"+str(s_v)+"_"+str(alpha_c)+"_"+str(s_p)+"_"+str(s_r)+"_"+str(A)+"_"+str(Ae)
    if rsd:
        savedir = savedir+"_rsd"
    if not newseed == 0:
        savedir = savedir+"_"+str(newseed)

    pos_full = np.zeros((1, 3))
    for whichchunk in range(params['numchunks']):

        # these are the pre fc mocks
        filename_cent = savedir+"/halos_gal_cent_"+str(whichchunk)
        filename_sat = savedir+"/halos_gal_sats_"+str(whichchunk)

        # read in the galaxy catalog
        fcent = np.fromfile(filename_cent)
        fsats = np.fromfile(filename_sat)

        # reshape the file data
        fcent = np.array(np.reshape(fcent, (-1, 8)))
        fsats = np.array(np.reshape(fsats, (-1, 8)))

        pos_cent = fcent[:,0:3] * params['h'] % params['Lboxh'] # h-1 mpc
        pos_sats = fsats[:,0:3] * params['h'] % params['Lboxh']
        print(pos_cent, pos_sats)

        # full galaxy catalog
        # pos_full = np.concatenate((pos_cent, pos_sats)) * params['h'] % params['Lboxh']
        pos_full = np.concatenate((pos_full, pos_cent, pos_sats))

    pos_full = pos_full[1:]

    print("loading mocks took ", time.time() - start)
    ND = float(len(pos_full))
    print("calculating xi for galaxies ", ND)

    # convert to h-1 mpc
    start = time.time()
    print("pos_full check", np.shape(pos_full), np.max(pos_full), np.min(pos_full))

    DD_counts = DDrppi(1, 5, pimax, rpbins,
        pos_full[:, 0], pos_full[:, 1], pos_full[:, 2], boxsize = params['Lboxh'], periodic = True)['npairs']
    DD_counts_new = np.array([np.sum(DD_counts[i:i+pi_bin_size]) for i in range(0, len(DD_counts), pi_bin_size)])
    DD_counts_new = DD_counts_new.reshape((len(rpbins) - 1, int(pimax/pi_bin_size)))

    # now calculate the RR count from theory
    RR_counts_new = np.zeros((len(rpbins) - 1, int(pimax/pi_bin_size)))
    for i in range(len(rpbins) - 1):
        RR_counts_new[i] = np.pi*(rpbins[i+1]**2 - rpbins[i]**2)*pi_bin_size / params['Lboxh']**3 * ND**2 * 2
    xirppi_reshaped = DD_counts_new / RR_counts_new - 1

    print("xi done, time spent : ", time.time() - start)

    fname = savedir+"/data_xirppi_natural"
    np.savez(fname, rbins = rpbins, pimax = pimax, xi = xirppi_reshaped, DD = DD_counts, ND = ND)

    return xirppi_reshaped# 1d array

# def calc_xir(design, decorations, rbins, params, newseed, rsd = rsd):
#     M_cut, M1, sigma, alpha, kappa = map(design.get, ('M_cut', 'M1', 'sigma', 'alpha', 'kappa'))
#     s, s_v, alpha_c, s_p, s_r, A, Ae = map(decorations.get, ('s', 's_v', 'alpha_c', 's_p', 's_r', 'A', 'Ae'))

#     # data directory
#     scratchdir = "/data_mocks_summit_new"
#     cdatadir = "/mnt/gosling1/syuan/scratch" + scratchdir
#     if rsd:
#         cdatadir = cdatadir+"_rsd"
#     savedir = cdatadir+"/rockstar_"\
#     +str(np.log10(M_cut))[0:10]+"_"+str(np.log10(M1))[0:10]+"_"+str(sigma)[0:6]+"_"+str(alpha)[0:6]+"_"+str(kappa)[0:6]\
#     +"_decor_"+str(s)+"_"+str(s_v)+"_"+str(alpha_c)+"_"+str(s_p)+"_"+str(s_r)+"_"+str(A)+"_"+str(Ae)
#     if rsd:
#         savedir = savedir+"_rsd"
#     if not newseed == 0:
#         savedir = savedir+"_"+str(newseed)


#     pos_full = np.zeros((1, 3))
#     for whichchunk in range(params['numchunks']):

#         # these are the pre fc mocks
#         filename_cent = savedir+"/halos_gal_cent_"+str(whichchunk)
#         filename_sat = savedir+"/halos_gal_sats_"+str(whichchunk)

#         # read in the galaxy catalog
#         fcent = np.fromfile(filename_cent)
#         fsats = np.fromfile(filename_sat)

#         # reshape the file data
#         fcent = np.array(np.reshape(fcent, (-1, 8)))
#         fsats = np.array(np.reshape(fsats, (-1, 8)))

#         pos_cent = fcent[:,0:3] * params['h'] % params['Lboxh']
#         pos_sats = fsats[:,0:3] * params['h'] % params['Lboxh']
#         print(len(pos_cent), len(pos_sats))

#         # full galaxy catalog
#         # pos_full = np.concatenate((pos_cent, pos_sats)) * params['h'] % params['Lboxh']
#         pos_full = np.concatenate((pos_full, pos_cent, pos_sats))

#     pos_full = pos_full[1:]

#     results = xi(params['Lboxh'], 1, rbins, pos_full[:, 0], pos_full[:, 1], pos_full[:, 2])
#     newxir = np.array([row[3] for row in results])

#     results_cent = xi(params['Lboxh'], 1, rbins, pos_cent[:, 0], pos_cent[:, 1], pos_cent[:, 2])
#     newxir_cent = np.array([row[3] for row in results_cent])

#     results_sats = xi(params['Lboxh'], 1, rbins, pos_sats[:, 0], pos_sats[:, 1], pos_sats[:, 2])
#     newxir_sats = np.array([row[3] for row in results_sats])

#     fname = savedir+"/data_xir"
#     np.savez(fname, rbins = rbins, xir = newxir, xir_cent = newxir_cent, xir_sats = newxir_sats)


def calc_wp(design, decorations, rpbins, params, newseed, rsd = rsd):
    M_cut, M1, sigma, alpha, kappa = map(design.get, ('M_cut', 'M1', 'sigma', 'alpha', 'kappa'))
    s, s_v, alpha_c, s_p, s_r, A, Ae = map(decorations.get, ('s', 's_v', 'alpha_c', 's_p', 's_r', 'A', 'Ae'))

    # data directory
    scratchdir = "/data_mocks_summit_new"
    cdatadir = "/mnt/marvin1/boryanah/scratch" + scratchdir
    #cdatadir = "/mnt/marvin1/syuan/scratch" + scratchdir
    if rsd:
        cdatadir = cdatadir+"_rsd"
    savedir = cdatadir+"/rockstar_"\
    +str(np.log10(M_cut))[0:10]+"_"+str(np.log10(M1))[0:10]+"_"+str(sigma)[0:6]+"_"+str(alpha)[0:6]+"_"+str(kappa)[0:6]\
    +"_decor_"+str(s)+"_"+str(s_v)+"_"+str(alpha_c)+"_"+str(s_p)+"_"+str(s_r)+"_"+str(A)+"_"+str(Ae)
    if rsd:
        savedir = savedir+"_rsd"

    pos_full = np.zeros((1, 3))
    for whichchunk in range(params['numchunks']):

        # these are the pre fc mocks
        filename_cent = savedir+"/halos_gal_cent_"+str(whichchunk)
        filename_sat = savedir+"/halos_gal_sats_"+str(whichchunk)

        # read in the galaxy catalog
        fcent = np.fromfile(filename_cent)
        fsats = np.fromfile(filename_sat)

        # reshape the file data
        fcent = np.array(np.reshape(fcent, (-1, 8)))
        fsats = np.array(np.reshape(fsats, (-1, 8)))

        pos_cent = fcent[:,0:3] * params['h'] % params['Lboxh']
        pos_sats = fsats[:,0:3] * params['h'] % params['Lboxh']
        print(len(pos_cent), len(pos_sats))

        # full galaxy catalog
        # pos_full = np.concatenate((pos_cent, pos_sats)) * params['h'] % params['Lboxh']
        pos_full = np.concatenate((pos_full, pos_cent, pos_sats))

    pos_full = pos_full[1:]

    xs = pos_full[:,0]
    ys = pos_full[:,1]
    zs = pos_full[:,2]

    wp_results = wp(params['Lboxh'], pimax, 1, rpbins, xs, ys, zs, 
        verbose=False, output_rpavg=False) # this is all done in mpc / h
    newwp = np.array([row[3] for row in wp_results])

    fname = savedir+"/data_wp"
    np.savez(fname, rbins = rpbins, pimax = pimax, wp = newwp)

    return newwp

def plot_xi(design, decorations, rpbins, params, newseeds, rsd = rsd):

    M_cut, M1, sigma, alpha, kappa = map(newdesign.get, ('M_cut', 'M1', 'sigma', 'alpha', 'kappa'))
    s, s_v, alpha_c, s_p, s_r, A, Ae = map(newdecor.get, ('s', 's_v', 'alpha_c', 's_p', 's_r', 'A', 'Ae'))

    xi_sum = 0
    for eseed in newseeds:
        # data directory
        scratchdir = "/data_mocks_summit_new"
        cdatadir = "/mnt/marvin1/boryanah/scratch" + scratchdir
        #cdatadir = "/mnt/marvin1/syuan/scratch" + scratchdir
        if rsd:
            cdatadir = cdatadir+"_rsd"
        savedir = cdatadir+"/rockstar_"\
        +str(np.log10(M_cut))[0:10]+"_"+str(np.log10(M1))[0:10]+"_"+str(sigma)[0:6]+"_"+str(alpha)[0:6]+"_"+str(kappa)[0:6]\
        +"_decor_"+str(s)+"_"+str(s_v)+"_"+str(alpha_c)+"_"+str(s_p)+"_"+str(s_r)+"_"+str(A)+"_"+str(Ae)
        if rsd:
            savedir = savedir+"_rsd"
        if not eseed == 0:
            savedir = savedir+"_"+str(eseed)

        fname = savedir+"/data_xirppi_natural"
        newxi = np.load(fname+".npz")['xi']
        xi_sum += newxi

    xi_avg = xi_sum/len(newseeds)

    print(xi_avg)

    fig = pl.figure(figsize = (5, 4))
    pim = 30
    pl.imshow(xi_avg.T, interpolation = 'nearest', origin = 'lower', aspect = 'auto',
        extent = [np.log10(np.min(rpbins)), np.log10(np.max(rpbins)), 0, pim], 
        cmap = cm.viridis, norm=colors.LogNorm(vmin = ximin, vmax = ximax))
    cbar = pl.colorbar()
    cbar.set_label('$\\xi(r_\perp, \pi)$', rotation = 270, labelpad = 20)
    # pl.xscale('log')
    pl.xlabel('$\log r_\perp$ ($h^{-1}$Mpc)')
    pl.ylabel('$\pi$ ($h^{-1}$Mpc)')
    pl.tight_layout()
    plotname = "./plots/plot_xirppi_mock_reseeded"+decorator
    fig.savefig(plotname+".png", dpi = 300)

def plot_wp(design, decorations, rps, params, newseeds, rsd = rsd):

    M_cut, M1, sigma, alpha, kappa = map(newdesign.get, ('M_cut', 'M1', 'sigma', 'alpha', 'kappa'))
    s, s_v, alpha_c, s_p, s_r, A, Ae = map(newdecor.get, ('s', 's_v', 'alpha_c', 's_p', 's_r', 'A', 'Ae'))

    wp_sum = 0
    for eseed in newseeds:
        # data directory
        scratchdir = "/data_mocks_summit_new"
        cdatadir = "/mnt/gosling1/syuan/scratch" + scratchdir
        if rsd:
            cdatadir = cdatadir+"_rsd"
        savedir = cdatadir+"/rockstar_"\
        +str(np.log10(M_cut))[0:10]+"_"+str(np.log10(M1))[0:10]+"_"+str(sigma)[0:6]+"_"+str(alpha)[0:6]+"_"+str(kappa)[0:6]\
        +"_decor_"+str(s)+"_"+str(s_v)+"_"+str(alpha_c)+"_"+str(s_p)+"_"+str(s_r)+"_"+str(A)+"_"+str(Ae)
        if rsd:
            savedir = savedir+"_rsd"

        fname = savedir+"/data_wp"
        newwp = np.load(fname+".npz")['wp']
        wp_sum += newwp

    wp_avg = wp_sum/len(newseeds)

    # hong's data
    hong_wp_data = np.loadtxt("./hong_data/wp_cmass_z0.46-0.6")
    rp_hong = hong_wp_data[:, 0] # h-1 mpc
    wp_hong = hong_wp_data[:, 1]
    rwp_hong = rp_hong * wp_hong  # (h-1 mpc)^2

    # covariance
    hong_wp_covmat = np.loadtxt("./hong_data/wpcov_cmass_z0.46-0.6")
    hong_rwp_covmat = np.zeros(np.shape(hong_wp_covmat))
    for i in range(np.shape(hong_wp_covmat)[0]):
        for j in range(np.shape(hong_wp_covmat)[1]):
            hong_rwp_covmat[i, j] = hong_wp_covmat[i, j]*rp_hong[i]*rp_hong[j]
    hong_rwp_covmat_inv = np.linalg.inv(hong_rwp_covmat)
    hong_rwp_covmat_inv_short = np.linalg.inv(hong_rwp_covmat)[2:, 2:]
    rwp_hong_err = 1/np.sqrt(np.diag(hong_rwp_covmat_inv))

    fig = pl.figure(figsize=(8.5, 4))
    gs = gridspec.GridSpec(1, 2, width_ratios = [1, 1]) 

    # load the rockstar results from abacuscosmos
    # rsdata = np.load("../s3PCF/data/data_wp_reseeded_compare_vanilla.npz")
    ax1 = fig.add_subplot(gs[0])
    ax1.set_xlabel('$r_{\perp}$ ($h^{-1} \mathrm{Mpc}$)')
    ax1.set_ylabel('$r_{\perp} w_p$ ($h^{-1} \mathrm{Mpc})^2$')
    ax1.errorbar(rp_hong[2:], rwp_hong[2:], yerr = rwp_hong_err[2:], label = 'observed')
    ax1.plot(rps[2:], wp_avg[2:]*rps[2:], label = 'Summit')
    # ax1.plot(rsdata['rp'], rsdata['wp']*rsdata['rp'], label = 'Rockstar')
    ax1.set_xscale('log')
    ax1.set_xlim(0.1, 50)
    ax1.legend(loc='best', prop={'size': 13})

    delta_rwp = (wp_avg * rps - rwp_hong)[2:]
    chi2s = delta_rwp * np.dot(hong_rwp_covmat_inv_short, delta_rwp)

    ax2 = fig.add_subplot(gs[1])
    ax2.set_xlabel('$r_{\perp}$ ($h^{-1} \mathrm{Mpc}$)')
    ax2.set_ylabel('$X^2$')
    ax2.set_xscale('log')
    ax2.plot(rps[2:], chi2s, 'r-o', label = "$X^2 = $"+str(np.sum(chi2s[2:]))[:6])
    ax2.set_xlim(0.1, 50)
    ax2.legend(loc='best', prop={'size': 13})

    pl.tight_layout()
    fig.savefig("./plots/plot_wp_reseeded"+decorator+".png", dpi=720)
    np.savez("./data/data_wp_reseeded"+decorator, wp = wp_avg, rp = rps)

def compare_to_boss(design, decorations, rpbins, params, newseeds, rsd = rsd):

    # load mock
    M_cut, M1, sigma, alpha, kappa = map(newdesign.get, ('M_cut', 'M1', 'sigma', 'alpha', 'kappa'))
    s, s_v, alpha_c, s_p, s_r, A, Ae = map(newdecor.get, ('s', 's_v', 'alpha_c', 's_p', 's_r', 'A', 'Ae'))

    xi_sum = 0
    for eseed in newseeds:
        # data directory
        scratchdir = "/data_mocks_summit_new"
        cdatadir = "/mnt/gosling1/syuan/scratch" + scratchdir
        if rsd:
            cdatadir = cdatadir+"_rsd"
        savedir = cdatadir+"/rockstar_"\
        +str(np.log10(M_cut))[0:10]+"_"+str(np.log10(M1))[0:10]+"_"+str(sigma)[0:6]+"_"+str(alpha)[0:6]+"_"+str(kappa)[0:6]\
        +"_decor_"+str(s)+"_"+str(s_v)+"_"+str(alpha_c)+"_"+str(s_p)+"_"+str(s_r)+"_"+str(A)+"_"+str(Ae)
        if rsd:
            savedir = savedir+"_rsd"
        if not eseed == 0:
            savedir = savedir+"_"+str(eseed)

        fname = savedir+"/data_xirppi_natural"
        newxi = np.load(fname+".npz")['xi']
        xi_sum += newxi

    xi_avg = xi_sum/len(newseeds)
    # covariance matrix 
    xicov = np.load("./data/data_xi_cov.npz")['xicov']
    xicov_inv = np.linalg.inv(xicov)
    xi_errs = np.sqrt(1/np.diag(np.linalg.inv(xicov))).reshape(np.shape(xi_avg))

    # load boss
    xi_boss = np.loadtxt("./hong_data/newbins/xip_cmass_z0.46-0.6")
    # print(np.shape(xi_avg), np.shape(xi_boss))
    delta_xi = xi_avg - xi_boss
    delta_xi_norm = (xi_avg - xi_boss)/xi_errs
    delta_xi[0] = 0
    delta_xi_norm[0] = 0

    zmin2 = -10 # np.min(delta_xi) # -3
    zmax2 = 10 # np.max(delta_xi) # 9.5
    mycmap2 = cm.get_cmap('bwr')

    # (xi_mock - xi_boss) / err
    fig = pl.figure(figsize=(5, 4))
    pim = 30
    pl.imshow(delta_xi_norm.T, interpolation = 'nearest', origin = 'lower', aspect = 'auto',
        extent = [np.log10(np.min(rpbins)), np.log10(np.max(rpbins)), 0, pim], 
        cmap = mycmap2, norm=MidpointNormalize(midpoint=0,vmin=zmin2, vmax=zmax2))
    cbar = pl.colorbar()
    cbar.set_label('$(\\xi_{\\rm{mock}}-\\xi_{\\rm{BOSS}})/\sigma(\\xi)$', rotation = 270, labelpad = 20)
    # pl.xscale('log')
    pl.xlabel('$\log r_\perp$ ($h^{-1}$Mpc)')
    pl.ylabel('$\pi$ ($h^{-1}$Mpc)')
    pl.tight_layout()
    plotname = "./plots/plot_delta_xirppi_mock_boss"+decorator
    fig.savefig(plotname+".png", dpi = 300)

    # (xi_mock - xi_boss) / xi_mock
    fig = pl.figure(figsize=(5, 4))
    pim = 30
    delta_xi_byxi = delta_xi / xi_avg
    print(np.min(delta_xi_byxi), np.max(delta_xi_byxi))
    pl.imshow(delta_xi_byxi.T, interpolation = 'nearest', origin = 'lower', aspect = 'auto',
        extent = [np.log10(np.min(rpbins)), np.log10(np.max(rpbins)), 0, pim], 
        cmap = mycmap2, norm=MidpointNormalize(midpoint=0,vmin=-0.25, vmax=0.25))
    cbar = pl.colorbar()
    cbar.set_label('$(\\xi_{\\rm{mock}}-\\xi_{\\rm{BOSS}})/\\xi_{\\rm{mock}}$', rotation = 270, labelpad = 20)
    pl.xlabel('$\log r_\perp$ ($h^{-1}$Mpc)')
    pl.ylabel('$\pi$ ($h^{-1}$Mpc)')
    pl.tight_layout()
    plotname = "./plots/plot_delta_xirppi_mock_boss_byxi"+decorator
    fig.savefig(plotname+".png", dpi = 300)

    # make a triple plot, xi, delta xi, chi2
    fig = pl.figure(figsize=(13, 5))
    gs = gridspec.GridSpec(ncols = 3, nrows = 2, width_ratios = [1, 1, 1], height_ratios = [1, 12]) 

    # plot 1
    ax1 = fig.add_subplot(gs[3])
    ax1.set_xlabel('$\log r_\perp$ ($h^{-1} \mathrm{Mpc}$)')
    ax1.set_ylabel('$\pi$ ($h^{-1} \mathrm{Mpc}$)')
    # ax1.set_xscale('log')
    col1 = ax1.imshow(xi_avg.T, interpolation = 'nearest', origin = 'lower', aspect = 'auto',
        extent = [np.log10(np.min(rpbins)), np.log10(np.max(rpbins)), 0, pim], 
        cmap = cm.viridis, norm=colors.LogNorm(vmin = ximin, vmax = ximax))

    ax0 = fig.add_subplot(gs[0])
    cbar = pl.colorbar(col1, cax = ax0, orientation="horizontal")
    cbar.set_label('$\\xi(r_\perp, \pi)$', labelpad = 10)
    cbar.ax.xaxis.set_label_position('top')

    # plot 2
    ax2 = fig.add_subplot(gs[4])
    ax2.set_xlabel('$\log r_\perp$ ($h^{-1} \mathrm{Mpc}$)')
    ax2.set_ylabel('$\pi$ ($h^{-1} \mathrm{Mpc}$)')
    # ax2.set_xscale('log')
    col2 = ax2.imshow(delta_xi_norm.T, interpolation = 'nearest', origin = 'lower', aspect = 'auto',
        extent = [np.log10(np.min(rpbins)), np.log10(np.max(rpbins)), 0, pim], 
        cmap = mycmap2, norm=MidpointNormalize(midpoint=0,vmin=zmin2, vmax=zmax2))

    ax3 = fig.add_subplot(gs[1])
    cbar = pl.colorbar(col2, cax = ax3, orientation="horizontal")
    cbar.set_label("$(\\xi_{\\rm{mock}}-\\xi_{\\rm{BOSS}})/\sigma(\\xi)$", labelpad = 10)
    cbar.ax.xaxis.set_label_position('top')
    # cbar.set_ticks(np.linspace(-1, 1, num = 5))

    # plot 3
    chi2s = (delta_xi.flatten() * np.dot(xicov_inv, delta_xi.flatten())).reshape(np.shape(delta_xi))
    ax2 = fig.add_subplot(gs[5])
    ax2.set_xlabel('$\log r_\perp$ ($h^{-1} \mathrm{Mpc}$)')
    ax2.set_ylabel('$\pi$ ($h^{-1} \mathrm{Mpc}$)')
    col2 = ax2.imshow(chi2s.T, interpolation = 'nearest', origin = 'lower', aspect = 'auto',
        extent = [np.log10(np.min(rpbins)), np.log10(np.max(rpbins)), 0, pim], 
        cmap = mycmap2, norm=MidpointNormalize(midpoint=0,vmin=-100, vmax=100))

    ax3 = fig.add_subplot(gs[2])
    cbar = pl.colorbar(col2, cax = ax3, orientation="horizontal")
    cbar.set_label("$X^2$", labelpad = 10)
    cbar.ax.xaxis.set_label_position('top')
    # cbar.set_ticks(np.linspace(-1, 1, num = 5))

    pl.subplots_adjust(wspace=20)
    pl.tight_layout()
    fig.savefig("./plots/plot_xi_mock_diff_2plot"+decorator+".png", dpi=720)


if __name__ == "__main__":

    newseeds = [0]
    for eseed in newseeds:
        # calc_wp(newdesign, newdecor, rp_bins, params, eseed, rsd = rsd)
        calc_xi_mock_natural(newdesign, newdecor, rp_bins_course, params, eseed, rsd = rsd)
        # calc_xir(newdesign, newdecor, rs_bins_ext, params, eseed, rsd = rsd)

    plot_xi(newdesign, newdecor, rp_bins_course, params, newseeds, rsd = rsd)
    # plot_xir(newdesign, newdecor, rs_mid_ext, params, newseeds, rsd = rsd)
    # plot_wp(newdesign, newdecor, rp_saito, params, newseeds, rsd = rsd)
    # compare_to_boss(newdesign, newdecor, rp_bins, params, newseeds, rsd = rsd)

