# allsims testhod rsd
import numpy as np
import copy
import os,sys
import random
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as pl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from matplotlib import rc, rcParams
rcParams.update({'font.size': 20})
from astropy.table import Table
import astropy.io.fits as pf
from astropy.cosmology import WMAP9 as cosmo
import scipy.spatial as spatial
import multiprocessing
from multiprocessing import Pool
from scipy import signal
import h5py

startclock = time.time()

from GRAND_HOD import gen_gal_catalog_rockstar_modified_subsampled as galcat

# constants
params = {}
params['z'] = 0.5
params['h'] = 0.6736
params['Lbox'] = 2000/params['h'] # Mpc, box size
params['Mpart'] = 3131059264.330557  # Msun, mass of each particle
params['velz2kms'] = 176311.8953571892/params['Lbox']
params['numchunks'] = 34
params['rsd'] = True

# run the avg_c for assembly bias
scratchdir = "/data_mocks_summit_new"
cdatadir = "/mnt/marvin1/syuan/scratch" + scratchdir
mydatadir = "/mnt/marvin1/syuan/scratch/data_summit/AbacusSummit_base_c000_ph006"

halo_data = []
particle_data = []
# load all the halo and particle data we need
for echunk in range(params['numchunks']):

    newfile = h5py.File(mydatadir+'/halos_xcom_'+str(echunk)+'_seed600_abacushod.h5', 'r')
    maskedhalos = newfile['halos']

    # extracting the halo properties that we need
    halo_ids = maskedhalos["id"] # halo IDs
    halo_pos = maskedhalos["x_com"]/params['h'] # halo positions, Mpc
    halo_vels = maskedhalos['v_com'] # halo velocities, km/s
    halo_vrms = maskedhalos["sigmav3d_com"] # halo velocity dispersions, km/s
    halo_mass = maskedhalos['N']*params['Mpart'] # halo mass, Msun, 200b
    halo_deltac = maskedhalos['deltac'] # halo concentration
    halo_fenv = maskedhalos['fenv_binnorm'] # halo velocities, km/s
    halo_pstart = maskedhalos['npstartA'] # starting index of particles
    halo_pnum = maskedhalos['npoutA'] # number of particles 
    halo_multi = maskedhalos['multi_halos']
    halo_submask = maskedhalos['mask_subsample']
    halo_data_chunk = np.concatenate((halo_ids[:, None], halo_pos, halo_vels, halo_vrms[:, None], 
        halo_mass[:, None], halo_deltac[:, None], halo_fenv[:, None], halo_pstart[:, None], halo_pnum[:, None], 
        halo_multi[:, None], halo_submask[:, None]), axis = 1)
    halo_data += [halo_data_chunk]

    # extract particle data that we need
    newpart = h5py.File(mydatadir+'/particles_xcom_'+str(echunk)+'_seed600_abacushod.h5', 'r')
    subsample = newpart['particles']
    part_pos = subsample['pos'] / params['h']
    part_vel = subsample['vel']
    part_halomass = subsample['halo_mass'] / params['h'] # msun
    part_haloid = subsample['halo_id']
    part_Np = subsample['Np'] # number of particles that end up in the halo
    part_subsample = subsample['downsample_halo']
    # part_ranks = subsample['ranks']
    # part_ranksv = subsample['ranksv']
    # part_ranksp = subsample['ranksp']
    # part_ranksr = subsample['ranksr']
    # part_data_chunk = np.concatenate((part_pos, part_vel, part_ranks[:, None], part_ranksv[:, None], 
    #     part_ranksp[:, None], part_ranksr[:, None]), axis = 1)
    part_data_chunk = np.concatenate((part_pos, part_vel, 
        part_halomass[:, None], part_haloid[:, None], part_Np[:, None], part_subsample[:, None]), axis = 1)
    particle_data += [part_data_chunk]

# method for running one hod on one sim with one reseeding
def gen_gal_onesim_onehod(whichchunk, design, decorations, savedir, eseed = 0, params = params):

    galcat.gen_gal_cat(whichchunk, halo_data[whichchunk], particle_data[whichchunk], design, decorations, params, 
        savedir, whatseed = eseed, rsd = params['rsd']) 

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
newdecor['ic'] = 0.97


newseed = 0

# only do this if this has not been done before 
M_cutn, M1n, sigman, alphan, kappan = map(newdesign.get, ('M_cut', 'M1', 'sigma', 'alpha', 'kappa'))
sn, s_vn, alpha_cn, s_pn, s_rn, An, Aen = map(newdecor.get, ('s', 's_v', 'alpha_c', 's_p', 's_r', 'A', 'Ae'))

datadir = cdatadir
if params['rsd']:
    datadir = datadir+"_rsd"
savedir = datadir+"/rockstar_"\
+str(np.log10(M_cutn))[0:10]+"_"+str(np.log10(M1n))[0:10]+"_"+str(sigman)[0:6]+"_"+str(alphan)[0:6]+"_"+str(kappan)[0:6]\
+"_decor_"+str(sn)+"_"+str(s_vn)+"_"+str(alpha_cn)+"_"+str(s_pn)+"_"+str(s_rn)+"_"+str(An)+"_"+str(Aen)
if params['rsd']:
    savedir = savedir+"_rsd"
if newseed == 0:
    savedir_seeded = savedir
else:
    savedir_seeded = savedir + "_" + str(newseed)

if not os.path.exists(savedir_seeded):
    os.makedirs(savedir_seeded)

def run_onebox(i):
    gen_gal_onesim_onehod(i, newdesign, newdecor, savedir_seeded, eseed = newseed, params = params)
    return 0


if __name__ == "__main__":

    start = time.time()
    # run_onebox(0)

    # multiprocess
    p = multiprocessing.Pool(17)
    p.map(run_onebox, range(params['numchunks']))
    
    print("Done ", time.time() - start)

