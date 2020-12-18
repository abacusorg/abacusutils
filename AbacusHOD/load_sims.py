import os
from pathlib import Path

import numpy as np
import random
import time
from astropy.table import Table
import h5py
from sklearn.neighbors import KDTree

#from abacus_halo_catalog import AbacusHaloCatalog
from abacus_halo_catalog import CompaSOHaloCatalog

import multiprocessing
from multiprocessing import Pool

simname = "/AbacusSummit_base_c000_ph006"
savedir = "/mnt/gosling1/syuan/scratch/data_summit"+simname

newseed = 600
np.random.seed(newseed)

if not os.path.exists(savedir):
    os.makedirs(savedir)

# the subsampling curve for halos
def subsample_halos(m):
    x = np.log10(m)
    return 1.0/(1.0 + 0.1*np.exp(-(x - 13.3)*7))

def subsample_particles(m):
    x = np.log10(m)
    # return 1.0/(1.0 + np.exp(-(x - 13.5)*3))
    return 0.1/(4.0 + np.exp(-(x - 13.8)*5)) * 4 


def load_chunk(i):
    # # if file already exists, just skip
    # if os.path.exists(savedir+'/halos_xcom_'+str(i)+'_seed'+str(newseed)+'_abacushod.h5') \
    # and os.path.exists(savedir+'/particles_xcom_'+str(i)+'_seed'+str(newseed)+'_abacushod.h5'):
    #     return 0

    # load the halo catalog chunk
    print("loading catalog")
    cat = AbacusHaloCatalog(
        '/mnt/store2/bigsims/AbacusSummit'+simname+'/halos/z0.500/halo_info/halo_info_'\
        +str(i).zfill(3)+'.asdf', load_subsamples = 'A_halo_rv')
    halos = cat.halos
    parts = cat.subsamples
    header = cat.header
    Mpart = header['ParticleMassHMsun'] # msun / h 
    H0 = header['H0']
    h = H0/100.0

    # mass cut 1e12 msun / h
    halos = halos[halos['N']*Mpart / h > 2e12] # a lil less than 1mil per chunk

    # # form a halo table of the columns i care about 
    # halos_red = halos['N', 'id', 'npstartA', 'npoutA', 
    # 'x_com', 'v_com', 'sigmav3d_com', 'r100_com', 'SO_radius', 
    # 'x_L2com', 'v_L2com', 'sigmav3d_L2com', 'r100_L2com', 'SO_L2max_radius',
    # 'r10_com', 'r25_com', 'r50_com', 'r90_com', 'r98_com', 'rvcirc_max_com', 'rvcirc_max_L2com']# these rs got weird units] 

    # creating a mask of which halos to keep, which halos to drop
    p_halos = subsample_halos(halos['N']*Mpart / h)
    mask_halos = np.random.random(len(halos)) < p_halos

    halos['mask_subsample'] = mask_halos
    halos['multi_halos'] = 1.0 / p_halos

    # compute fenv around center of mass
    print("finding pairs")
    allpos = halos['x_com']
    allmasses = halos['N']*Mpart / h
    allpos_tree = KDTree(allpos)
    allinds_inner = allpos_tree.query_radius(allpos, r = halos['r98_com'])
    allinds_outer = allpos_tree.query_radius(allpos, r = 5)
    print("computng m stacks")
    starttime = time.time()
    Menv = np.array([np.sum(allmasses[allinds_outer[ind]]) - np.sum(allmasses[allinds_inner[ind]]) \
        for ind in np.arange(len(halos))])
    # Menv mean by mass bin 
    nbins = 100
    mbins = np.logspace(np.log10(2e12), 15.5, nbins + 1)
    fenv_binnorm = np.zeros(len(Menv))
    for ibin in range(nbins):
        mmask = (halos['N']*Mpart / h > mbins[ibin]) & (halos['N']*Mpart / h < mbins[ibin + 1])
        Menv_mean = np.mean(Menv[mmask])
        fenv_binnorm[mmask] = Menv[mmask] / Menv_mean

    fenv = Menv / np.mean(Menv)
    halos['fenv'] = fenv
    halos['fenv_binnorm'] = fenv_binnorm

    # compute delta concentration
    halos_c = halos['r90_com']/halos['r25_com']
    halos_deltac = np.zeros(len(halos))
    for ibin in range(nbins):
        mmask = (halos['N']*Mpart / h > mbins[ibin]) & (halos['N']*Mpart / h < mbins[ibin + 1])
        halos_deltac[mmask] = halos_c[mmask] - np.median(halos_c[mmask])
    halos['deltac'] = halos_deltac

    # the new particle start, len, and multiplier
    halos_pstart = halos['npstartA']
    halos_pnum = halos['npoutA']
    halos_pstart_new = np.zeros(len(halos))
    halos_pnum_new = np.zeros(len(halos))

    # particle arrays for ranks and mask 
    mask_parts = np.zeros(len(parts))
    len_old = len(parts)
    # ranks_parts = -np.ones(len(parts))
    # ranksv_parts = -np.ones(len(parts))
    # ranksr_parts = -np.ones(len(parts))
    # ranksp_parts = -np.ones(len(parts))
    pos_parts = -np.ones((len_old, 3))
    vel_parts = -np.ones((len_old, 3))
    Mh_parts = -np.ones(len_old)
    Np_parts = -np.ones(len_old)
    downsample_parts = -np.ones((len_old))
    idh_parts = -np.ones((len_old))

    start_tracker = 0
    for j in np.arange(len(halos)):
        if j % 1000 == 0:
            print(j)
        if mask_halos[j]:
            # updating the mask tagging the particles we want to preserve
            subsample_factor = subsample_particles(halos['N'][j]*Mpart / h)
            submask = np.random.binomial(n = 1, p = subsample_factor, size = halos_pnum[j])
            # updating the particles' masks, downsample factors, halo mass
            mask_parts[halos_pstart[j]: halos_pstart[j] + halos_pnum[j]] = submask
            downsample_parts[halos_pstart[j]: halos_pstart[j] + halos_pnum[j]] = p_halos[j]
            Mh_parts[halos_pstart[j]: halos_pstart[j] + halos_pnum[j]] = halos['N'][j]*Mpart # in msun / h
            Np_parts[halos_pstart[j]: halos_pstart[j] + halos_pnum[j]] = np.sum(submask)
            idh_parts[halos_pstart[j]: halos_pstart[j] + halos_pnum[j]] = halos['id'][j]

            # updating the pstart, pnum, for the halos
            halos_pstart_new[j] = start_tracker
            halos_pnum_new[j] = np.sum(submask)
            start_tracker += np.sum(submask)

            # make the rankings
            theseparts = parts[halos_pstart[j]: halos_pstart[j] + halos_pnum[j]][submask.astype(bool)]
            theseparts_pos = theseparts['pos']
            theseparts_vel = theseparts['vel']
            theseparts_halo_pos = halos['x_com'][j]
            theseparts_halo_vel = halos['v_com'][j]

            # indices_parts = np.arange(halos_pstart[j], halos_pstart[j] + halos_pnum[j])[submask.astype(bool)]
            # indices_parts = indices_parts.astype(int)
            # dist2_rel = np.sum((theseparts_pos - theseparts_halo_pos)**2, axis = 1)
            # ranks_parts[indices_parts] = dist2_rel.argsort().argsort() 

            # v2_rel = np.sum((theseparts_vel - theseparts_halo_vel)**2, axis = 1)
            # ranksv_parts[indices_parts] = v2_rel.argsort().argsort() 

            # # get rps
            # # calc relative positions
            # r_rel = theseparts_pos - theseparts_halo_pos 
            # r0 = np.sqrt(np.sum(r_rel**2, axis = 1))
            # r_rel_norm = r_rel/r0[:, None]

            # # list of peculiar velocities of the particles
            # vels_rel = theseparts_vel - theseparts_halo_vel # velocity km/s
            # # relative speed to halo center squared
            # v_rel2 = np.sum(vels_rel**2, axis = 1) 

            # # calculate radial and tangential peculiar velocity
            # vel_rad = np.sum(vels_rel*r_rel_norm, axis = 1)
            # ranksr_parts[indices_parts] = vel_rad.argsort().argsort() 

            # # radial component
            # v_rad2 = vel_rad**2 # speed
            # # tangential component
            # v_tan2 = v_rel2 - v_rad2

            # # compute the perihelion distance for NFW profile
            # m = halos['N'][j]*Mpart / h
            # rs = halos['r25_com'][j]
            # c = halos['r90_com'][j]/rs
            # r0_kpc = r0*1000 # kpc
            # alpha = 1.0/(np.log(1+c)-c/(1+c))*2*6.67e-11*m*2e30/r0_kpc/3.086e+19/1e6

            # # iterate a few times to solve for rp
            # x2 = v_tan2/(v_tan2+v_rad2)

            # num_iters = 20 # how many iterations do we want
            # factorA = v_tan2 + v_rad2
            # factorB = np.log(1+r0_kpc/rs)
            # for it in range(num_iters):
            #     oldx = np.sqrt(x2)
            #     x2 = v_tan2/(factorA + alpha*(np.log(1+oldx*r0_kpc/rs)/oldx - factorB))
            # x2[np.isnan(x2)] = 1
            # # final perihelion distance 
            # rp2 = r0_kpc**2*x2

            # ranksp_parts[indices_parts] = rp2.argsort().argsort() 

        else:
            halos_pstart_new[j] = -1
            halos_pnum_new[j] = -1

    halos['npstartA'] = halos_pstart_new
    halos['npoutA'] = halos_pnum_new

    # output halo file 
    print("outputting new halo file ")
    output_dir = savedir+'/halos_xcom_'+str(i)+'_seed'+str(newseed)+'_abacushod.h5'
    if os.path.exists(output_dir):
        os.remove(output_dir)
    newfile = h5py.File(output_dir, 'w')
    dataset = newfile.create_dataset('halos', data = halos)
    newfile.close()

    # output the new particle file
    print("adding rank fields to particle data ")
    parts = parts[mask_parts.astype(bool)]
    print("pre process particle number ", len_old, " post process particle number ", len(parts))
    # parts['ranks'] = ranks_parts[mask_parts.astype(bool)]
    # parts['ranksv'] = ranksv_parts[mask_parts.astype(bool)]
    # parts['ranksr'] = ranksr_parts[mask_parts.astype(bool)]
    # parts['ranksp'] = ranksp_parts[mask_parts.astype(bool)]
    parts['downsample_halo'] = downsample_parts[mask_parts.astype(bool)]
    parts['halo_mass'] = Mh_parts[mask_parts.astype(bool)]
    parts['Np'] = Np_parts[mask_parts.astype(bool)]
    parts['halo_id'] = idh_parts[mask_parts.astype(bool)]

    print("are there any negative particle values? ", np.sum(parts['downsample_halo'] < 0), 
        np.sum(parts['halo_mass'] < 0))
    print("outputting new particle file ")
    output_dir = savedir+'/particles_xcom_'+str(i)+'_seed'+str(newseed)+'_abacushod.h5'
    if os.path.exists(output_dir):
        os.remove(output_dir)
    newfile = h5py.File(output_dir, 'w')
    dataset = newfile.create_dataset('particles', data = parts)
    newfile.close()
    print("pre process particle number ", len_old, " post process particle number ", len(parts))

    return 0

# load_chunk(0)

# do further subsampling 
p = multiprocessing.Pool(7)
p.map(load_chunk, range(34))
p.close()
p.join()



print("done")

