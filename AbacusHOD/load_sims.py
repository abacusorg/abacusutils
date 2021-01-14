import os
from pathlib import Path

import numpy as np
import random
import time
from astropy.table import Table
import h5py
from sklearn.neighbors import KDTree

from abacus_halo_catalog import AbacusHaloCatalog
# from abacus_halo_catalog import CompaSOHaloCatalog

import multiprocessing
from multiprocessing import Pool

simname = "/AbacusSummit_base_c000_ph006"
z_mock = 0.800
savedir = "/mnt/marvin1/syuan/scratch/data_summit"+simname+"/z"+str(z_mock).ljust(5, '0')

newseed = 600

if not os.path.exists(savedir):
    os.makedirs(savedir)

# https://arxiv.org/pdf/2001.06018.pdf Figure 13 shows redshift evolution of LRG HOD 
# the subsampling curve for halos
def subsample_halos(m):
    x = np.log10(m)
    # return 1.0/(1.0 + 0.1*np.exp(-(x - 13.3)*5)) # LRG only
    return 1.0/(1.0 + 10*np.exp(-(x - 11.2)*25)) # MT


def subsample_particles(m):
    x = np.log10(m)
    # return 4/(200.0 + np.exp(-(x - 13.7)*8)) # LRG only
    return 4/(200.0 + np.exp(-(x - 13.2)*6)) # MT



def load_chunk(i):
    np.random.seed(newseed + i)
    # # if file already exists, just skip
    # if os.path.exists(savedir+'/halos_xcom_'+str(i)+'_seed'+str(newseed)+'_abacushod.h5') \
    # and os.path.exists(savedir+'/particles_xcom_'+str(i)+'_seed'+str(newseed)+'_abacushod.h5'):
    #     return 0

    # load the halo catalog chunk
    print("loading catalog")
    cat = AbacusHaloCatalog(
        '/mnt/store2/bigsims/AbacusSummit'+simname+'/halos/z'+str(z_mock).ljust(5, '0')+'/halo_info/halo_info_'\
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
    fenv_rank = np.zeros(len(Menv))
    for ibin in range(nbins):
        mmask = (halos['N']*Mpart / h > mbins[ibin]) & (halos['N']*Mpart / h < mbins[ibin + 1])
        if np.sum(mmask) > 0:
            if np.sum(mmask) == 1:
                fenv_rank[mmask] = 0
            else:
                Menv_mean = np.mean(Menv[mmask])
                newfenv = Menv[mmask] / Menv_mean
                new_fenv_rank = newfenv.argsort().argsort()
                fenv_rank[mmask] = new_fenv_rank / np.max(new_fenv_rank) - 0.5

    fenv = Menv / np.mean(Menv)
    halos['fenv'] = fenv
    halos['fenv_rank'] = fenv_rank

    # compute delta concentration
    halos_c = halos['r90_com']/halos['r25_com']
    deltac_rank = np.zeros(len(halos))
    for ibin in range(nbins):
        mmask = (halos['N']*Mpart / h > mbins[ibin]) & (halos['N']*Mpart / h < mbins[ibin + 1])
        if np.sum(mmask) > 0:
            if np.sum(mmask) == 1:
                deltac_rank[mmask] = 0
            else:
                new_deltac = halos_c[mmask] - np.median(halos_c[mmask])
                new_deltac_rank = new_deltac.argsort().argsort()
                deltac_rank[mmask] = new_deltac_rank / np.max(new_deltac_rank) - 0.5
    halos['deltac_rank'] = deltac_rank
    print(np.sum(np.isnan(deltac_rank)), np.max(deltac_rank), np.min(deltac_rank))
    print(np.sum(np.isnan(fenv_rank)), np.max(fenv_rank), np.min(fenv_rank))

    # the new particle start, len, and multiplier
    halos_pstart = halos['npstartA']
    halos_pnum = halos['npoutA']
    halos_pstart_new = np.zeros(len(halos))
    halos_pnum_new = np.zeros(len(halos))

    # particle arrays for ranks and mask 
    mask_parts = np.zeros(len(parts))
    len_old = len(parts)
    ranks_parts = -np.ones(len(parts))
    ranksv_parts = -np.ones(len(parts))
    ranksr_parts = -np.ones(len(parts))
    ranksp_parts = -np.ones(len(parts))
    pos_parts = -np.ones((len_old, 3))
    vel_parts = -np.ones((len_old, 3))
    hvel_parts = -np.ones((len_old, 3))
    Mh_parts = -np.ones(len_old)
    Np_parts = -np.ones(len_old)
    downsample_parts = -np.ones((len_old))
    idh_parts = -np.ones((len_old))
    deltach_parts = -np.ones((len_old))
    fenvh_parts = -np.ones((len_old))

    start_tracker = 0
    for j in np.arange(len(halos)):
        if j % 20000 == 0:
            print(j)
        if mask_halos[j]:
            # updating the mask tagging the particles we want to preserve
            subsample_factor = subsample_particles(halos['N'][j] * Mpart / h)
            submask = np.random.binomial(n = 1, p = subsample_factor, size = halos_pnum[j])
            # updating the particles' masks, downsample factors, halo mass
            mask_parts[halos_pstart[j]: halos_pstart[j] + halos_pnum[j]] = submask
            # print(j, halos_pstart, halos_pnum, p_halos, downsample_parts)
            downsample_parts[halos_pstart[j]: halos_pstart[j] + halos_pnum[j]] = p_halos[j]
            hvel_parts[halos_pstart[j]: halos_pstart[j] + halos_pnum[j]] = halos['v_com'][j]
            Mh_parts[halos_pstart[j]: halos_pstart[j] + halos_pnum[j]] = halos['N'][j] * Mpart # in msun / h
            Np_parts[halos_pstart[j]: halos_pstart[j] + halos_pnum[j]] = np.sum(submask)
            idh_parts[halos_pstart[j]: halos_pstart[j] + halos_pnum[j]] = halos['id'][j]
            deltach_parts[halos_pstart[j]: halos_pstart[j] + halos_pnum[j]] = deltac_rank[j]
            fenvh_parts[halos_pstart[j]: halos_pstart[j] + halos_pnum[j]] = fenv_rank[j]

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
            indices_parts = np.arange(halos_pstart[j], halos_pstart[j] + halos_pnum[j])[submask.astype(bool)]
            indices_parts = indices_parts.astype(int)
            if np.sum(submask) == 1:
                ranks_parts[indices_parts] = 0
                ranksv_parts[indices_parts] = 0
                ranksp_parts[indices_parts] = 0
                ranksr_parts[indices_parts] = 0

            else:
                dist2_rel = np.sum((theseparts_pos - theseparts_halo_pos)**2, axis = 1)
                newranks = dist2_rel.argsort().argsort() 
                ranks_parts[indices_parts] = (newranks - np.mean(newranks)) / np.mean(newranks)

                v2_rel = np.sum((theseparts_vel - theseparts_halo_vel)**2, axis = 1)
                newranksv = v2_rel.argsort().argsort() 
                ranksv_parts[indices_parts] = (newranksv - np.mean(newranksv)) / np.mean(newranksv)

                # get rps
                # calc relative positions
                r_rel = theseparts_pos - theseparts_halo_pos 
                r0 = np.sqrt(np.sum(r_rel**2, axis = 1))
                r_rel_norm = r_rel/r0[:, None]

                # list of peculiar velocities of the particles
                vels_rel = theseparts_vel - theseparts_halo_vel # velocity km/s
                # relative speed to halo center squared
                v_rel2 = np.sum(vels_rel**2, axis = 1) 

                # calculate radial and tangential peculiar velocity
                vel_rad = np.sum(vels_rel*r_rel_norm, axis = 1)
                newranksr = vel_rad.argsort().argsort() 
                ranksr_parts[indices_parts] = (newranksr - np.mean(newranksr)) / np.mean(newranksr)

                # radial component
                v_rad2 = vel_rad**2 # speed
                # tangential component
                v_tan2 = v_rel2 - v_rad2

                # compute the perihelion distance for NFW profile
                m = halos['N'][j]*Mpart / h
                rs = halos['r25_com'][j]
                c = halos['r90_com'][j]/rs
                r0_kpc = r0*1000 # kpc
                alpha = 1.0/(np.log(1+c)-c/(1+c))*2*6.67e-11*m*2e30/r0_kpc/3.086e+19/1e6

                # iterate a few times to solve for rp
                x2 = v_tan2/(v_tan2+v_rad2)

                num_iters = 20 # how many iterations do we want
                factorA = v_tan2 + v_rad2
                factorB = np.log(1+r0_kpc/rs)
                for it in range(num_iters):
                    oldx = np.sqrt(x2)
                    x2 = v_tan2/(factorA + alpha*(np.log(1+oldx*r0_kpc/rs)/oldx - factorB))
                x2[np.isnan(x2)] = 1
                # final perihelion distance 
                rp2 = r0_kpc**2*x2
                newranksp = rp2.argsort().argsort() 
                ranksp_parts[indices_parts] = (newranksp - np.mean(newranksp)) / np.mean(newranksp)

        else:
            halos_pstart_new[j] = -1
            halos_pnum_new[j] = -1

    halos['npstartA'] = halos_pstart_new
    halos['npoutA'] = halos_pnum_new
    halos['randoms'] = np.random.random(len(halos)) # attaching random numbers
    halos['randoms_gaus_vrms'] = np.random.normal(loc = 0, scale = halos["sigmav3d_com"]/np.sqrt(3), size = len(halos)) # attaching random numbers

    # output halo file 
    print("outputting new halo file ")
    output_dir = savedir+'/halos_xcom_'+str(i)+'_seed'+str(newseed)+'_abacushodMT.h5'
    if os.path.exists(output_dir):
        os.remove(output_dir)
    newfile = h5py.File(output_dir, 'w')
    dataset = newfile.create_dataset('halos', data = halos)
    newfile.close()

    # output the new particle file
    print("adding rank fields to particle data ")
    parts = parts[mask_parts.astype(bool)]
    print("pre process particle number ", len_old, " post process particle number ", len(parts))
    parts['ranks'] = ranks_parts[mask_parts.astype(bool)]
    parts['ranksv'] = ranksv_parts[mask_parts.astype(bool)]
    parts['ranksr'] = ranksr_parts[mask_parts.astype(bool)]
    parts['ranksp'] = ranksp_parts[mask_parts.astype(bool)]
    parts['downsample_halo'] = downsample_parts[mask_parts.astype(bool)]
    parts['halo_vel'] = hvel_parts[mask_parts.astype(bool)]
    parts['halo_mass'] = Mh_parts[mask_parts.astype(bool)]
    parts['Np'] = Np_parts[mask_parts.astype(bool)]
    parts['halo_id'] = idh_parts[mask_parts.astype(bool)]
    parts['randoms'] = np.random.random(len(parts))
    parts['halo_deltac'] = deltach_parts[mask_parts.astype(bool)]
    parts['halo_fenv'] = fenvh_parts[mask_parts.astype(bool)]

    print("are there any negative particle values? ", np.sum(parts['downsample_halo'] < 0), 
        np.sum(parts['halo_mass'] < 0))
    print("outputting new particle file ")
    output_dir = savedir+'/particles_xcom_'+str(i)+'_seed'+str(newseed)+'_abacushodMT.h5'
    if os.path.exists(output_dir):
        os.remove(output_dir)
    newfile = h5py.File(output_dir, 'w')
    dataset = newfile.create_dataset('particles', data = parts)
    newfile.close()
    print("pre process particle number ", len_old, " post process particle number ", len(parts))

    return 0

# load_chunk(0)

# do further subsampling 
p = multiprocessing.Pool(5)
p.map(load_chunk, range(34))
p.close()
p.join()



print("done")
