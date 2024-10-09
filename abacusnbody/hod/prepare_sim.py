"""
This is a script for loading simulation data and generating subsamples.

Usage
-----
$ python -m abacusnbody.hod.AbacusHOD.prepare_sim --path2config /path/to/config.yaml
"""

import argparse
import concurrent.futures
import gc
import glob
import itertools
import multiprocessing
import os
import time
from pathlib import Path

import h5py
import numba
import numpy as np
import yaml
from numba import njit
from scipy.interpolate import interpn
from scipy.spatial import cKDTree

from abacusnbody.data.compaso_halo_catalog import CompaSOHaloCatalog
from abacusnbody.data.read_abacus import read_asdf

from ..analysis.shear import get_shear, smooth_density
from ..analysis.tsc import tsc_parallel

DEFAULTS = {}
DEFAULTS['path2config'] = 'config/abacus_hod.yaml'


# https://arxiv.org/pdf/2001.06018.pdf Figure 13 shows redshift evolution of LRG HOD
# standard power law satellites
def subsample_halos(m, MT):
    x = np.log10(m)
    downfactors = np.zeros(len(x))
    if MT:
        # for elgs
        mask1 = x < 11.4
        mask2 = x < 11.6
        downfactors[mask1] = 0.2 / (1.0 + 10 * np.exp(-(x[mask1] - 11.2) * 25))
        downfactors[mask2 & (~mask1)] = 0.4 / (
            1.0 + 10 * np.exp(-(x[mask2 & (~mask1)] - 11.3) * 25)
        )
        downfactors[~mask2] = 1.0 / (1.0 + 0.1 * np.exp(-(x[~mask2] - 11.7) * 10))

        # # for bgs
        # mask1 = x < 11.0
        # mask2 = x < 11.2
        # downfactors[mask2&(~mask1)] = 0.1 # 0.4/(1.0 + 10*np.exp(-(x[mask2&(~mask1)] - 10.9)*25))
        # downfactors[~mask2] = 1 # 1.0/(1.0 + 0.1*np.exp(-(x[~mask2] - 11.3)*10))
        return downfactors
    else:
        downfactors = 1.0 / (
            1.0 + 0.1 * np.exp(-(x - 11.8) * 10)
        )  # LRG only, default 12.3, set to 12.0 for z = 1.1
        downfactors[x > 13.0] = 1
        return downfactors


# # new version for negative alpha ELG satellites
# def subsample_halos(m, MT):
#     x = np.log10(m)
#     downfactors = np.zeros(len(x))
#     if MT:
#         mask1 = x < 11.35
#         mask2 = x < 11.52
#         downfactors[mask1] = 0.5/(1.0 + 10*np.exp(-(x[mask1] - 11.1)*25))
#         downfactors[mask2&(~mask1)] = 0.8/(1.0 + 10*np.exp(-(x[mask2&(~mask1)] - 11.25)*25))
#         downfactors[~mask2] = 1.0/(1.0 + 0.1*np.exp(-(x[~mask2] - 11.6)*10))
#         # save all halos that could host a satellite
#         # downfactors[x>11.2] = 1
#         return downfactors
#     else:
#         downfactors = 1.0/(1.0 + 0.1*np.exp(-(x - 12.0)*10)) # LRG only, might be able to step back to 12.5 depending on bestfit
#         downfactors[x > 13.0] = 1
#         return downfactors

# # standard satellites
# def submask_particles(m_in, n_in, MT):
#     x = np.log10(m_in)

#     if MT:
#         if m_in < 1e11:
#             return np.zeros(n_in)
#         else:
#             # a target number of particles
#             ntarget = np.minimum(n_in, int(1 + 1.5*10**(x-13)))
#             submask = np.zeros(n_in).astype(int)
#             submask[np.random.choice(n_in, ntarget, replace = False)] = 1
#             return submask
#     else:
#         if 10**x < 1e12:
#             return np.zeros(n_in) # essentially removing particles in halos below Mmin
#         else:
#             ntarget = np.minimum(n_in, int(1 + 1.5*10**(x-13)))
#             submask = np.zeros(n_in).astype(int)
#             submask[np.random.choice(n_in, ntarget, replace = False)] = 1
#             return submask


# conformity fix
def submask_particles(m_in, n_in, MT):
    x = np.log10(m_in)

    if MT:
        if m_in < 1e11:
            return np.zeros(n_in)
        else:
            # a target number of particles
            # ntarget = np.minimum(n_in, int(1 + 1.5*10**(x-11.8)))
            ntarget = np.minimum(n_in, int(1 + 1.5 * 10 ** (x - 12.5)))
            ntarget = np.minimum(ntarget, 100)
            submask = np.zeros(n_in).astype(int)
            submask[np.random.choice(n_in, ntarget, replace=False)] = 1
            return submask
    else:
        if 10**x < 1e12:
            return np.zeros(n_in)  # essentially removing particles in halos below Mmin
        else:
            ntarget = np.minimum(n_in, int(1 + 1.5 * 10 ** (x - 13)))
            submask = np.zeros(n_in).astype(int)
            submask[np.random.choice(n_in, ntarget, replace=False)] = 1
            return submask


def get_vertices_cube(units=0.5, N=3):
    vertices = 2 * ((np.arange(2**N)[:, None] & (1 << np.arange(N))) > 0) - 1
    return vertices * units


def is_in_cube(x_pos, y_pos, z_pos, verts):
    x_min = np.min(verts[:, 0])
    x_max = np.max(verts[:, 0])
    y_min = np.min(verts[:, 1])
    y_max = np.max(verts[:, 1])
    z_min = np.min(verts[:, 2])
    z_max = np.max(verts[:, 2])

    mask = (
        (x_pos > x_min)
        & (x_pos <= x_max)
        & (y_pos > y_min)
        & (y_pos <= y_max)
        & (z_pos > z_min)
        & (z_pos <= z_max)
    )
    return mask


def gen_rand(N, chi_min, chi_max, fac, Lbox, offset, origins):
    # number of randoms to generate
    N_rands = fac * N

    # location of observer
    origin = origins[0]

    # generate randoms on the unit sphere
    if (
        origins.shape[0] > 1
    ):  # not true of only the huge box where the origin is at the center
        assert origins.shape[0] == 3
        assert np.all(origins[1] + np.array([0.0, 0.0, Lbox]) == origins[0])
        assert np.all(origins[2] + np.array([0.0, Lbox, 0.0]) == origins[0])
        costheta = np.random.rand(N_rands)  # between zero and one
        phi = np.random.rand(N_rands) * np.pi / 2.0
    else:
        costheta = np.random.rand(N_rands) * 2.0 - 1.0
        phi = np.random.rand(N_rands) * 2.0 * np.pi
    theta = np.arccos(costheta)
    x_cart = np.sin(theta) * np.cos(phi)
    y_cart = np.sin(theta) * np.sin(phi)
    z_cart = np.cos(theta)
    rands_chis = np.random.rand(N_rands) * (chi_max - chi_min) + chi_min

    # multiply the unit vectors by that
    x_cart *= rands_chis
    y_cart *= rands_chis
    z_cart *= rands_chis

    # vector between centers of the cubes and origin in Mpc/h (i.e. placing observer at 0, 0, 0)
    box0 = np.array([0.0, 0.0, 0.0]) - origin
    if (
        origins.shape[0] > 1
    ):  # not true of only the huge box where the origin is at the center
        assert origins.shape[0] == 3
        assert np.all(origins[1] + np.array([0.0, 0.0, Lbox]) == origins[0])
        assert np.all(origins[2] + np.array([0.0, Lbox, 0.0]) == origins[0])
        box1 = np.array([0.0, 0.0, Lbox]) - origin
        box2 = np.array([0.0, Lbox, 0.0]) - origin

    # vertices of a cube centered at 0, 0, 0
    vert = get_vertices_cube(units=Lbox / 2.0)

    # remove edges because this is inherent to the light cone catalogs
    x_vert = vert[:, 0]
    y_vert = vert[:, 1]
    z_vert = vert[:, 2]
    vert[x_vert < 0, 0] += offset
    vert[x_vert > 0, 0] -= offset
    vert[y_vert < 0, 1] += offset
    vert[z_vert < 0, 2] += offset
    if origins.shape[0] == 1:  # true of the huge box where the origin is at the center
        vert[y_vert > 0, 1] -= offset
        vert[z_vert > 0, 2] -= offset

    # vertices for all three boxes
    vert0 = box0 + vert
    if origins.shape[0] > 1 and chi_max >= (
        Lbox - offset
    ):  # not true of only the huge boxes and at low zs for base
        vert1 = box1 + vert
        vert2 = box2 + vert

    # mask for whether or not the coordinates are within the vertices
    mask0 = is_in_cube(x_cart, y_cart, z_cart, vert0)
    if origins.shape[0] > 1 and chi_max >= (Lbox - offset):
        mask1 = is_in_cube(x_cart, y_cart, z_cart, vert1)
        mask2 = is_in_cube(x_cart, y_cart, z_cart, vert2)
        mask = mask0 | mask1 | mask2
    else:
        mask = mask0
    print('masked randoms = ', np.sum(mask) * 100.0 / len(mask))

    rands_pos = np.vstack((x_cart[mask], y_cart[mask], z_cart[mask])).T
    rands_chis = rands_chis[mask]
    rands_pos += origin

    return rands_pos, rands_chis


def concat_to_arr(lists, dtype=np.int64):
    """Concatenate an iterable of lists to a flat Numpy array.
    Returns the concatenated array and the index where each list starts.
    """
    starts = np.empty(len(lists) + 1, dtype=np.int64)
    starts[0] = 0
    starts[1:] = np.cumsum(
        np.fromiter((len(ell) for ell in lists), count=len(lists), dtype=np.int64)
    )
    N = starts[-1]
    res = np.fromiter(itertools.chain.from_iterable(lists), count=N, dtype=dtype)
    return res, starts


@njit(parallel=True)
def calc_Menv(masses, inner_arr, inner_starts, outer_arr, outer_starts):
    N = len(inner_starts) - 1
    Menv = np.zeros(N, dtype=np.float32)
    for p in numba.prange(N):
        j = inner_starts[p]
        k = inner_starts[p + 1]
        inner_mass = np.sum(masses[inner_arr[j:k]])

        j = outer_starts[p]
        k = outer_starts[p + 1]
        outer_mass = np.sum(masses[outer_arr[j:k]])

        Menv[p] = outer_mass - inner_mass
    return Menv


@njit(parallel=True)
def calc_fenv_opt(Menv, mbins, halosM):
    fenv_rank = np.zeros(len(Menv))
    for ibin in numba.prange(len(mbins) - 1):
        mmask = (halosM > mbins[ibin]) & (halosM < mbins[ibin + 1])
        Nmask = np.sum(mmask)
        if Nmask > 1:
            new_fenv_rank = Menv[mmask].argsort().argsort()
            fenv_rank[mmask] = (
                new_fenv_rank / (Nmask - 1) - 0.5
            )  # max rank is always Nmask - 1
    return fenv_rank


def do_Menv_from_tree(
    allpos, allmasses, r_inner, r_outer, halo_lc, Lbox, nthread, mcut=1e11
):
    """Calculate a local mass environment by taking the difference in
    total neighbor halo mass at two apertures
    """

    if halo_lc:
        querypos = allpos
        treebox = None  # periodicity not needed for halo light cones
    else:
        # note that periodicity exists only in y and z directions
        querypos = (
            allpos + Lbox / 2.0
        ) % Lbox  # needs to be within 0 and Lbox for periodicity
        treebox = Lbox

    mmask = allmasses > mcut
    pos_cut = querypos[mmask]

    print('Building and querying trees for mass env calculation')
    querypos_tree = cKDTree(querypos, boxsize=treebox)
    if isinstance(r_inner, (list, tuple, np.ndarray)):
        r_inner = np.array(r_inner)[mmask]
    allinds_inner = querypos_tree.query_ball_point(pos_cut, r=r_inner, workers=nthread)
    inner_arr, inner_starts = concat_to_arr(allinds_inner)  # 7 sec
    del allinds_inner
    gc.collect()

    if isinstance(r_outer, (list, tuple, np.ndarray)):
        r_outer = np.array(r_outer)[mmask]
    allinds_outer = querypos_tree.query_ball_point(pos_cut, r=r_outer, workers=nthread)
    del querypos, querypos_tree
    gc.collect()

    outer_arr, outer_starts = concat_to_arr(allinds_outer)
    del allinds_outer
    gc.collect()

    print('starting Menv')
    numba.set_num_threads(nthread)

    Menv = np.zeros(len(allmasses))
    Menv[mmask] = calc_Menv(allmasses, inner_arr, inner_starts, outer_arr, outer_starts)
    return Menv


def prepare_slab(
    i,
    savedir,
    simdir,
    simname,
    z_mock,
    z_type,
    tracer_flags,
    MT,
    want_ranks,
    want_AB,
    want_shear,
    shearmark,
    cleaning,
    newseed,
    halo_lc=False,
    nthread=1,
    overwrite=1,
    mcut=1e11,
    rad_outer=10,
):
    outfilename_halos = (
        savedir
        + '/halos_xcom_'
        + str(i)
        + '_seed'
        + str(newseed)
        + '_abacushod_oldfenv'
    )
    outfilename_particles = (
        savedir
        + '/particles_xcom_'
        + str(i)
        + '_seed'
        + str(newseed)
        + '_abacushod_oldfenv'
    )
    print('processing slab ', i)
    if MT:
        outfilename_halos += '_MT'
        outfilename_particles += '_MT'
    if want_ranks:
        outfilename_particles += '_withranks'
    outfilename_particles += '_new.h5'
    outfilename_halos += '_new.h5'

    np.random.seed(newseed + i)
    # if file already exists, just skip
    overwrite = int(overwrite)
    if (
        (not overwrite)
        and (os.path.exists(outfilename_halos))
        and (os.path.exists(outfilename_particles))
    ):
        print('files exists, skipping ', i)
        return 0

    # load the halo catalog slab
    print('loading halo catalog ')
    if halo_lc:
        slabname = (
            simdir
            + '/'
            + simname
            + '/z'
            + str(z_mock).ljust(5, '0')
            + '/lc_halo_info.asdf'
        )
        id_key = 'index_halo'
        pos_key = 'pos_interp'
        vel_key = 'vel_interp'
        N_key = 'N_interp'
    else:
        slabname = (
            simdir
            + '/'
            + simname
            + '/halos/z'
            + str(z_mock).ljust(5, '0')
            + '/halo_info/halo_info_'
            + str(i).zfill(3)
            + '.asdf'
        )
        id_key = 'id'
        pos_key = 'x_L2com'
        vel_key = 'v_L2com'
        N_key = 'N'

    if z_type == 'primary' or z_type == 'lightcone':
        cat = CompaSOHaloCatalog(
            slabname,
            subsamples=dict(A=True, rv=True),
            fields=[
                N_key,
                pos_key,
                vel_key,
                'r90_L2com',
                'r25_L2com',
                'r98_L2com',
                'npstartA',
                'npoutA',
                id_key,
                'sigmav3d_L2com',
            ],
            cleaned=cleaning,
        )
    else:
        cat = CompaSOHaloCatalog(
            slabname,
            fields=[
                N_key,
                pos_key,
                vel_key,
                'r90_L2com',
                'r25_L2com',
                'r98_L2com',
                'npstartA',
                'npoutA',
                id_key,
                'sigmav3d_L2com',
            ],
            cleaned=cleaning,
        )
    assert halo_lc == cat.halo_lc

    halos = cat.halos
    if halo_lc:
        halos['id'] = halos[id_key]
        halos['x_L2com'] = halos[pos_key]
        halos['v_L2com'] = halos[vel_key]
        halos['N'] = halos[N_key]
    if cleaning:
        halos = halos[halos['N'] > 0]

    if z_type == 'primary' or z_type == 'lightcone':
        parts = cat.subsamples
    header = cat.header
    Lbox = cat.header['BoxSizeHMpc']
    Mpart = header['ParticleMassHMsun']  # msun / h
    H0 = header['H0']
    h = H0 / 100.0

    # # form a halo table of the columns i care about
    # creating a mask of which halos to keep, which halos to drop
    p_halos = subsample_halos(halos['N'] * Mpart, MT)
    mask_halos = np.random.random(len(halos)) < p_halos
    print('total number of halos, ', len(halos), 'keeping ', np.sum(mask_halos))

    halos['mask_subsample'] = mask_halos
    halos['multi_halos'] = 1.0 / p_halos

    # only generate fenv ranks and c ranks if the user wants to enable secondary biases
    if want_AB:
        nbins = 100
        mbins = np.logspace(np.log10(mcut), 15.5, nbins + 1)

        # # grid based environment calculation
        # dens_grid = np.array(h5py.File(savedir+"/density_field.h5", 'r')['dens'])
        # ixs = np.floor((np.array(halos['x_L2com']) + Lbox/2) / (Lbox/N_dim)).astype(np.int) % N_dim
        # halos_overdens = dens_grid[ixs[:, 0], ixs[:, 1], ixs[:, 2]]
        # fenv_rank = np.zeros(len(halos))
        # for ibin in range(nbins):
        #     mmask = (halos['N']*Mpart > mbins[ibin]) & (halos['N']*Mpart < mbins[ibin + 1])
        #     if np.sum(mmask) > 0:
        #         if np.sum(mmask) == 1:
        #             fenv_rank[mmask] = 0
        #         else:
        #             new_fenv_rank = halos_overdens[mmask].argsort().argsort()
        #             fenv_rank[mmask] = new_fenv_rank / np.max(new_fenv_rank) - 0.5
        # halos['fenv_rank'] = fenv_rank

        allpos = halos['x_L2com']
        allmasses = halos['N'] * Mpart

        if halo_lc:
            # origin dependent and simulation dependent
            origins = np.array(header['LightConeOrigins']).reshape(-1, 3)
            alldist = np.sqrt(np.sum((allpos - origins[0]) ** 2.0, axis=1))
            offset = 10.0  # offset intrinsic to light cones catalogs (removing edges +/- 10 Mpc/h from the sides of the box)

            r_min = alldist.min()
            r_max = alldist.max()
            x_min_edge = -(Lbox / 2.0 - offset - rad_outer)
            y_min_edge = -(Lbox / 2.0 - offset - rad_outer)
            z_min_edge = -(Lbox / 2.0 - offset - rad_outer)
            x_max_edge = Lbox / 2.0 - offset - rad_outer
            r_min_edge = alldist.min() + rad_outer
            r_max_edge = alldist.max() - rad_outer
            if (
                origins.shape[0] == 1
            ):  # true only of the huge box where the origin is at the center
                y_max_edge = Lbox / 2.0 - offset - rad_outer
                z_max_edge = Lbox / 2.0 - offset - rad_outer
            else:
                y_max_edge = 3.0 / 2 * Lbox - rad_outer
                z_max_edge = 3.0 / 2 * Lbox - rad_outer

            bounds_edge = (
                (x_min_edge <= allpos[:, 0])
                & (x_max_edge >= allpos[:, 0])
                & (y_min_edge <= allpos[:, 1])
                & (y_max_edge >= allpos[:, 1])
                & (z_min_edge <= allpos[:, 2])
                & (z_max_edge >= allpos[:, 2])
                & (r_min_edge <= alldist)
                & (r_max_edge >= alldist)
            )
            index_bounds = np.arange(allpos.shape[0], dtype=int)[~bounds_edge]
            del bounds_edge, alldist

            if len(index_bounds) > 0:
                # factor of rands to generate
                rand = 50  # to ensure 6 times more randoms than haloes in the octant.
                rand_N = allpos.shape[0] * rand

                # generate randoms in L shape
                randpos, randdist = gen_rand(
                    allpos.shape[0], r_min, r_max, rand, Lbox, offset, origins
                )
                rand_n = rand_N / (4.0 / 3.0 * np.pi * (r_max**3 - r_min**3))

                # boundaries of the random particles for cutting
                randbounds_edge = (
                    (x_min_edge <= randpos[:, 0])
                    & (x_max_edge >= randpos[:, 0])
                    & (y_min_edge <= randpos[:, 1])
                    & (y_max_edge >= randpos[:, 1])
                    & (z_min_edge <= randpos[:, 2])
                    & (z_max_edge >= randpos[:, 2])
                    & (r_min_edge <= randdist)
                    & (r_max_edge >= randdist)
                )
                randpos = randpos[~randbounds_edge]
                del randbounds_edge, randdist

                if randpos.shape[0] > 0:
                    # random points on the edges
                    rand_N = randpos.shape[0]
                    randpos_tree = cKDTree(randpos)
                    randinds_inner = randpos_tree.query_ball_point(
                        allpos[index_bounds],
                        r=halos['r98_L2com'][index_bounds],
                        workers=nthread,
                    )
                    randinds_outer = randpos_tree.query_ball_point(
                        allpos[index_bounds], r=rad_outer, workers=nthread
                    )
                    rand_norm = np.zeros(len(index_bounds))
                    for ind in np.arange(len(index_bounds)):
                        rand_norm[ind] = len(randinds_outer[ind]) - len(
                            randinds_inner[ind]
                        )
                    rand_norm /= (
                        (rad_outer**3.0 - halos['r98_L2com'][index_bounds] ** 3.0)
                        * 4.0
                        / 3.0
                        * np.pi
                        * rand_n
                    )  # expected number
                else:
                    rand_norm = np.ones(len(index_bounds))

        Menv = do_Menv_from_tree(
            allpos,
            allmasses,
            r_inner=halos['r98_L2com'],
            r_outer=rad_outer,
            halo_lc=halo_lc,
            Lbox=Lbox,
            nthread=nthread,
            mcut=mcut,
        )
        gc.collect()

        if halo_lc and len(index_bounds) > 0:
            mask = rand_norm == 0.0
            rand_norm[mask] = 1.0
            tmp = Menv[index_bounds]
            tmp /= rand_norm  # fixed (pull request #142)
            tmp[mask] = 0.0
            Menv[index_bounds] = tmp
            del mask
            gc.collect()
        halos['fenv_rank'] = calc_fenv_opt(Menv, mbins, allmasses)

        # compute delta concentration
        print('computing c rank')
        halos_c = halos['r98_L2com'] / halos['r25_L2com']
        deltac_rank = np.zeros(len(halos))
        for ibin in range(nbins):
            mmask = (allmasses > mbins[ibin]) & (allmasses < mbins[ibin + 1])
            if np.sum(mmask) > 0:
                if np.sum(mmask) == 1:
                    deltac_rank[mmask] = 0
                else:
                    new_deltac = halos_c[mmask] - np.median(halos_c[mmask])
                    new_deltac_rank = new_deltac.argsort().argsort()
                    deltac_rank[mmask] = new_deltac_rank / np.max(new_deltac_rank) - 0.5
        halos['deltac_rank'] = deltac_rank

    else:
        halos['fenv_rank'] = np.zeros(len(halos))
        halos['deltac_rank'] = np.zeros(len(halos))

    if want_shear:
        assert len(np.unique(shearmark.shape)) == 1
        N_dim = len(shearmark)
        cell = Lbox / N_dim
        shear_rank = np.zeros(len(halos))
        for ibin in range(nbins):
            mmask = (allmasses > mbins[ibin]) & (allmasses < mbins[ibin + 1])
            if np.sum(mmask) > 0:
                if np.sum(mmask) == 1:
                    deltac_rank[mmask] = 0
                else:
                    GroupPos = (halos[mmask]['x_L2com'] / cell).astype(int) % N_dim
                    halo_shears = interpn(
                        (np.arange(N_dim), np.arange(N_dim), np.arange(N_dim)),
                        shearmark,
                        GroupPos,
                    )
                    new_shear_rank = halo_shears.argsort().argsort()
                    shear_rank[mmask] = new_shear_rank / np.max(new_shear_rank) - 0.5
        halos['shear_rank'] = shear_rank
        print('finished shear compute')
    else:
        halos['shear_rank'] = np.zeros(len(halos))

    # the new particle start, len, and multiplier
    halos_pstart = halos['npstartA']
    halos_pnum = halos['npoutA']
    halos_pstart_new = np.zeros(len(halos))
    halos_pnum_new = np.zeros(len(halos))

    # particle arrays for ranks and mask
    if z_type == 'primary' or z_type == 'lightcone':
        mask_parts = np.zeros(len(parts))
        len_old = len(parts)
        ranks_parts = np.full(len_old, -1.0)
        ranksv_parts = np.full(len_old, -1.0)
        ranksr_parts = np.full(len_old, -1.0)
        ranksp_parts = np.full(len_old, -1.0)
        ranksc_parts = np.full(len_old, -1.0)
        # pos_parts = np.full((len_old, 3), -1.0)
        # vel_parts = np.full((len_old, 3), -1.0)
        hvel_parts = np.full((len_old, 3), -1.0)
        Mh_parts = np.full(len_old, -1.0)
        Np_parts = np.full(len_old, -1.0)
        downsample_parts = np.full(len_old, -1.0)
        idh_parts = np.full(len_old, -1)
        deltach_parts = np.full(len_old, -1.0)
        fenvh_parts = np.full(len_old, -1.0)
        shearh_parts = np.full(len_old, -1.0)

        print('compiling particle subsamples')
        start_tracker = 0
        for j in np.arange(len(halos)):
            if j % 10000 == 0:
                print('halo id', j, end='\r')
            if mask_halos[j] and halos['npoutA'][j] > 0:
                # subsample_factor = subsample_particles(halos['N'][j] * Mpart, halos['npoutA'][j], MT)
                # submask = np.random.binomial(n = 1, p = subsample_factor, size = halos_pnum[j])
                submask = submask_particles(
                    halos['N'][j] * Mpart, halos['npoutA'][j], MT
                )

                # updating the particles' masks, downsample factors, halo mass
                mask_parts[halos_pstart[j] : halos_pstart[j] + halos_pnum[j]] = submask
                # print(j, halos_pstart, halos_pnum, p_halos, downsample_parts)
                downsample_parts[halos_pstart[j] : halos_pstart[j] + halos_pnum[j]] = (
                    p_halos[j]
                )
                hvel_parts[halos_pstart[j] : halos_pstart[j] + halos_pnum[j]] = halos[
                    'v_L2com'
                ][j]
                Mh_parts[halos_pstart[j] : halos_pstart[j] + halos_pnum[j]] = (
                    halos['N'][j] * Mpart
                )  # in msun / h
                Np_parts[halos_pstart[j] : halos_pstart[j] + halos_pnum[j]] = np.sum(
                    submask
                )
                idh_parts[halos_pstart[j] : halos_pstart[j] + halos_pnum[j]] = halos[
                    'id'
                ][j]
                deltach_parts[halos_pstart[j] : halos_pstart[j] + halos_pnum[j]] = (
                    halos['deltac_rank'][j]
                )
                fenvh_parts[halos_pstart[j] : halos_pstart[j] + halos_pnum[j]] = halos[
                    'fenv_rank'
                ][j]
                shearh_parts[halos_pstart[j] : halos_pstart[j] + halos_pnum[j]] = halos[
                    'shear_rank'
                ][j]

                # updating the pstart, pnum, for the halos
                halos_pstart_new[j] = start_tracker
                halos_pnum_new[j] = np.sum(submask)
                start_tracker += np.sum(submask)

                if want_ranks:
                    if np.sum(submask) == 0:
                        continue
                    # extract particle index
                    indices_parts = np.arange(
                        halos_pstart[j], halos_pstart[j] + halos_pnum[j]
                    )[submask.astype(bool)]
                    indices_parts = indices_parts.astype(int)
                    if np.sum(submask) == 1:
                        ranks_parts[indices_parts] = 0
                        ranksv_parts[indices_parts] = 0
                        ranksp_parts[indices_parts] = 0
                        ranksr_parts[indices_parts] = 0
                        ranksc_parts[indices_parts] = 0
                        continue

                    # make the rankings
                    theseparts = parts[
                        halos_pstart[j] : halos_pstart[j] + halos_pnum[j]
                    ][submask.astype(bool)]
                    theseparts_pos = theseparts['pos']
                    theseparts_vel = theseparts['vel']
                    theseparts_halo_pos = halos['x_L2com'][j]
                    theseparts_halo_vel = halos['v_L2com'][j]

                    # construct particle tree to find nearest neighbors
                    parts_tree = cKDTree(
                        parts[halos_pstart[j] : halos_pstart[j] + halos_pnum[j]]['pos']
                    )
                    dist2_neighbors = parts_tree.query(theseparts_pos, k=2)[0][:, 1]
                    newranksc = dist2_neighbors.argsort().argsort()
                    ranksc_parts[indices_parts] = (
                        newranksc - np.mean(newranksc)
                    ) / np.mean(newranksc)

                    dist2_rel = np.sum(
                        (theseparts_pos - theseparts_halo_pos) ** 2, axis=1
                    )
                    newranks = dist2_rel.argsort().argsort()
                    ranks_parts[indices_parts] = (
                        newranks - np.mean(newranks)
                    ) / np.mean(newranks)

                    v2_rel = np.sum((theseparts_vel - theseparts_halo_vel) ** 2, axis=1)
                    newranksv = v2_rel.argsort().argsort()
                    ranksv_parts[indices_parts] = (
                        newranksv - np.mean(newranksv)
                    ) / np.mean(newranksv)

                    # get rps
                    # calc relative positions
                    r_rel = theseparts_pos - theseparts_halo_pos
                    r0 = np.sqrt(np.sum(r_rel**2, axis=1))
                    r_rel_norm = r_rel / r0[:, None]

                    # list of peculiar velocities of the particles
                    vels_rel = theseparts_vel - theseparts_halo_vel  # velocity km/s
                    # relative speed to halo center squared
                    v_rel2 = np.sum(vels_rel**2, axis=1)

                    # calculate radial and tangential peculiar velocity
                    vel_rad = np.sum(vels_rel * r_rel_norm, axis=1)
                    newranksr = vel_rad.argsort().argsort()
                    ranksr_parts[indices_parts] = (
                        newranksr - np.mean(newranksr)
                    ) / np.mean(newranksr)

                    # radial component
                    v_rad2 = vel_rad**2  # speed
                    # tangential component
                    v_tan2 = v_rel2 - v_rad2

                    # compute the perihelion distance for NFW profile
                    m = halos['N'][j] * Mpart / h  # in kg
                    rs = halos['r25_L2com'][j]
                    c = halos['r98_L2com'][j] / rs
                    r0_kpc = r0 * 1000  # kpc
                    alpha = (
                        1.0
                        / (np.log(1 + c) - c / (1 + c))
                        * 2
                        * 6.67e-11
                        * m
                        * 2e30
                        / r0_kpc
                        / 3.086e19
                        / 1e6
                    )

                    # iterate a few times to solve for rp
                    x2 = v_tan2 / (v_tan2 + v_rad2)

                    num_iters = 20  # how many iterations do we want
                    factorA = v_tan2 + v_rad2
                    factorB = np.log(1 + r0_kpc / rs)
                    for it in range(num_iters):
                        oldx = np.sqrt(x2)
                        x2 = v_tan2 / (
                            factorA
                            + alpha * (np.log(1 + oldx * r0_kpc / rs) / oldx - factorB)
                        )
                    x2[np.isnan(x2)] = 1
                    # final perihelion distance
                    rp2 = r0_kpc**2 * x2
                    newranksp = rp2.argsort().argsort()
                    ranksp_parts[indices_parts] = (
                        newranksp - np.mean(newranksp)
                    ) / np.mean(newranksp)

            else:
                halos_pstart_new[j] = -1
                halos_pnum_new[j] = -1

    halos['npstartA'] = halos_pstart_new
    halos['npoutA'] = halos_pnum_new
    halos['randoms'] = np.random.random(len(halos))  # attaching random numbers
    halos['randoms_exp'] = (
        np.random.randint(0, 2, size=(len(halos), 3)) * 2 - 1
    ) * np.random.exponential(
        scale=np.repeat(halos['sigmav3d_L2com'], 3).reshape((-1, 3)) / np.sqrt(3),
        size=(len(halos), 3),
    )  # attaching random numbers
    halos['randoms_gaus_vrms'] = np.random.normal(
        loc=0,
        scale=np.repeat(halos['sigmav3d_L2com'], 3).reshape((-1, 3)) / np.sqrt(3),
        size=(len(halos), 3),
    )  # attaching random numbers

    # output halo file
    print('outputting new halo file ')
    # output_dir = savedir+'/halos_xcom_'+str(i)+'_seed'+str(newseed)+'_abacushodMT_new.h5'
    if os.path.exists(outfilename_halos):
        os.remove(outfilename_halos)
    newfile = h5py.File(outfilename_halos, 'w')
    newfile.create_dataset('halos', data=halos[mask_halos])
    newfile.close()

    # output the new particle file
    if z_type == 'primary' or z_type == 'lightcone':
        print('adding rank fields to particle data ')
        mask_parts = mask_parts.astype(bool)
        parts = parts[mask_parts]
        print(
            'pre process particle number ',
            len_old,
            ' post process particle number ',
            len(parts),
        )
        if want_ranks:
            parts['ranks'] = ranks_parts[mask_parts]
            parts['ranksv'] = ranksv_parts[mask_parts]
            parts['ranksr'] = ranksr_parts[mask_parts]
            parts['ranksp'] = ranksp_parts[mask_parts]
            parts['ranksc'] = ranksc_parts[mask_parts]
        parts['downsample_halo'] = downsample_parts[mask_parts]
        parts['halo_vel'] = hvel_parts[mask_parts]
        parts['halo_mass'] = Mh_parts[mask_parts]
        parts['Np'] = Np_parts[mask_parts]
        parts['halo_id'] = idh_parts[mask_parts]
        parts['randoms'] = np.random.random(len(parts))
        parts['halo_deltac'] = deltach_parts[mask_parts]
        parts['halo_fenv'] = fenvh_parts[mask_parts]
        parts['halo_shear'] = shearh_parts[mask_parts]

        print(
            'are there any negative particle values? ',
            np.sum(parts['downsample_halo'] < 0),
            np.sum(parts['halo_mass'] < 0),
        )
        print('outputting new particle file ')
        # output_dir = savedir+'/particles_xcom_'+str(i)+'_seed'+str(newseed)+'_abacushodMT_new.h5'
        if os.path.exists(outfilename_particles):
            os.remove(outfilename_particles)
        newfile = h5py.File(outfilename_particles, 'w')
        newfile.create_dataset('particles', data=parts)
        newfile.close()

        print(
            'pre process particle number ',
            len_old,
            ' post process particle number ',
            len(parts),
        )


def calc_shearmark(simdir, simname, z_mock, N_dim, R, fn, partdown=100):
    start = time.time()

    fns = glob.glob(
        simdir
        + '/'
        + simname
        + '/halos/z'
        + str(z_mock).ljust(5, '0')
        + '/field_rv_A/*asdf'
    )
    partpos = []
    for efn in fns:
        ecat = read_asdf(efn, load=['pos'])
        partpos += [
            ecat['pos'][
                np.random.choice(
                    len(ecat['pos']),
                    size=int(len(ecat['pos']) / partdown),
                    replace=False,
                )
            ]
        ]

    fns = glob.glob(
        simdir
        + '/'
        + simname
        + '/halos/z'
        + str(z_mock).ljust(5, '0')
        + '/halo_rv_A/*asdf'
    )
    for efn in fns:
        ecat = read_asdf(efn, load=['pos'])
        partpos += [
            ecat['pos'][
                np.random.choice(
                    len(ecat['pos']),
                    size=int(len(ecat['pos']) / partdown),
                    replace=False,
                )
            ]
        ]

    pos_parts = np.concatenate(partpos)
    print('compiled all particles', len(pos_parts), 'took time', time.time() - start)

    start = time.time()
    cat = CompaSOHaloCatalog(
        simdir + '/' + simname + '/halos/z' + str(z_mock).ljust(5, '0'),
        fields=['N'],
        cleaned=True,
    )
    header = cat.header
    Lbox = header['BoxSizeHMpc']
    Lbox / N_dim
    print('compiled all halos', 'took time', time.time() - start)

    start = time.time()
    # dens = np.zeros((N_dim, N_dim, N_dim))
    # numba_tsc_3D(pos_parts, dens, Lbox)
    dens = tsc_parallel(pos_parts, N_dim, Lbox)
    print('finished TSC, took time', time.time() - start)
    start = time.time()
    dens_smooth = smooth_density(dens, R, N_dim, Lbox)
    print('finished smoothing, took time', time.time() - start)
    start = time.time()
    shearmark = get_shear(dens_smooth, N_dim, Lbox)
    print('finished shear mark, took time', time.time() - start)

    # output file
    np.save(fn + '.npy', shearmark)
    return shearmark


def main(
    path2config,
    params=None,
    alt_simname=None,
    alt_z=None,
    newseed=600,
    halo_lc=False,
    overwrite=1,
):
    print('compiling compaso halo catalogs into subsampled catalogs')

    config = yaml.safe_load(open(path2config))
    # update params if needed
    if params:
        config.update(params)
    if alt_simname:
        config['sim_params']['sim_name'] = alt_simname
    if alt_z:
        config['sim_params']['z_mock'] = alt_z

    simname = config['sim_params']['sim_name']  # "AbacusSummit_base_c000_ph006"
    simdir = config['sim_params']['sim_dir']
    z_mock = float(config['sim_params']['z_mock'])
    savedir = (
        config['sim_params']['subsample_dir']
        + simname
        + '/z'
        + str(z_mock).ljust(5, '0')
    )
    cleaning = config['sim_params']['cleaned_halos']
    if 'halo_lc' in config['sim_params'].keys():
        halo_lc = config['sim_params']['halo_lc']

    # build in some redshift checks
    ztype = None
    if halo_lc:
        ztype = 'lightcone'
    elif z_mock in [3.0, 2.5, 2.0, 1.7, 1.4, 1.1, 0.8, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]:
        ztype = 'primary'
    elif z_mock in [
        0.15,
        0.25,
        0.35,
        0.45,
        0.575,
        0.65,
        0.725,
        0.875,
        0.95,
        1.025,
        1.175,
        1.25,
        1.325,
        1.475,
        1.55,
        1.625,
        1.85,
        2.25,
        2.75,
        3.0,
        5.0,
        8.0,
    ]:
        ztype = 'secondary'
    else:
        raise Exception('illegal redshift')

    if halo_lc:
        halo_info_fns = [
            str(
                Path(simdir) / Path(simname) / ('z%4.3f' % z_mock) / 'lc_halo_info.asdf'
            )
        ]
    else:
        halo_info_fns = list(
            sorted(
                (
                    Path(simdir)
                    / Path(simname)
                    / 'halos'
                    / ('z%4.3f' % z_mock)
                    / 'halo_info'
                ).glob('*.asdf')
            )
        )
    numslabs = len(halo_info_fns)

    os.makedirs(savedir, exist_ok=True)

    if numslabs == 0:
        raise ValueError('prepare_sim could not find any slabs!')

    tracer_flags = config['HOD_params']['tracer_flags']
    MT = False
    if tracer_flags['ELG'] or tracer_flags['QSO']:
        MT = True
    want_ranks = config['HOD_params'].get('want_ranks', False)
    want_AB = config['HOD_params'].get('want_AB', False)
    want_shear = config['HOD_params'].get('want_shear', False)
    # if want shear, calculate shear field first
    if want_shear:
        if (not ztype == 'primary') and (not halo_lc):
            raise Exception('redshift does not have particle data, cant compute shear')
        Ndim = config['HOD_params'].get('shear_N', 1000)
        Rsm = config['HOD_params'].get('shear_R', 2)
        partdown = config['HOD_params'].get('partdown', 100)
        shear_fn = (
            savedir + '/shear_N' + str(Ndim) + '_R' + str(Rsm) + '_down' + str(partdown)
        )
        if os.path.exists(shear_fn + '.npy'):
            shearmark = np.load(shear_fn + '.npy')
        else:
            print('computing shear field')
            shearmark = calc_shearmark(
                simdir, simname, z_mock, Ndim, Rsm, shear_fn, partdown
            )
    else:
        shearmark = None
    # N_dim = config['HOD_params']['Ndim']
    nthread = config['prepare_sim'].get('Nthread_per_load', 'auto')
    if nthread == 'auto':
        nthread = (
            len(os.sched_getaffinity(0)) // config['prepare_sim']['Nparallel_load']
        )
        print(f'prepare_sim inferred Nthread_per_load = {nthread}')
    else:
        nthread = int(nthread)

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=config['prepare_sim']['Nparallel_load'],
        mp_context=multiprocessing.get_context('spawn'),
    ) as pool:
        futures = [
            pool.submit(
                prepare_slab,
                i,
                savedir=savedir,
                simdir=simdir,
                simname=simname,
                z_mock=z_mock,
                z_type=ztype,
                tracer_flags=tracer_flags,
                MT=MT,
                want_ranks=want_ranks,
                want_AB=want_AB,
                want_shear=want_shear,
                shearmark=shearmark,
                cleaning=cleaning,
                newseed=newseed,
                halo_lc=halo_lc,
                nthread=nthread,
                overwrite=overwrite,
            )
            for i in range(numslabs)
        ]

    # check that all futures succeeded
    for future in concurrent.futures.as_completed(futures):
        try:
            future.result()
        except concurrent.futures.process.BrokenProcessPool as bpp:
            raise RuntimeError(
                'A subprocess died in prepare_sim. Did prepare_slab() run out of memory?'
            ) from bpp
    # print("done, took time ", time.time() - start)


class ArgParseFormatter(
    argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter
):
    pass


if __name__ == '__main__':
    # parsing arguments
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=ArgParseFormatter
    )
    parser.add_argument(
        '--path2config', help='Path to the config file', default=DEFAULTS['path2config']
    )
    parser.add_argument(
        '--alt_simname',
        help='alternative simname to process, like "AbacusSummit_base_c000_ph003"',
    )
    parser.add_argument(
        '--alt_z',
        help='alternative z to process, like "0.8"',
        type=float,
    )
    parser.add_argument(
        '--newseed',
        help='alternative random number seed, positive integer',
        default=600,
        type=int,
    )
    parser.add_argument(
        '--overwrite', help='overwrite existing subsamples', default=1, type=int
    )
    args = vars(parser.parse_args())
    main(**args)

    print('done')
