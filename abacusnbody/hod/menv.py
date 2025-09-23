"""
Local mass environment calculation.
"""

import itertools
from typing import Literal

import numba
import numpy as np
from scipy.spatial import KDTree

from ..util import cumsum

__all__ = ['do_Menv_from_tree']

DEFAULT_BATCH_SIZE = 10**5


def do_Menv_from_tree(
    pos,
    mass,
    r_inner,
    r_outer,
    halo_lc,
    Lbox,
    nthread: int,
    mcut=1e11,
    batch_size: int = DEFAULT_BATCH_SIZE,
):
    """Calculate a local mass environment by taking the difference in
    total neighbor halo mass at two apertures. Neighbor mass includes
    all halos, but only halos above mcut are used as centers (0 returned
    for all others).
    """

    if halo_lc:
        treebox = None  # periodicity not needed for halo light cones
    else:
        # note that periodicity exists only in y and z directions
        # don't modify the user's input in place!
        pos = (pos + Lbox / 2.0) % Lbox  # needs to be within 0 and Lbox for periodicity
        treebox = Lbox

    mmask = mass > mcut
    pos_cut = pos[mmask]
    N = len(pos_cut)

    r_inner = np.asarray(r_inner)
    if r_inner.ndim > 0:
        r_inner = r_inner[mmask]

    r_outer = np.asarray(r_outer)
    if r_outer.ndim > 0:
        r_outer = r_outer[mmask]

    print('Building and querying trees for mass env calculation')
    tree = KDTree(pos, boxsize=treebox)

    # we're taking potentially large differences, use float64
    Menv_cut = np.zeros(N, dtype=np.float64)
    msum_in_batches(
        Menv_cut,
        pos_cut,
        mass,
        r_outer,
        tree,
        nthread=nthread,
        sign=1,
        batch_size=batch_size,
    )

    # now subtract the inner mass
    msum_in_batches(
        Menv_cut,
        pos_cut,
        mass,
        r_inner,
        tree,
        nthread=nthread,
        sign=-1,
        batch_size=batch_size,
    )

    Menv = np.zeros_like(mass)
    Menv[mmask] = Menv_cut

    return Menv


def msum_in_batches(
    msum_out,
    pos,
    mass,
    r,
    tree: KDTree,
    nthread: int,
    sign: Literal[1, -1] = 1,
    batch_size: int = DEFAULT_BATCH_SIZE,
):
    """Calculate the sum of masses within a radius r of each point in pos."""
    N = len(pos)

    for i in range(0, N, batch_size):
        j = min(i + batch_size, N)
        pbatch = pos[i:j]
        mout_batch = msum_out[i:j]
        if r.ndim > 0:
            rbatch = r[i:j]
        else:
            rbatch = r
        # mass is not batched because the indices from the tree query
        # are all relative to the original mass array
        msum_batch(mout_batch, pbatch, mass, rbatch, tree, sign, nthread)

    return msum_out


def msum_batch(
    out,
    pos,
    mass,
    r,
    tree: KDTree,
    sign: Literal[1, -1],
    nthread: int,
):
    inds, starts = query_inds(pos, r, tree, nthread)
    msum_core(
        out,
        mass,
        inds,
        starts,
        sign,
        nthread=nthread,
    )


def query_inds(pos, r, tree: KDTree, nthread: int):
    """Query the tree for indices of neighbors within radius r"""
    allinds = tree.query_ball_point(pos, r=r, workers=nthread)
    # flatten the list of lists
    inds, starts = concat_to_arr(allinds)
    return inds, starts


@numba.njit(parallel=True)
def msum_core(msum_out, masses, inds, starts, sign, nthread: int = 1):
    numba.set_num_threads(nthread)
    N = len(starts) - 1
    for p in numba.prange(N):
        j = starts[p]
        k = starts[p + 1]
        msum_out[p] += sign * np.sum(masses[inds[j:k]])


def concat_to_arr(lists, dtype=np.int64):
    """Concatenate an iterable of lists to a flat Numpy array.
    Returns the concatenated array and the index where each list starts.
    """
    starts = np.empty(len(lists) + 1, dtype=np.int64)
    cumsum([len(ell) for ell in lists], starts, initial=True, final=True)
    res = np.fromiter(
        itertools.chain.from_iterable(lists), count=starts[-1], dtype=dtype
    )
    return res, starts
