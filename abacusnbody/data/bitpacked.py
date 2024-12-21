"""
A collection of routines related to various Abacus bitpacked
formats, like RVint and encoding of information in the PIDs.

Most users will not use this module directly, but will instead use
:mod:`abacusnbody.data.compaso_halo_catalog` or
:func:`abacusnbody.data.read_abacus.read_asdf`.
"""

import numba as nb
import numpy as np

__all__ = ['unpack_rvint', 'unpack_pids']

# Constants
AUXDENS = np.uint64(0x07FE000000000000)
ZERODEN = np.uint64(49)  # The density bits are 49-58.

AUXXPID = np.uint64(0x7FFF)  # bits 0-14
AUXYPID = np.uint64(0x7FFF0000)  # bits 16-30
AUXZPID = np.uint64(0x7FFF00000000)  # bits 32-46
AUXPID = AUXXPID | AUXYPID | AUXZPID  # all of the above bits
AUXTAGGED = np.uint64(48)  # tagged bit is 48

# The names of the bit-packed PID fields that the user can request
PID_FIELDS = ['pid', 'lagr_pos', 'tagged', 'density', 'lagr_idx', 'packedpid']


def unpack_rvint(intdata, boxsize, float_dtype=np.float32, posout=None, velout=None):
    """
    Unpack rvint data into pos and vel.

    Parameters
    ----------
    intdata: ndarray of dtype np.int32
        The rvint data
    boxsize: float
        The box size, used to scale the positions
    float_dtype: np.dtype, optional
        The precision in which to store the unpacked values.
        Default: np.float32
    posout: ndarray, None, or False; optional
        The array in which to store the unpacked positions.
        `None` can be given (the default), in which case an
        array is constructed and retured as the first return value.
        `False` can be given, in which case the positions are not unpacked.
    velout: optional
        Same as posout, but for the velocities

    Returns
    -------
    pos,vel: tuple
        A tuple of the unpacked position and velocity arrays,
        or the number of unpacked particles if an output array
        was given.

    """
    intdata = intdata.reshape(-1, 3)
    assert intdata.dtype == np.int32
    N = len(intdata)

    if posout is None:
        _posout = np.empty((N, 3), dtype=float_dtype)
    elif posout is False:
        _posout = None
    else:
        # In NumPy >= 2.1, we can use arr.reshape(..., copy=False).
        # This is a workaround for earlier NumPy.
        _posout = posout.view()
        _posout.shape = (-1, 3)

    if velout is None:
        _velout = np.empty((N, 3), dtype=float_dtype)
    elif velout is False:
        _velout = None
    else:
        _velout = velout.view()
        _velout.shape = (-1, 3)

    _unpack_rvint(intdata, boxsize, _posout, _velout)

    ret = []
    if posout is None:
        ret += [_posout]
    elif posout is False:
        ret += [0]
    else:
        ret += [N]

    if velout is None:
        ret += [_velout]
    elif velout is False:
        ret += [0]
    else:
        ret += [N]

    return tuple(ret)


@nb.njit
def _unpack_rvint(intdata, boxsize, posout, velout):
    """Helper for unpack_rvint"""

    N = len(intdata)
    posscale = boxsize / 1e6
    velscale = 6000.0 / 2048
    vmask = np.uint32(0xFFF)

    for i in range(N):
        if posout is not None:
            posout[i, 0] = (intdata[i, 0] >> np.uint32(12)) * posscale
            posout[i, 1] = (intdata[i, 1] >> np.uint32(12)) * posscale
            posout[i, 2] = (intdata[i, 2] >> np.uint32(12)) * posscale
        if velout is not None:
            velout[i, 0] = ((intdata[i, 0] & vmask) - 2048) * velscale
            velout[i, 1] = ((intdata[i, 1] & vmask) - 2048) * velscale
            velout[i, 2] = ((intdata[i, 2] & vmask) - 2048) * velscale


def unpack_pids(
    packed,
    box=None,
    ppd=None,
    pid=False,
    lagr_pos=False,
    tagged=False,
    density=False,
    lagr_idx=False,
    float_dtype=np.float32,
):
    """
    Extract fields from bit-packed PIDs.  The PID (really, the 64-bit aux field)
    enocdes the particle ID, the Lagrangian index (and therefore position), the
    density, and the L2 tagged field.

    Parameters
    ----------
    packed: array-like of np.uint64, shape (N,)
        The bit-packed PID (i.e. the aux field)

    box: float, optional
        The box size, needed only for ``lagr_pos``

    ppd: int, optional
        The particles-per-dimension, needed only for ``lagr_pos``

    pid: bool, optional
        Whether to unpack and return the unique particle ID.

        Loaded as a ``np.int64`` array of shape `(N,)`.

    lagr_idx: bool, optional
        Whether to unpack and return the Lagrangian index, which is the `(i,j,k)`
        integer coordinates of the particle in the cubic lattice used in Abacus
        pre-initial conditions. The ``lagr_pos`` field will automatically convert
        this index to a position.

        Loaded as a ``np.int16`` array of shape `(N,3)`.

    lagr_pos: bool, optional
        Whether to unpack and return the Lagrangian position of the particles,
        based on the Lagrangian index (``lagr_idx``).

        Loaded as array of type ``dtype`` and shape `(N,3)`.

    tagged: bool, optional
        Whether to unpack and return the CompaSO L2 tagged bit of the particles---
        whether the particle was ever part of an L2 group (i.e. halo core).

        Loaded as a ``np.bool8`` array of shape `(N,)`.

    density: bool, optional
        Whether to unpack and return the local density estimate, in units of mean
        density.

        Loaded as a array of type ``dtype`` and shape `(N,)`.

    float_dtype: np.dtype, optional
        The dtype in which to store float arrays. Default: ``np.float32``

    Returns
    -------
    unpacked_arrays: dict of ndarray
        A dictionary of all fields that were unpacked
    """
    packed = np.asanyarray(packed, dtype=np.uint64)

    if lagr_pos is not False:
        if box is None:
            raise ValueError('Must supply `box` if requesting `lagr_pos`')
        if ppd is None:
            raise ValueError('Must supply `ppd` if requesting `lagr_pos`')

    N = len(packed)

    if ppd is not None:
        if not np.isclose(ppd, int(round(ppd))):
            raise ValueError(f'ppd "{ppd}" not valid int?')
        ppd = int(round(ppd))
    else:
        ppd = 1

    if box is None:
        box = float_dtype(1.0)

    arr = {}
    if pid is True:
        arr['pid'] = np.empty(N, dtype=np.int64)
    if lagr_pos is True:
        arr['lagr_pos'] = np.empty((N, 3), dtype=float_dtype)
    if lagr_idx is True:
        arr['lagr_idx'] = np.empty((N, 3), dtype=np.int16)
    if tagged is True:
        arr['tagged'] = np.empty(N, dtype=np.uint8)
    if density is True:
        arr['density'] = np.empty(N, dtype=float_dtype)

    _unpack_pids(packed, box, ppd, float_dtype=float_dtype, **arr)

    return arr


def empty_bitpacked_arrays(N, unpack_bits, float_dtype=np.float32):
    """
    Create empty arrays for bit-packed fields.

    Parameters
    ----------
    N: int
        The number of particles

    unpack_bits: list or bool
        The fields to unpack. If True, all fields are unpacked. If False,
        just the pid field is unpacked.

    float_dtype: np.dtype, optional
        The dtype in which to store float arrays. Default: ``np.float32``

    Returns
    -------
    empty_arrays: dict of ndarray
        A dictionary of empty arrays for all fields that can be unpacked
    """

    if type(unpack_bits) is str:
        unpack_bits = [unpack_bits]

    if unpack_bits is True:
        unpack_bits = PID_FIELDS
    elif unpack_bits is False:
        unpack_bits = ['pid']

    arr = {}
    if 'pid' in unpack_bits:
        arr['pid'] = np.empty(N, dtype=np.int64)
    if 'lagr_pos' in unpack_bits:
        arr['lagr_pos'] = np.empty((N, 3), dtype=float_dtype)
    if 'lagr_idx' in unpack_bits:
        arr['lagr_idx'] = np.empty((N, 3), dtype=np.int16)
    if 'tagged' in unpack_bits:
        arr['tagged'] = np.empty(N, dtype=np.uint8)
    if 'density' in unpack_bits:
        arr['density'] = np.empty(N, dtype=float_dtype)
    if 'packedpid' in unpack_bits:
        arr['packedpid'] = np.empty(N, dtype=np.uint64)

    return arr


@nb.njit
def _unpack_pids(
    packed,
    box,
    ppd,
    pid=None,
    lagr_pos=None,
    tagged=None,
    density=None,
    lagr_idx=None,
    float_dtype=np.float32,
):
    """Helper to extract the Lagrangian position, tagged info, and
    density from the ids of the particles
    """

    N = len(packed)
    box = np.float64(box)
    inv_ppd = float_dtype(box / ppd)
    half = float_dtype(box / 2)

    for i in range(N):
        if lagr_idx is not None:
            lagr_idx[i, 0] = packed[i] & AUXXPID
            lagr_idx[i, 1] = (packed[i] & AUXYPID) >> np.uint64(16)
            lagr_idx[i, 2] = (packed[i] & AUXZPID) >> np.uint64(32)

        if lagr_pos is not None:
            lagr_pos[i, 0] = (packed[i] & AUXXPID) * inv_ppd - half
            lagr_pos[i, 1] = ((packed[i] & AUXYPID) >> np.uint64(16)) * inv_ppd - half
            lagr_pos[i, 2] = ((packed[i] & AUXZPID) >> np.uint64(32)) * inv_ppd - half

        if tagged is not None:
            tagged[i] = (packed[i] >> AUXTAGGED) & np.uint64(1)

        if density is not None:
            density[i] = (
                (packed[i] & AUXDENS) >> ZERODEN
            ) ** 2  # max is 2**10, squaring gets to 2**20

        if pid is not None:
            pid[i] = packed[i] & AUXPID
