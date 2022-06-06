'''
A collection of routines related to various Abacus bitpacked
formats, like RVint and encoding of information in the PIDs.

Most users will not use this module directly, but will instead use
:mod:`abacusnbody.data.compaso_halo_catalog` or
:func:`abacusnbody.data.read_abacus.read_asdf`.
'''

import numpy as np
import numba as nb

__all__ = ['unpack_rvint', 'unpack_pids']

# Constants
AUXDENS = 0x07fe000000000000
ZERODEN = 49 #The density bits are 49-58. 

AUXXPID = 0x7fff #bits 0-14
AUXYPID = 0x7fff0000 #bits 16-30
AUXZPID = 0x7fff00000000 #bits 32-46
AUXPID = AUXXPID | AUXYPID | AUXZPID # all of the above bits
AUXTAGGED = 48 # tagged bit is 48

# The names of the bit-packed PID fields that the user can request
PID_FIELDS=['pid', 'lagr_pos', 'tagged', 'density', 'lagr_idx']


def unpack_rvint(intdata, boxsize, float_dtype=np.float32, posout=None, velout=None):
    '''
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
    
    '''
    intdata = intdata.reshape(-1)
    assert(intdata.dtype == np.int32)
    N = len(intdata)
    
    if posout is None:
        _posout = np.empty(N, dtype=float_dtype)
    elif posout is False:
        _posout = np.empty(0)
    else:
        _posout = posout.view()
        _posout.shape = -1  # enforces no copy
    
    if velout is None:
        _velout = np.empty(N, dtype=float_dtype)
    elif velout is False:
        _velout = np.empty(0)
    else:
        _velout = velout.view()
        _velout.shape = -1  # enforces no copy
            
    _unpack_rvint(intdata, boxsize, _posout, _velout)
    
    ret = []
    if posout is None:
        ret += [_posout.reshape(N//3,3)]
    elif posout is False:
        ret += [0]
    else:
        ret += [N//3]
        
    if velout is None:
        ret += [_velout.reshape(N//3,3)]
    elif velout is False:
        ret += [0]
    else:
        ret += [N//3]
        
    return tuple(ret)


@nb.njit
def _unpack_rvint(intdata, boxsize, posout, velout):
    '''Helper for unpack_rvint
    '''

    N = len(intdata)
    posscale = boxsize*(2.**-12.)/1e6
    velscale = 6000./2048
    pmask = np.int32(0xfffff000)
    vmask = np.int32(0xfff)
    
    lenp = len(posout)
    lenv = len(velout)
    
    for i in range(N):
        if lenp > 0:
            posout[i] = (intdata[i]&pmask)*posscale
        if lenv > 0:
            velout[i] = ((intdata[i]&vmask) - 2048)*velscale
            

def unpack_pids(packed, box=None, ppd=None, pid=False, lagr_pos=False, tagged=False, density=False, lagr_idx=False, float_dtype=np.float32):
    '''
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
    '''
    packed = np.asanyarray(packed, dtype=np.uint64)
    
    if lagr_pos is not False:
        if box is None:
            raise ValueError('Must supply `box` if requesting `lagr_pos`')
        if ppd is None:
            raise ValueError('Must supply `ppd` if requesting `lagr_pos`')
            
    N = len(packed)
    if not np.isclose(ppd, int(round(ppd))):
        raise ValueError(f'ppd "{ppd}" not valid int?')
    ppd = int(round(ppd))
    
    arr = {}
    if pid is True:
        arr['pid'] = np.empty(N, dtype=np.int64)
    if lagr_pos is True:
        arr['lagr_pos'] = np.empty((N,3), dtype=float_dtype)
    if lagr_idx is True:
        arr['lagr_idx'] = np.empty((N,3), dtype=np.int16)
    if tagged is True:
        arr['tagged'] = np.empty(N, dtype=np.bool8)
    if density is True:
        arr['density'] = np.empty(N, dtype=float_dtype)
        
    _unpack_pids(packed, box, ppd, float_dtype=float_dtype, **arr)
        
    return arr
    

@nb.njit
def _unpack_pids(packed, box, ppd, pid=None, lagr_pos=None, tagged=None, density=None, lagr_idx=None, float_dtype=np.float32):
    '''Helper to extract the Lagrangian position, tagged info, and
    density from the ids of the particles
    '''

    N = len(packed)
    box = np.float64(box)
    inv_ppd = float_dtype(box/ppd)
    half = float_dtype(box/2)

    for i in range(N):
        if lagr_idx is not None:
            lagr_idx[i,0] =  packed[i] & AUXXPID
            lagr_idx[i,1] = (packed[i] & AUXYPID) >> 16
            lagr_idx[i,2] = (packed[i] & AUXZPID) >> 32
            
        if lagr_pos is not None:
            lagr_pos[i,0] = (packed[i] & AUXXPID)*inv_ppd - half
            lagr_pos[i,1] = ((packed[i] & AUXYPID) >> 16)*inv_ppd - half
            lagr_pos[i,2] = ((packed[i] & AUXZPID) >> 32)*inv_ppd - half

        if tagged is not None:
            tagged[i] = (packed[i] >> AUXTAGGED) & 1

        if density is not None:
            density[i] = ((packed[i] & AUXDENS) >> ZERODEN)**2  # max is 2**10, squaring gets to 2**20

        if pid is not None:
            pid[i] = packed[i] & AUXPID
