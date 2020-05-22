'''
A collection of routines related to various Abacus bitpacked
formats, like RVint and encoding of information in the PIDs.
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


@nb.njit
def unpack_pids(packed, box, ppd, float_dtype=np.float32):
    '''extract the Lagrangian position, tagged info, and
    density from the ids of the particles
    '''

    N = len(packed)
    
    justpid = np.empty(N, dtype=np.int64)
    lagr_pos = np.empty((N,3), dtype=float_dtype)
    tagged = np.empty(N, dtype=np.bool8)
    density = np.empty(N, dtype=float_dtype)

    for i in range(N):
        lagr_pos[i,0] = ((packed[i] & AUXXPID)/ppd - 0.5)*box
        lagr_pos[i,1] = (((packed[i] & AUXYPID) >> 16)/ppd - 0.5)*box
        lagr_pos[i,2] = (((packed[i] & AUXZPID) >> 32)/ppd - 0.5)*box

        tagged[i] = (packed[i] >> AUXTAGGED) & 1

        density[i] = ((packed[i] & AUXDENS) >> ZERODEN)**2  # max is 2**10, squaring gets to 2**20

        justpid[i] = packed[i] & AUXPID
    
    return justpid, lagr_pos, tagged, density
