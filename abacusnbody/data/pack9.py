'''
Unpack pack9 particle data, which encodes the pos + vel in 9 bytes, and then the
PID + aux in another 8 bytes in a separate file.  The 9-byte part is handled by this
module; the 8-byte (PID) part is handled by ``bitpacked.unpack_pids()``.

Most users will not use this module directly, but will instead use the
:func:`abacusnbody.data.read_abacus.read_asdf` function.
'''

import numpy as np
import numba as nb

__all__ = ['unpack_pack9']

def unpack_pack9(data, boxsize, velzspace_to_kms, float_dtype=np.float32, posout=None, velout=None):
    data = np.asanyarray(data, dtype=np.ubyte)
    Nmax = len(data)  # some pack9s will be cell headers
    
    if posout is None:
        _posout = np.empty((Nmax,3), dtype=float_dtype)
    elif posout is False:
        _posout = None
    else:
        _posout = posout
    
    if velout is None:
        _velout = np.empty((Nmax,3), dtype=float_dtype)
    elif velout is False:
        _velout = None
    else:
        _velout = velout
        
    npart = _unpack_pack9(data, boxsize, velzspace_to_kms, _posout, _velout, float_dtype)
    
    ret = []
    if posout is None:
        ret += [_posout[:npart]]
    elif posout is False:
        ret += [0]
    else:
        ret += [npart]
        
    if velout is None:
        ret += [_velout[:npart]]
    elif velout is False:
        ret += [0]
    else:
        ret += [npart]
        
    return tuple(ret)

@nb.njit
def _unpack_pack9(data, boxsize, velzspace_to_kms, posout, velout, dtype):
    w = np.int64(0)  # written
    N = len(data)
    boxsize = dtype(boxsize)
    velzspace_to_kms = dtype(velzspace_to_kms)
    halfbox = boxsize/2
    
    # header state
    csize = dtype(np.nan)  # in user units
    vscale = dtype(np.nan)
    cellx = dtype(np.nan)
    celly = dtype(np.nan)
    cellz = dtype(np.nan)
    pscale = dtype(np.nan)
    
    # each coord is packed in 12 bits, which we will expand to 16 and store in shorts
    sh = np.empty(6, dtype=np.int16)
    
    dop = posout is not None
    dov = velout is not None
    
    for i in range(N):
        p9 = data[i]
        _expand_to_short(p9, sh)
        if p9[0] == np.ubyte(0xff):
            # new header!
            invcpd = dtype(1./(sh[1] + 2000))
            csize = boxsize*invcpd
            vscale = dtype((sh[2] + 2000)*0.0005)*invcpd*velzspace_to_kms
            cellx = dtype((sh[3] + 2000.5)*csize - halfbox)
            celly = dtype((sh[4] + 2000.5)*csize - halfbox)
            cellz = dtype((sh[5] + 2000.5)*csize - halfbox)
            pscale = dtype(0.0005*csize)
            #print(f'invcpd ({invcpd}) vscale ({vscale}) cellx ({cellx}) celly ({celly}) cellz ({cellz}), velz ({velzspace_to_kms})')
        else:
            #print(sh)
            # particle
            #assert not np.isnan(csize)  # valid header
            if dop:
                posout[w,0] = sh[0]*pscale + cellx
                posout[w,1] = sh[1]*pscale + celly
                posout[w,2] = sh[2]*pscale + cellz
            if dov:
                velout[w,0] = sh[3]*vscale
                velout[w,1] = sh[4]*vscale
                velout[w,2] = sh[5]*vscale
            w += 1
    
    return w


@nb.njit
def _expand_to_short(c, s):
    # inflate 9 chars to 6 shorts
    # it seems like numba promotes all bitwise operations to 64-bit
    # which is not ideal for performance, but should be safe
    s[0] = (c[1] & 0x0f) | (c[0] << 4)
    s[1] = ((c[1] & 0xf0) << 4) | c[2]
    s[2] = (c[4] & 0x0f) | (c[3] << 4)
    s[3] = ((c[4] & 0xf0) << 4) | c[5]
    s[4] = (c[7] & 0x0f) | (c[6] << 4)
    s[5] = ((c[7] & 0xf0) << 4) | c[8]
    
    for i in range(6):
        s[i] -= 2048
