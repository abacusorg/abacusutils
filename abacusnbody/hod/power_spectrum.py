import gc

import numba
import numpy as np
from numba import njit
#from np.fft import fftfreq, fftn, ifftn
from scipy.fft import fftfreq, fftn
from scipy.special import legendre


def get_k_mu_box(L_hMpc, n_xy, n_z):
    """
    Compute the size of the k vector and mu for each mode. Assumes z direction is LOS
    """
    # cell width in (x,y) directions (Mpc/h)
    d_xy = L_hMpc / n_xy
    k_xy = (fftfreq(n_xy, d=d_xy) * 2. * np.pi).astype(np.float32)

    # cell width in z direction (Mpc/h)
    d_z = L_hMpc / n_z
    k_z = (fftfreq(n_z, d=d_z) * 2. * np.pi).astype(np.float32)
    
    # h/Mpc
    x = k_xy[:, np.newaxis, np.newaxis]
    y = k_xy[np.newaxis, :, np.newaxis]
    z = k_z[np.newaxis, np.newaxis, :]
    k_box = np.sqrt(x**2 + y**2 + z**2)

    # construct mu in two steps, without NaN warnings
    mu_box = z/np.ones_like(k_box)
    mu_box[k_box > 0.] /= k_box[k_box > 0.]
    mu_box[k_box == 0.] = 0.

    # I believe the definition is that and it allows you to count all modes (agrees with nbodykit)
    mu_box = np.abs(mu_box)
    print("k box, mu box", k_box.dtype, mu_box.dtype)
    return k_box, mu_box

@numba.vectorize
def rightwrap(x, L):
    if x >= L:
        return x - L
    return x


@njit(nogil=True)
def numba_cic_3D(positions, density, boxsize, weights=np.empty(0)):
    """
    Compute density using the cloud-in-cell algorithm. Assumes cubic box
    """
    gx = np.uint32(density.shape[0])
    gy = np.uint32(density.shape[1])
    gz = np.uint32(density.shape[2])
    threeD = gz != 1
    W = 1.
    Nw = len(weights)
    for n in range(len(positions)):
        # broadcast scalar weights
        if Nw == 1:
            W = weights[0]
        elif Nw > 1:
            W = weights[n]

        # convert to a position in the grid
        px = (positions[n,0]/boxsize)*gx # used to say boxsize+0.5
        py = (positions[n,1]/boxsize)*gy # used to say boxsize+0.5
        if threeD:
            pz = (positions[n,2]/boxsize)*gz # used to say boxsize+0.5

        # round to nearest cell center
        ix = np.int32(round(px))
        iy = np.int32(round(py))
        if threeD:
            iz = np.int32(round(pz))

        # calculate distance to cell center
        dx = ix - px
        dy = iy - py
        if threeD:
            dz = iz - pz

        # find the tsc weights for each dimension
        wx = 1. - np.abs(dx)
        if dx > 0.: # on the right of the center ( < )
            wxm1 = dx
            wxp1 = 0.
        else: # on the left of the center
            wxp1 = -dx
            wxm1 = 0.
        wy = 1. - np.abs(dy)
        if dy > 0.:
            wym1 = dy
            wyp1 = 0.
        else:
            wyp1 = -dy
            wym1 = 0.
        if threeD:
            wz = 1. - np.abs(dz)
            if dz > 0.:
                wzm1 = dz
                wzp1 = 0.
            else:
                wzp1 = -dz
                wzm1 = 0.
        else:
            wz = 1.

        # find the wrapped x,y,z grid locations of the points we need to change
        # negative indices will be automatically wrapped
        ixm1 = rightwrap(ix - 1, gx)
        ixw  = rightwrap(ix    , gx)
        ixp1 = rightwrap(ix + 1, gx)
        iym1 = rightwrap(iy - 1, gy)
        iyw  = rightwrap(iy    , gy)
        iyp1 = rightwrap(iy + 1, gy)
        if threeD:
            izm1 = rightwrap(iz - 1, gz)
            izw  = rightwrap(iz    , gz)
            izp1 = rightwrap(iz + 1, gz)
        else:
            izw = np.uint32(0)

        # change the 9 or 27 cells that the cloud touches
        density[ixm1, iym1, izw ] += wxm1*wym1*wz  *W
        density[ixm1, iyw , izw ] += wxm1*wy  *wz  *W
        density[ixm1, iyp1, izw ] += wxm1*wyp1*wz  *W
        density[ixw , iym1, izw ] += wx  *wym1*wz  *W
        density[ixw , iyw , izw ] += wx  *wy  *wz  *W
        density[ixw , iyp1, izw ] += wx  *wyp1*wz  *W
        density[ixp1, iym1, izw ] += wxp1*wym1*wz  *W
        density[ixp1, iyw , izw ] += wxp1*wy  *wz  *W
        density[ixp1, iyp1, izw ] += wxp1*wyp1*wz  *W

        if threeD:
            density[ixm1, iym1, izm1] += wxm1*wym1*wzm1*W
            density[ixm1, iym1, izp1] += wxm1*wym1*wzp1*W

            density[ixm1, iyw , izm1] += wxm1*wy  *wzm1*W
            density[ixm1, iyw , izp1] += wxm1*wy  *wzp1*W

            density[ixm1, iyp1, izm1] += wxm1*wyp1*wzm1*W
            density[ixm1, iyp1, izp1] += wxm1*wyp1*wzp1*W

            density[ixw , iym1, izm1] += wx  *wym1*wzm1*W
            density[ixw , iym1, izp1] += wx  *wym1*wzp1*W

            density[ixw , iyw , izm1] += wx  *wy  *wzm1*W
            density[ixw , iyw , izp1] += wx  *wy  *wzp1*W

            density[ixw , iyp1, izm1] += wx  *wyp1*wzm1*W
            density[ixw , iyp1, izp1] += wx  *wyp1*wzp1*W

            density[ixp1, iym1, izm1] += wxp1*wym1*wzm1*W
            density[ixp1, iym1, izp1] += wxp1*wym1*wzp1*W

            density[ixp1, iyw , izm1] += wxp1*wy  *wzm1*W
            density[ixp1, iyw , izp1] += wxp1*wy  *wzp1*W

            density[ixp1, iyp1, izm1] += wxp1*wyp1*wzm1*W
            density[ixp1, iyp1, izp1] += wxp1*wyp1*wzp1*W

@njit(nogil=True)
def numba_tsc_3D(positions, density, boxsize, weights=np.empty(0)):
    """
    Compute density using the triangle-shape-cloud algorithm. Assumes cubic box
    """
    gx = np.uint32(density.shape[0])
    gy = np.uint32(density.shape[1])
    gz = np.uint32(density.shape[2])
    threeD = gz != 1
    W = 1.
    Nw = len(weights)
    for n in range(len(positions)):
        # broadcast scalar weights
        if Nw == 1:
            W = weights[0]
        elif Nw > 1:
            W = weights[n]

        # convert to a position in the grid
        px = (positions[n,0]/boxsize)*gx # used to say boxsize+0.5
        py = (positions[n,1]/boxsize)*gy # used to say boxsize+0.5
        if threeD:
            pz = (positions[n,2]/boxsize)*gz # used to say boxsize+0.5

        # round to nearest cell center
        ix = np.int32(round(px))
        iy = np.int32(round(py))
        if threeD:
            iz = np.int32(round(pz))

        # calculate distance to cell center
        dx = ix - px
        dy = iy - py
        if threeD:
            dz = iz - pz

        # find the tsc weights for each dimension
        wx = .75 - dx**2
        wxm1 = .5*(.5 + dx)**2 # og not 1.5 cause wrt to adjacent cell
        wxp1 = .5*(.5 - dx)**2
        wy = .75 - dy**2
        wym1 = .5*(.5 + dy)**2
        wyp1 = .5*(.5 - dy)**2
        if threeD:
            wz = .75 - dz**2
            wzm1 = .5*(.5 + dz)**2
            wzp1 = .5*(.5 - dz)**2
        else:
            wz = 1.

        # find the wrapped x,y,z grid locations of the points we need to change
        # negative indices will be automatically wrapped
        ixm1 = rightwrap(ix - 1, gx)
        ixw  = rightwrap(ix    , gx)
        ixp1 = rightwrap(ix + 1, gx)
        iym1 = rightwrap(iy - 1, gy)
        iyw  = rightwrap(iy    , gy)
        iyp1 = rightwrap(iy + 1, gy)
        if threeD:
            izm1 = rightwrap(iz - 1, gz)
            izw  = rightwrap(iz    , gz)
            izp1 = rightwrap(iz + 1, gz)
        else:
            izw = np.uint32(0)

        # change the 9 or 27 cells that the cloud touches
        density[ixm1, iym1, izw ] += wxm1*wym1*wz  *W
        density[ixm1, iyw , izw ] += wxm1*wy  *wz  *W
        density[ixm1, iyp1, izw ] += wxm1*wyp1*wz  *W
        density[ixw , iym1, izw ] += wx  *wym1*wz  *W
        density[ixw , iyw , izw ] += wx  *wy  *wz  *W
        density[ixw , iyp1, izw ] += wx  *wyp1*wz  *W
        density[ixp1, iym1, izw ] += wxp1*wym1*wz  *W
        density[ixp1, iyw , izw ] += wxp1*wy  *wz  *W
        density[ixp1, iyp1, izw ] += wxp1*wyp1*wz  *W

        if threeD:
            density[ixm1, iym1, izm1] += wxm1*wym1*wzm1*W
            density[ixm1, iym1, izp1] += wxm1*wym1*wzp1*W

            density[ixm1, iyw , izm1] += wxm1*wy  *wzm1*W
            density[ixm1, iyw , izp1] += wxm1*wy  *wzp1*W

            density[ixm1, iyp1, izm1] += wxm1*wyp1*wzm1*W
            density[ixm1, iyp1, izp1] += wxm1*wyp1*wzp1*W

            density[ixw , iym1, izm1] += wx  *wym1*wzm1*W
            density[ixw , iym1, izp1] += wx  *wym1*wzp1*W

            density[ixw , iyw , izm1] += wx  *wy  *wzm1*W
            density[ixw , iyw , izp1] += wx  *wy  *wzp1*W

            density[ixw , iyp1, izm1] += wx  *wyp1*wzm1*W
            density[ixw , iyp1, izp1] += wx  *wyp1*wzp1*W

            density[ixp1, iym1, izm1] += wxp1*wym1*wzm1*W
            density[ixp1, iym1, izp1] += wxp1*wym1*wzp1*W

            density[ixp1, iyw , izm1] += wxp1*wy  *wzm1*W
            density[ixp1, iyw , izp1] += wxp1*wy  *wzp1*W

            density[ixp1, iyp1, izm1] += wxp1*wyp1*wzm1*W
            density[ixp1, iyp1, izp1] += wxp1*wyp1*wzp1*W
            

@njit(nogil=True, parallel=False)
def mean2d_numba_seq(tracks, bins, ranges, logk, weights=np.empty(0), dtype=np.float32):
    """
    Compute the mean number of modes per 2D bin.
    This implementation is 8-9 times faster than np.histogramdd and can be threaded (nogil!)
    """
    tracks = tracks.astype(dtype)
    ranges = ranges.astype(dtype)
    H = np.zeros((bins[0], bins[1]), dtype=dtype)
    N = np.zeros((bins[0], bins[1]), dtype=dtype)
    if logk:
        delta0 = 1./(np.log(ranges[0, 1]/ranges[0, 0]) / bins[0])
    else:
        delta0 = 1./((ranges[0, 1] - ranges[0, 0]) / bins[0])
    delta1 = 1./((ranges[1, 1] - ranges[1, 0]) / bins[1])
    Nw = len(weights)
    for t in range(tracks.shape[1]):
        if logk:
            i = np.log(tracks[0, t]/ranges[0, 0]) * delta0
        else:
            i = (tracks[0, t] - ranges[0, 0]) * delta0
        j = (tracks[1, t] - ranges[1, 0]) * delta1

        if 0. <= i < bins[0] and 0. <= j < bins[1]:
            N[int(i), int(j)] += 1.
            H[int(i), int(j)] += weights[t]
            
    for i in range(bins[0]):
        for j in range(bins[1]):
            if N[i, j] > 0.:
                H[i, j] /= N[i, j]
    return H, N

def get_k_mu_edges(L_hMpc, k_hMpc_max, n_k_bins, n_mu_bins, logk):

    # define k-binning (in 1/Mpc)
    if logk:
        # set minimum k to make sure we cover fundamental mode
        k_hMpc_min = (1.-1.e-4)*2.*np.pi/L_hMpc
        k_bin_edges = np.geomspace(k_hMpc_min, k_hMpc_max, n_k_bins+1)
    else:
        #dk = 2.*np.pi/L_hMpc
        #k_bin_edges = np.linspace(k_hMpc_min, k_hMpc_max, n_k_bins+1)
        k_bin_edges = np.linspace(0., k_hMpc_max, n_k_bins+1)
        #k_bin_edges = np.arange(0., n_k_bins+1) * dk
    
    # define mu-binning
    mu_bin_edges = np.linspace(0., 1., n_mu_bins + 1)
    # ask Lehman; not doing this misses the +/- 1 mu modes (i.e. I want the rightmost edge of the mu bins to be inclusive)
    mu_bin_edges[-1] += 1.e-6 
    #k_bin_edges[1:] += 1.e-6 # includes the 2pi/L modes in the first bin, though I think they shouldn't be there...
    
    return k_bin_edges, mu_bin_edges

def get_k_mu_box_edges(L_hMpc, n_xy, n_z, n_k_bins, n_mu_bins, k_hMpc_max, logk):
    """
    Compute the size of the k vector and mu for each mode and also bin edges for both. Assumes z direction is along LOS
    """
    
    # this stores *all* Fourier wavenumbers in the box (no binning)
    k_box, mu_box = get_k_mu_box(L_hMpc, n_xy, n_z)
    k_box = k_box.flatten()
    mu_box = mu_box.flatten()
    
    # define k and mu-binning
    k_bin_edges, mu_bin_edges = get_k_mu_edges(L_hMpc, k_hMpc_max, n_k_bins, n_mu_bins, logk)
    return k_box, mu_box, k_bin_edges, mu_bin_edges

def calc_pk3d(field_fft, L_hMpc, k_box, mu_box, k_bin_edges, mu_bin_edges, logk, field2_fft=None, poles=[]):
    """
    Calculate the P3D for a given field (in h/Mpc units). Answer returned in (Mpc/h)^3 units
    """
    
    # get raw power
    if field2_fft is not None:
        raw_p3d = (np.conj(field_fft)*field2_fft).real.flatten()
    else:
        raw_p3d = (np.abs(field_fft)**2).flatten()
    del field_fft; gc.collect()
    
    # for the histograming
    ranges = ((k_bin_edges[0], k_bin_edges[-1]),(mu_bin_edges[0], mu_bin_edges[-1])) 
    nbins2d = (len(k_bin_edges)-1, len(mu_bin_edges)-1)
    nbins2d = np.asarray(nbins2d).astype(np.int64)
    ranges = np.asarray(ranges).astype(np.float64)

    # power spectrum
    binned_p3d, N3d = mean2d_numba_seq(np.array([k_box, mu_box]), bins=nbins2d, ranges=ranges, logk=logk, weights=raw_p3d)
    
    # if poles is not empty, then compute P_ell for mu_box
    binned_poles = []
    Npoles = []
    if len(poles) > 0:
        # ask Lehman; not doing this misses the +/- 1 mu modes (i.e. I want the rightmost edge of the mu bins to be inclusive)
        ranges = ((k_bin_edges[0], k_bin_edges[-1]), (0., 1.+1.e-6)) # not doing this misses the +/- 1 mu modes in the first few bins
        nbins2d = (len(k_bin_edges)-1, 1)
        nbins2d = np.asarray(nbins2d).astype(np.int64)
        ranges = np.asarray(ranges).astype(np.float64)
        for i in range(len(poles)):
            Ln = legendre(poles[i])
            binned_pole, Npole = mean2d_numba_seq(np.array([k_box, mu_box]), bins=nbins2d, ranges=ranges, logk=logk, weights=raw_p3d*Ln(mu_box)*(2.*poles[i]+1.)) # ask Lehman (I think the equation is (2 ell + 1)/2 but for some reason I don't need the division by 2)
            binned_poles.append(binned_pole * L_hMpc**3)
        Npoles.append(Npole)
        Npoles = np.array(Npoles)
        binned_poles = np.array(binned_poles)
            
    # quantity above is dimensionless, multiply by box size (in Mpc/h)
    p3d_hMpc = binned_p3d * L_hMpc**3
    return p3d_hMpc, N3d, binned_poles, Npoles

def get_field(pos, lbox, num_cells, paste, w=None, d=0.):
    # check if weights are requested
    if w is None:
        w = np.empty(0)
    else:
        assert pos.shape[0] == len(w)
    pos = pos.astype(np.float32)
    field = np.zeros((num_cells, num_cells, num_cells), dtype=np.float32)
    if paste == 'TSC':
        if d != 0.:
            numba_tsc_3D(pos + np.float32(d), field, lbox, weights=w)
        else:
            numba_tsc_3D(pos, field, lbox, weights=w)
    elif paste == 'CIC':
        if d != 0.:
            numba_cic_3D(pos + np.float32(d), field, lbox, weights=w)
        else:
            numba_cic_3D(pos, field, lbox, weights=w)
    if len(w) == 0: # in the zcv code the weights are already normalized, so don't normalize here
        field /= (pos.shape[0]/num_cells**3.) # same as passing "Value" to nbodykit (1+delta)(x) V(x)
        field -= 1. # leads to -1 in the complex field
    return field

def get_interlaced_field_fft(pos, field, lbox, num_cells, paste, w):

    # cell width
    d = lbox / num_cells

    # nyquist frequency
    kN = np.pi / d

    # natural wavemodes
    k = (fftfreq(num_cells, d=d) * 2. * np.pi).astype(np.float32) # h/Mpc

    # offset by half a cell
    field_shift = get_field(pos, lbox, num_cells, paste, w, d=0.5*d)
    print("shift", field_shift.dtype, pos.dtype)
    del pos, w; gc.collect()

    # fourier transform shifted field and sum them up
    #field_fft = fftn(field) / field.size
    #field_shift_fft = fftn(field_shift) / field.size
    field_fft = np.zeros((len(k), len(k), len(k)), dtype=np.complex64)
    field_fft[:, :, :] = fftn(field) + fftn(field_shift) * \
                         np.exp(0.5 * 1j * (k[:, np.newaxis, np.newaxis] + \
                                            k[np.newaxis, :, np.newaxis] + \
                                            k[np.newaxis, np.newaxis, :]) *d)
    field_fft *= 0.5 / field.size
    print("field fft", field_fft.dtype)
    
    # inverse fourier transform
    #field_fft *= field.size
    #field = ifftn(field_fft) # we work in fourier
    return field_fft
    
def get_field_fft(pos, lbox, num_cells, paste, w, W, compensated, interlaced):

    # get field in real space
    field = get_field(pos, lbox, num_cells, paste, w)
    print("field, pos", field.dtype, pos.dtype)
    if interlaced:
        # get interlaced field
        field_fft = get_interlaced_field_fft(pos, field, lbox, num_cells, paste, w)
    else:
        # get Fourier modes from skewers grid
        field_fft = fftn(field) / field.size
    # get rid of pos, field
    del pos, w, field; gc.collect()

    # apply compensation filter
    if compensated:
        assert W is not None
        field_fft /= (W[:, np.newaxis, np.newaxis] * W[np.newaxis, :, np.newaxis] * W[np.newaxis, np.newaxis, :])
    return field_fft
    
def get_W_compensated(lbox, num_cells, paste, interlaced):
    """ 
    Compute the TSC/CIC kernel convolution for a given set of wavenumbers.
    """
    
    # cell width
    d = lbox / num_cells

    # nyquist frequency
    kN = np.pi / d

    # natural wavemodes
    k = (fftfreq(num_cells, d=d) * 2. * np.pi).astype(np.float32) # h/Mpc
    
    # apply deconvolution
    if interlaced:
        if paste == 'TSC':
            p = 3.
        elif paste == 'CIC':
            p = 2.
        W = np.sinc(0.5*k/kN)**p # sinc def
    else: # first order correction of interlacing (aka aliasing)
        s = np.sin(0.5 * np.pi * k/kN)**2
        if paste == 'TSC':
            W = (1 - s + 2./15 * s**2) ** 0.5
        elif paste == 'CIC':
            W = (1 - 2./3 * s) ** 0.5 
        del s
    print(W.astype)
    return W

def calc_power(x1, y1, z1, nbins_k, nbins_mu, k_hMpc_max, logk, lbox, paste, num_cells, compensated, interlaced, w = None, x2 = None, y2 = None, z2 = None, w2 = None, poles=[]):
    """
    Compute the 3D power spectrum given particle positions by first painting them on a cubic mesh and then applying the fourier transforms and mode counting.
    """

    # get the window function
    if compensated:
        W = get_W_compensated(lbox, num_cells, paste, interlaced)
    else:
        W = None

    # express more conveniently
    pos = np.zeros((len(x1), 3), dtype=np.float32)
    pos[:, 0] = x1; pos[:, 1] = y1; pos[:, 2] = z1
    del x1, y1, z1; gc.collect()

    # convert to fourier space
    field_fft = get_field_fft(pos, lbox, num_cells, paste, w, W, compensated, interlaced)
    del pos; gc.collect()

    # if second field provided
    if x2 is not None:
        assert (y2 is not None) and (z2 is not None)
        # assemble the positions and compute density field
        pos2 = np.zeros((len(x2), 3), dtype=np.float32)
        pos2[:, 0] = x2; pos2[:, 1] = y2; pos2[:, 2] = z2
        del x2, y2, z2; gc.collect()

        # convert to fourier space
        field2_fft = get_field_fft(pos2, lbox, num_cells, paste, w2, W, compensated, interlaced)
        del pos2; gc.collect()
    else:
        field2_fft = None
        
    # calculate power spectrum
    n_perp = n_los = num_cells # cubic box
    k_box, mu_box, k_bin_edges, mu_bin_edges = get_k_mu_box_edges(lbox, n_perp, n_los, nbins_k, nbins_mu, k_hMpc_max, logk)
    pk3d, N3d, binned_poles, Npoles = calc_pk3d(field_fft, lbox, k_box, mu_box, k_bin_edges, mu_bin_edges, logk, field2_fft=field2_fft, poles=poles)

    # define bin centers
    k_binc = (k_bin_edges[1:] + k_bin_edges[:-1])*.5
    mu_binc = (mu_bin_edges[1:] + mu_bin_edges[:-1])*.5
    return k_binc, mu_binc, pk3d, N3d, binned_poles, Npoles
