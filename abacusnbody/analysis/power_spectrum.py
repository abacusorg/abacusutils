"""
tuks poles and rfft and in the other functions too and get rid of k_box and mu box for most -- only exception is expand poles
and also make the prepare steps easier (and for zenbu Xi too)
"""
import gc

import numba
import numpy as np
import numba as nb
from numba import njit
#from np.fft import fftfreq, fftn, ifftn
from scipy.fft import fftfreq, fftn, rfftfreq, rfftn
from scipy.special import legendre

from .tsc import tsc_parallel


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
def numba_cic_3D(positions, density, boxsize, weights=None):
    """
    Compute density using the cloud-in-cell algorithm. Assumes cubic box
    """
    gx = np.uint32(density.shape[0])
    gy = np.uint32(density.shape[1])
    gz = np.uint32(density.shape[2])
    threeD = gz != 1
    W = 1.
    have_W = weights is not None

    for n in range(len(positions)):
        if have_W:
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


@njit
def factorial_slow(x):
    """ 
    Computes factorial assuming x integer
    """
    n = 1
    for i in range(2, x+1):
        n *= i
    return n

LOOKUP_TABLE = np.array([
    1, 1, 2, 6, 24, 120, 720, 5040, 40320,
    362880, 3628800, 39916800, 479001600,
    6227020800, 87178291200, 1307674368000,
    20922789888000, 355687428096000, 6402373705728000,
    121645100408832000, 2432902008176640000], dtype='int64')

@nb.jit
def factorial(n):
    """ 
    Computes factorial assuming n integer
    """
    if n > 20:
        raise ValueError
    return LOOKUP_TABLE[n]

@njit
def n_choose_k(n, k):
    """ 
    Computes binomial coefficient assuming n, k integers
    """
    return factorial(n)//(factorial(k)*factorial(n-k))

@njit
def P_n(x, n, dtype=np.float32):
    """ 
    Computes Legendre polynomial for some squared x quantity (maximum tested n is 10)
    """
    sum = dtype(0.)
    for k in range(n//2+1):
        factor = dtype(n_choose_k(n, k) * n_choose_k(2*n - 2*k, n))
        if k % 2 == 0:
            sum += factor*x**(dtype(0.5*(n-2*k)))
        else:
            sum -= factor*x**(dtype(0.5*(n-2*k)))
    return dtype(0.5**n)*sum

            
@numba.njit(parallel=True, fastmath=True)
def bin_kmu(n1d, L, kedges, Nmu, weights, poles=np.array([]), dtype=np.float32):
    '''
    Count modes in (k,mu) bins for a 3D rfft mesh of shape (n1d, n1d, n1d//2+1).

    The k and mu values are constructed on the fly.  We use the monotonicity of
    k and mu with respect to kz to accelerate the bin search.

    The main opportunity we're not taking advantage of here is an eightfold
    symmetry in k-space. One just has to take care to single count the boundary
    modes and double count the others, as one does with the rfft symmetry.
    '''

    kzlen = n1d//2 + 1
    Nk = len(kedges) - 1

    kedges2 = ((kedges/(2*np.pi/L))**2).astype(dtype)
    muedges2 = (np.linspace(0., 1., Nmu+1)**2).astype(dtype)

    nthread = numba.get_num_threads()
    counts = np.zeros((nthread, Nk, Nmu), dtype=np.int64)
    weighted_counts = np.zeros((nthread, Nk, Nmu), dtype=dtype)
    poles = np.array(poles)
    
    if len(poles) > 0:
        Np = len(poles)
        counts_poles = np.zeros((nthread, Nk), dtype=np.int64)
        weighted_counts_poles = np.zeros((nthread, Np, Nk), dtype=dtype)
    else:
        Np = 1 # t
        counts_poles = np.zeros((nthread, Nk), dtype=np.int64) # t
        weighted_counts_poles = np.zeros((nthread, Np, Nk), dtype=dtype) # t
        #counts_poles = np.array([[]])
        #weighted_counts_poles = np.array([])
        
    # Loop over all k vectors
    for i in numba.prange(n1d):
        tid = numba.get_thread_id()
        i2 = i**2 if i <= n1d//2 else (n1d - i)**2
        for j in range(n1d):
            bk,bmu = 0,0
            j2 = j**2 if j <= n1d//2 else (n1d - j)**2
            for k in range(kzlen):
                kmag2 = dtype(i2 + j2 + k**2)
                if kmag2 > 0:
                    invkmag2 = kmag2**-1
                    mu2 = dtype(k**2) * invkmag2
                else:
                    mu2 = dtype(0.) # matches nbodykit

                if kmag2 < kedges2[0]:
                    continue

                if kmag2 >= kedges2[-1]:
                    break

                while kmag2 > kedges2[bk+1]:
                    bk += 1

                while mu2 > muedges2[bmu+1]:
                    bmu += 1

                counts[tid, bk, bmu] += 1 if k == 0 else 2
                weighted_counts[tid, bk, bmu] += weights[i, j, k] if k == 0 else dtype(2.)*weights[i, j, k]
                if Np > 0:
                    counts_poles[tid, bk] += 1 if k == 0 else 2
                    for ip in range(Np):
                        pole = poles[ip]
                        if pole == 0:
                            weighted_counts_poles[tid, ip, bk] += weights[i, j, k] if k == 0 else dtype(2.)*weights[i, j, k]
                        else:
                            pw = dtype(2*pole + 1)*P_n(mu2, pole)
                            weighted_counts_poles[tid, ip, bk] += weights[i, j, k]*pw if k == 0 else dtype(2.)*weights[i, j, k]*pw
    
    counts = counts.sum(axis=0)
    weighted_counts = weighted_counts.sum(axis=0)
    #if Np > 0: # tuks
    counts_poles = counts_poles.sum(axis=0)
    weighted_counts_poles = weighted_counts_poles.sum(axis=0)
        
    for i in range(Nk):
        if Np > 0:
            if counts_poles[i] != 0:
                for ip in range(Np):
                    weighted_counts_poles[ip, i] /= dtype(counts_poles[i])
        for j in range(Nmu):
            if counts[i, j] != 0:
                weighted_counts[i, j] /= dtype(counts[i, j])
    return weighted_counts, counts, weighted_counts_poles, counts_poles

def get_k_mu_edges(L_hMpc, k_hMpc_max, n_k_bins, n_mu_bins, logk):

    # define k-binning
    if logk:
        # set minimum k to make sure we cover fundamental mode
        k_hMpc_min = (1.-1.e-4)*2.*np.pi/L_hMpc
        k_bin_edges = np.geomspace(k_hMpc_min, k_hMpc_max, n_k_bins+1)
    else:
        k_bin_edges = np.linspace(0., k_hMpc_max, n_k_bins+1)

    # define mu-binning
    mu_bin_edges = np.linspace(0., 1., n_mu_bins + 1)
    # ask Lehman; not doing this misses the +/- 1 mu modes (i.e. I want the rightmost edge of the mu bins to be inclusive)
    mu_bin_edges[-1] += 1.e-6 # tuks maybe not necessary?

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

def project_3d_to_poles(k_bin_edges, k_box, mu_box, logk, raw_p3d, L_hMpc, poles):
    assert np.max(poles) <= 10, "numba implementation works up to ell = 10"
    
    binned_poles = []
    Npoles = []

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
    return binned_poles, Npoles

def calc_pk3d(field_fft, L_hMpc, k_box, mu_box, k_bin_edges, mu_bin_edges, logk, field2_fft=None, poles=[]):
    """
    Calculate the P3D for a given field (in h/Mpc units). Answer returned in (Mpc/h)^3 units
    """
    
    # get raw power
    if field2_fft is not None:
        raw_p3d = (np.conj(field_fft)*field2_fft).real
    else:
        raw_p3d = (np.abs(field_fft)**2)
    del field_fft
    gc.collect()

    # power spectrum
    nmesh = raw_p3d.shape[0] # tuks rfft
    Nmu = len(mu_bin_edges) - 1 # tuks, get rid of function args: logk, k_box, mu_box
    binned_p3d, N3d, binned_poles, Npoles = bin_kmu(nmesh, L_hMpc, k_bin_edges, Nmu, raw_p3d, poles)

    # quantity above is dimensionless, multiply by box size (in Mpc/h)
    p3d_hMpc = binned_p3d * L_hMpc**3
    if len(poles) > 0:
        binned_poles *= L_hMpc**3
    return p3d_hMpc, N3d, binned_poles, Npoles

# @profile
def get_field(pos, lbox, num_cells, paste, w=None, d=0.):
    # check if weights are requested
    if w is not None:
        assert pos.shape[0] == len(w)
    pos = pos.astype(np.float32)
    field = np.zeros((num_cells, num_cells, num_cells), dtype=np.float32)
    if paste == 'TSC':
        if d != 0.:
            # TODO: could add an offset parameter to tsc_parallel
            tsc_parallel(pos + np.float32(d), field, lbox, weights=w)
        else:
            tsc_parallel(pos, field, lbox, weights=w)
    elif paste == 'CIC':
        if d != 0.:
            numba_cic_3D(pos + np.float32(d), field, lbox, weights=w)
        else:
            numba_cic_3D(pos, field, lbox, weights=w)
    if w is None: # in the zcv code the weights are already normalized, so don't normalize here
        field /= (pos.shape[0]/num_cells**3.) # same as passing "Value" to nbodykit (1+delta)(x) V(x)
        field -= 1. # leads to -1 in the complex field
    return field

def get_interlaced_field_fft(pos, field, lbox, num_cells, paste, w):

    # cell width
    d = lbox / num_cells

    # nyquist frequency
    # kN = np.pi / d

    # natural wavemodes
    k = (fftfreq(num_cells, d=d) * 2. * np.pi).astype(np.float32) # h/Mpc

    # offset by half a cell
    field_shift = get_field(pos, lbox, num_cells, paste, w, d=0.5*d)
    print("shift", field_shift.dtype, pos.dtype)
    del pos, w
    gc.collect()

    # fourier transform shifted field and sum them up
    #field_fft = fftn(field) / field.size
    #field_shift_fft = fftn(field_shift) / field.size
    field_fft = np.zeros((len(k), len(k), len(k)), dtype=np.complex64)
    field_fft[:, :, :] = fftn(field, workers=-1) + fftn(field_shift, workers=-1) * \
                         np.exp(0.5 * 1j * (k[:, np.newaxis, np.newaxis] + \
                                            k[np.newaxis, :, np.newaxis] + \
                                            k[np.newaxis, np.newaxis, :]) *d)
    field_fft *= 0.5 / field.size
    print("field fft", field_fft.dtype)

    # inverse fourier transform
    #field_fft *= field.size
    #field = ifftn(field_fft) # we work in fourier
    return field_fft


def get_interlaced_field_rfft(pos, field, lbox, num_cells, paste, w):

    # cell width
    d = lbox / num_cells

    # nyquist frequency
    # kN = np.pi / d

    # natural wavemodes
    k = (fftfreq(num_cells, d=d) * 2. * np.pi).astype(np.float32) # h/Mpc
    kz = (rfftfreq(num_cells, d=d) * 2. * np.pi).astype(np.float32) # h/Mpc

    # offset by half a cell
    field_shift = get_field(pos, lbox, num_cells, paste, w, d=0.5*d)
    print("shift", field_shift.dtype, pos.dtype)
    del pos, w
    gc.collect()

    # fourier transform shifted field and sum them up
    #field_fft = fftn(field) / field.size
    #field_shift_fft = fftn(field_shift) / field.size
    field_fft = np.zeros((len(k), len(k), len(kz)), dtype=np.complex64)
    field_fft[:, :, :] = rfftn(field, workers=-1) + rfftn(field_shift, workers=-1) * \
                         np.exp(0.5 * 1j * (k[:, np.newaxis, np.newaxis] + \
                                            k[np.newaxis, :, np.newaxis] + \
                                            kz[np.newaxis, np.newaxis, :]) *d)
    field_fft *= 0.5 / field.size
    print("field fft", field_fft.dtype)

    # inverse fourier transform
    #field_fft *= field.size
    #field = ifftn(field_fft) # we work in fourier
    return field_fft

# @profile
def get_field_fft(pos, lbox, num_cells, paste, w, W, compensated, interlaced):

    # get field in real space
    field = get_field(pos, lbox, num_cells, paste, w)
    print("field, pos", field.dtype, pos.dtype)
    if interlaced:
        # get interlaced field
        #field_fft = get_interlaced_field_fft(pos, field, lbox, num_cells, paste, w)
        field_fft = get_interlaced_field_rfft(pos, field, lbox, num_cells, paste, w)
    else:
        # get Fourier modes from skewers grid
        #field_fft = fftn(field, workers=-1) / field.size
        field_fft = rfftn(field, workers=-1) / field.size
    # get rid of pos, field
    del pos, w, field
    gc.collect()

    # apply compensation filter
    if compensated:
        assert W is not None
        #field_fft /= (W[:, np.newaxis, np.newaxis] * W[np.newaxis, :, np.newaxis] * W[np.newaxis, np.newaxis, :])
        field_fft /= (W[:, np.newaxis, np.newaxis] * W[np.newaxis, :, np.newaxis] * W[np.newaxis, np.newaxis, :(num_cells//2+1)]) # tuks
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
    return W

# @profile
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
    pos[:, 0] = x1
    pos[:, 1] = y1
    pos[:, 2] = z1
    del x1, y1, z1
    gc.collect()

    # convert to fourier space
    #field_fft = get_field_fft(pos, lbox, num_cells, paste, w, W, compensated, interlaced)
    field_rfft = get_field_fft(pos, lbox, num_cells, paste, w, W, compensated, interlaced)
    del pos
    gc.collect()

    # if second field provided
    if x2 is not None:
        assert (y2 is not None) and (z2 is not None)
        # assemble the positions and compute density field
        pos2 = np.zeros((len(x2), 3), dtype=np.float32)
        pos2[:, 0] = x2
        pos2[:, 1] = y2
        pos2[:, 2] = z2
        del x2, y2, z2
        gc.collect()

        # convert to fourier space
        #field2_fft = get_field_fft(pos2, lbox, num_cells, paste, w2, W, compensated, interlaced)
        field2_rfft = get_field_fft(pos2, lbox, num_cells, paste, w2, W, compensated, interlaced)
        del pos2
        gc.collect()
    else:
        #field2_fft = None
        field2_rfft = None

    # calculate power spectrum
    n_perp = n_los = num_cells # cubic box
    k_box, mu_box, k_bin_edges, mu_bin_edges = get_k_mu_box_edges(lbox, n_perp, n_los, nbins_k, nbins_mu, k_hMpc_max, logk)
    #pk3d, N3d, binned_poles, Npoles = calc_pk3d(field_fft, lbox, k_box, mu_box, k_bin_edges, mu_bin_edges, logk, field2_fft=field2_fft, poles=poles)
    pk3d, N3d, binned_poles, Npoles = calc_pk3d(field_rfft, lbox, k_box, mu_box, k_bin_edges, mu_bin_edges, logk, field2_fft=field2_rfft, poles=poles)

    # define bin centers
    k_binc = (k_bin_edges[1:] + k_bin_edges[:-1])*.5
    mu_binc = (mu_bin_edges[1:] + mu_bin_edges[:-1])*.5
    return k_binc, mu_binc, pk3d, N3d, binned_poles, Npoles
