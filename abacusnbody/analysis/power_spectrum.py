"""
# immediate important:
make the prepare steps easier (and for zenbu Xi too)
(ask Lehman) do documentation (maybe start with docstrings)

you might be able to speed up ic_fields
maybe make the power nmesh not a required argument (with get) and then provide bare minimum files

# immediate cosmetic:
you could fix jdr's bias thing to be yours which would help if a user wants custom 1 delta delta2 etc.

# longer-term important:
add cic parallel (should be easy) _tsc_parallel (maybe ask Lehman about this)

# longer-term cosmetic:
maybe can select k_hMpc_min?
(maybe not) compute_power and calc_power give it directly bin_edges
"""

import gc

import numpy as np
import numba
import asdf
from scipy.fft import rfftn, irfftn, fftfreq

from .tsc import tsc_parallel
from .cic import cic_serial

FACTORIAL_LOOKUP_TABLE = np.array([
    1, 1, 2, 6, 24, 120, 720, 5040, 40320,
    362880, 3628800, 39916800, 479001600,
    6227020800, 87178291200, 1307674368000,
    20922789888000, 355687428096000, 6402373705728000,
    121645100408832000, 2432902008176640000], dtype='int64')

@numba.njit
def factorial(n):
    """
    Computes factorial assuming n integer
    """
    if n > 20:
        raise ValueError
    return FACTORIAL_LOOKUP_TABLE[n]

@numba.njit
def factorial_slow(x):
    """
    Computes factorial assuming x integer
    """
    n = 1
    for i in range(2, x+1):
        n *= i
    return n

@numba.njit
def n_choose_k(n, k):
    """
    Computes binomial coefficient assuming n, k integers
    """
    return factorial(n)//(factorial(k)*factorial(n-k))

@numba.njit
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
def bin_kmu(n1d, L, kedges, Nmu, weights, poles=np.array([]), dtype=np.float32, space='fourier'):
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

    if space == 'fourier':
        dk = 2.*np.pi / L
    elif space == 'real':
        dk = L / n1d
    kedges2 = ((kedges/dk)**2).astype(dtype)
    muedges2 = (np.linspace(0., 1., Nmu+1)**2).astype(dtype)

    nthread = numba.get_num_threads()
    counts = np.zeros((nthread, Nk, Nmu), dtype=np.int64)
    weighted_counts = np.zeros((nthread, Nk, Nmu), dtype=dtype)
    Np = len(poles)
    if Np == 0:
        poles = np.zeros(1, dtype=np.int64) # so that compiler does not complain
    else:
        poles = poles.astype(np.int64)
    counts_poles = np.zeros((nthread, Nk), dtype=np.int64)
    weighted_counts_poles = np.zeros((nthread, len(poles), Nk), dtype=dtype)

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
                    for ip in range(len(poles)):
                        pole = poles[ip]
                        if pole == 0:
                            weighted_counts_poles[tid, ip, bk] += weights[i, j, k] if k == 0 else dtype(2.)*weights[i, j, k]
                        else:
                            pw = dtype(2*pole + 1)*P_n(mu2, pole)
                            weighted_counts_poles[tid, ip, bk] += weights[i, j, k]*pw if k == 0 else dtype(2.)*weights[i, j, k]*pw

    counts = counts.sum(axis=0)
    weighted_counts = weighted_counts.sum(axis=0)
    counts_poles = counts_poles.sum(axis=0)
    weighted_counts_poles = weighted_counts_poles.sum(axis=0)

    for i in range(Nk):
        if Np > 0:
            if counts_poles[i] != 0:
                for ip in range(len(poles)):
                    weighted_counts_poles[ip, i] /= dtype(counts_poles[i])
        for j in range(Nmu):
            if counts[i, j] != 0:
                weighted_counts[i, j] /= dtype(counts[i, j])
    return weighted_counts, counts, weighted_counts_poles, counts_poles

def project_3d_to_poles(k_bin_edges, raw_p3d, L_hMpc, poles):
    """
    Project 3D power spectrum into multipoles of the power spectrum.
    """
    assert np.max(poles) <= 10, "numba implementation works up to ell = 10"
    nmesh = raw_p3d.shape[0]
    poles = np.array(poles)
    raw_p3d = np.asarray(raw_p3d)
    binned_p3d, N3d, binned_poles, Npoles = bin_kmu(nmesh, L_hMpc, k_bin_edges, Nmu=1, weights=raw_p3d, poles=poles)
    binned_poles *= L_hMpc**3
    return binned_poles, Npoles

@numba.njit(parallel=True, fastmath=True)
def expand_poles_to_3d(k_ell, P_ell, n1d, L, poles, dtype=np.float32):
    '''
    Expand power spectrum multipoles to a 3D power spectrum evaluated at the fundamental modes of the box.
    Uses a custom version of linear interpolation, since np.interp is very slow when not vectorized.
    '''
    assert np.abs((k_ell[1]-k_ell[0]) - (k_ell[-1]-k_ell[-2])) < 1.e-6
    kzlen = n1d//2 + 1
    numba.get_num_threads()
    Pk = np.zeros((n1d, n1d, kzlen), dtype=dtype)
    dk = dtype(2. * np.pi / L)
    k_ell = k_ell.astype(dtype)
    P_ell = P_ell.astype(dtype)

    # Loop over all k vectors
    for i in numba.prange(n1d):
        numba.get_thread_id()
        i2 = i**2 if i <= n1d//2 else (n1d - i)**2
        for j in range(n1d):
            j2 = j**2 if j <= n1d//2 else (n1d - j)**2
            for k in range(kzlen):
                kmag2 = dtype(i2 + j2 + k**2)
                if kmag2 > 0:
                    invkmag2 = kmag2**-1
                    mu2 = dtype(k**2) * invkmag2
                else:
                    mu2 = dtype(0.) # matches nbodykit
                for ip in range(len(poles)):
                    if poles[ip] == 0:
                        Pk[i, j, k] += linear_interp(np.sqrt(kmag2)*dk, k_ell, P_ell[ip])
                    else:
                        Pk[i, j, k] += linear_interp(np.sqrt(kmag2)*dk, k_ell, P_ell[ip]) * P_n(mu2, poles[ip])
    return Pk

@numba.njit
def linear_interp(xd, x, y):
    """
    Linear interpolation. Assumes x entries are equidistant and increasing.
    """
    if xd < x[0]:
        return y[0]
    elif xd > x[-1]:
        return y[-1]
    dx = x[1]-x[0]
    f = (xd-x[0])/dx
    fl = np.int64(f)
    return y[fl] + (f - fl) * (y[fl+1] - y[fl])

@numba.njit(parallel=True, fastmath=True)
def get_smoothing(n1d, L, R, dtype=np.float32):
    '''
    Gaussian smoothing for a 3D k-vector of the form exp(-k^2 R^2/2)
    '''
    kzlen = n1d//2 + 1
    numba.get_num_threads()
    Sk = np.zeros((n1d, n1d, kzlen), dtype=dtype)
    dk = dtype(2. * np.pi / L)
    dk2 = dtype(dk**2)
    R2 = dtype(R**2)

    # Loop over all k vectors
    for i in numba.prange(n1d):
        numba.get_thread_id()
        i2 = i**2 if i <= n1d//2 else (n1d - i)**2
        for j in range(n1d):
            j2 = j**2 if j <= n1d//2 else (n1d - j)**2
            for k in range(kzlen):
                kmag2 = dtype(i2 + j2 + k**2)
                Sk[i, j, k] = np.exp(-kmag2*dk2*R2/2.)
    return Sk

@numba.njit(parallel=True, fastmath=True)
def get_delta_mu2(delta, n1d, dtype_c=np.complex64, dtype_f=np.float32):
    '''
    Multiply delta by mu2
    '''
    kzlen = n1d//2 + 1
    numba.get_num_threads()
    delta_mu2 = np.zeros((n1d, n1d, kzlen), dtype=dtype_c)

    # Loop over all k vectors
    for i in numba.prange(n1d):
        numba.get_thread_id()
        i2 = i**2 if i <= n1d//2 else (n1d - i)**2
        for j in range(n1d):
            j2 = j**2 if j <= n1d//2 else (n1d - j)**2
            for k in range(kzlen):
                kmag2 = dtype_f(i2 + j2 + k**2)
                if kmag2 > 0:
                    invkmag2 = kmag2**-1
                    mu2 = dtype_f(k**2) * invkmag2
                else:
                    mu2 = dtype_f(0.)
                delta_mu2[i, j, k] = delta[i, j, k] * mu2
    return delta_mu2

def pk_to_xi(pk_fn, Lbox, r_bins, poles=[0, 2, 4], key='P_k3D_tr_tr'):
    """
    Transform power spectrum into correlation function
    """
    # apply fourier transform to get 3D correlation
    Pk = asdf.open(pk_fn)['data'][key] # open inside function to save space
    Xi = irfftn(Pk, workers=-1).real
    del Pk; gc.collect() # noqa: E702

    # define r bins
    r_binc = (r_bins[1:]+r_bins[:-1])*.5

    # bin into xi_ell(r)
    nmesh = Xi.shape[0]
    poles = np.array(poles)
    _, _, binned_poles, Npoles = bin_kmu(nmesh, Lbox, r_bins, Nmu=1, weights=Xi, poles=poles, space='real')
    binned_poles *= nmesh**3
    return r_binc, binned_poles, Npoles

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
    return k_bin_edges, mu_bin_edges

def calc_pk3d(field_fft, L_hMpc, k_bin_edges, mu_bin_edges, field2_fft=None, poles=[]):
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
    nmesh = raw_p3d.shape[0]
    Nmu = len(mu_bin_edges) - 1
    poles = np.array(poles)
    binned_p3d, N3d, binned_poles, Npoles = bin_kmu(nmesh, L_hMpc, k_bin_edges, Nmu, raw_p3d, poles)

    # quantity above is dimensionless, multiply by box size (in Mpc/h)
    p3d_hMpc = binned_p3d * L_hMpc**3
    if len(poles) > 0:
        binned_poles *= L_hMpc**3
    return p3d_hMpc, N3d, binned_poles, Npoles

# @profile
def get_field(pos, L_hMpc, nmesh, paste, w=None, d=0.):
    # check if weights are requested
    if w is not None:
        assert pos.shape[0] == len(w)
    pos = pos.astype(np.float32)
    field = np.zeros((nmesh, nmesh, nmesh), dtype=np.float32)
    if paste == 'TSC':
        if d != 0.:
            # TODO: could add an offset parameter to tsc_parallel
            tsc_parallel(pos + np.float32(d), field, L_hMpc, weights=w)
        else:
            tsc_parallel(pos, field, L_hMpc, weights=w)
    elif paste == 'CIC':
        if d != 0.:
            cic_serial(pos + np.float32(d), field, L_hMpc, weights=w)
        else:
            cic_serial(pos, field, L_hMpc, weights=w)
    if w is None: # in the zcv code the weights are already normalized, so don't normalize here
        field /= (pos.shape[0]/nmesh**3.) # same as passing "Value" to nbodykit (1+delta)(x) V(x)
        field -= 1. # leads to -1 in the complex field
    return field

@numba.njit(parallel=True, fastmath=True)
def shift_field_fft(field_fft, field_shift_fft, n1d, L, d, dtype=np.float32):
    '''
    Expand power spectrum multipoles to a 3D power spectrum evaluated at the fundamental modes of the box.
    '''
    kzlen = n1d//2 + 1
    numba.get_num_threads()
    dk = dtype(2. * np.pi / L)
    d = dtype(d)
    norm = dtype(0.5/n1d**3)
    fac = dtype(0.5 * d) * 1j

    # Loop over all k vectors
    for i in numba.prange(n1d):
        #tid = numba.get_thread_id()
        kx = dtype(i)*dk if i < n1d//2 else dtype(i - n1d)*dk
        for j in range(n1d):
            ky = dtype(j)*dk if j < n1d//2 else dtype(j - n1d)*dk
            for k in range(kzlen):
                kz = dtype(k)*dk
                field_fft[i, j, k] += field_shift_fft[i, j, k]*np.exp(fac * (kx + ky + kz))
                field_fft[i, j, k] *= norm

def get_interlaced_field_fft(pos, field, L_hMpc, nmesh, paste, w):

    # cell width
    d = L_hMpc / nmesh

    # offset by half a cell
    field_shift = get_field(pos, L_hMpc, nmesh, paste, w, d=0.5*d)
    print("shift", field_shift.dtype, pos.dtype)
    del pos, w
    gc.collect()

    # fourier transform shifted field and sum them up
    field_fft = rfftn(field, workers=-1)
    field_shift_fft = rfftn(field_shift, workers=-1)
    shift_field_fft(field_fft, field_shift_fft, nmesh, L_hMpc, d)
    del field_shift_fft
    gc.collect()
    print("field fft", field_fft.dtype)
    return field_fft

# @profile
def get_field_fft(pos, L_hMpc, nmesh, paste, w, W, compensated, interlaced):

    # get field in real space
    field = get_field(pos, L_hMpc, nmesh, paste, w)
    print("field, pos", field.dtype, pos.dtype)
    if interlaced:
        # get interlaced field
        field_fft = get_interlaced_field_fft(pos, field, L_hMpc, nmesh, paste, w)
    else:
        # get Fourier modes from skewers grid
        field_fft = rfftn(field, workers=-1) / np.float32(field.size)
    # get rid of pos, field
    del pos, w, field
    gc.collect()

    # apply compensation filter
    if compensated:
        assert W is not None
        field_fft /= (W[:, np.newaxis, np.newaxis] * W[np.newaxis, :, np.newaxis] * W[np.newaxis, np.newaxis, :(nmesh//2+1)])
    return field_fft

def get_W_compensated(L_hMpc, nmesh, paste, interlaced):
    """
    Compute the TSC/CIC kernel convolution for a given set of wavenumbers.
    """

    # cell width
    d = L_hMpc / nmesh

    # nyquist frequency
    kN = np.pi / d

    # natural wavemodes
    k = (fftfreq(nmesh, d=d) * 2. * np.pi).astype(np.float32) # h/Mpc

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
def calc_power(x1, y1, z1, nbins_k, nbins_mu, k_hMpc_max, logk, L_hMpc, paste, nmesh, compensated, interlaced, w = None, x2 = None, y2 = None, z2 = None, w2 = None, poles=[]):
    """
    Compute the 3D power spectrum given particle positions by first painting them on a cubic mesh and then applying the fourier transforms and mode counting.
    """

    # get the window function
    if compensated:
        W = get_W_compensated(L_hMpc, nmesh, paste, interlaced)
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
    field_fft = get_field_fft(pos, L_hMpc, nmesh, paste, w, W, compensated, interlaced)
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
        field2_fft = get_field_fft(pos2, L_hMpc, nmesh, paste, w2, W, compensated, interlaced)
        del pos2
        gc.collect()
    else:
        field2_fft = None

    # calculate power spectrum
    k_bin_edges, mu_bin_edges = get_k_mu_edges(L_hMpc, k_hMpc_max, nbins_k, nbins_mu, logk)
    pk3d, N3d, binned_poles, Npoles = calc_pk3d(field_fft, L_hMpc, k_bin_edges, mu_bin_edges, field2_fft=field2_fft, poles=poles)

    # define bin centers
    k_binc = (k_bin_edges[1:] + k_bin_edges[:-1])*.5
    mu_binc = (mu_bin_edges[1:] + mu_bin_edges[:-1])*.5
    return k_binc, mu_binc, pk3d, N3d, binned_poles, Npoles
