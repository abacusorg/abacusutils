
r"""
The power spectrum module contains various useful tools for computing power 
spectra (wedges, multipoles and beyond) in the cubic box.
"""

import gc

import numpy as np
import numba
import asdf
from scipy.fft import rfftn, irfftn, fftfreq

from .tsc import tsc_parallel
from .cic import cic_serial

# the first 20 factorials
FACTORIAL_LOOKUP_TABLE = np.array([
    1, 1, 2, 6, 24, 120, 720, 5040, 40320,
    362880, 3628800, 39916800, 479001600,
    6227020800, 87178291200, 1307674368000,
    20922789888000, 355687428096000, 6402373705728000,
    121645100408832000, 2432902008176640000], dtype=np.int64)

@numba.njit
def factorial(n):
    r"""
    Compute the factorial for some integer.
    
    Parameters
    ----------
    n : int
        integer number for which to calculate the factorial.
        Must be less than or equal to 20 and non-negative.
    
    Returns
    -------
    factorial : int
        the factorial of the requested integer.
    """
    if n > 20 or n < 0:
        raise ValueError
    factorial = FACTORIAL_LOOKUP_TABLE[n]
    return factorial

@numba.njit
def factorial_slow(x):
    r"""
    Brute-force compute the factorial for some integer.
    
    Parameters
    ----------
    x : int
        integer number for which to calculate the factorial.
    
    Returns
    -------
    n : int
        the factorial of the requested integer.
    """
    n = 1
    for i in range(2, x+1):
        n *= i
    return n

@numba.njit
def n_choose_k(n, k):
    r"""
    Compute binomial coefficient for a choice of two integers (n, k).
    
    Parameters
    ----------
    n : int
        the integer `n` in n-choose-k.
    k : int
        the integer `k` in n-choose-k.
    
    Returns
    -------
    x : int
        binomial coefficient, n-choose-k.
    """
    x = factorial(n)//(factorial(k)*factorial(n-k))
    return x

@numba.njit
def P_n(x, n, dtype=np.float32):
    r"""
    Computes Legendre polynomial of order n for some squared quantity x. Maximum tested 
    order of the polynomial is 10, after which we see deviations from `scipy`.

    Parameters
    ----------
    x : float
        variable in the polynomial.
    n : int
        order of the Legendre polynomial.

    Returns
    -------
   sum : float
        evaluation of the polynomial at `x`.
    """
    sum = dtype(0.)
    for k in range(n//2+1):
        factor = dtype(n_choose_k(n, k) * n_choose_k(2*n - 2*k, n))
        if k % 2 == 0:
            sum += factor*x**(dtype(0.5*(n-2*k)))
        else:
            sum -= factor*x**(dtype(0.5*(n-2*k)))
    sum *= dtype(0.5**n)
    return sum


@numba.njit(parallel=True, fastmath=True)
def bin_kmu(n1d, L, kedges, Nmu, weights, poles=np.array([]), dtype=np.float32, space='fourier'):
    r"""
    Compute mean and count modes in (k,mu) bins for a 3D rfft mesh of shape (n1d, n1d, n1d//2+1)
    or a real mesh of shape (n1d, n1d, n1d).

    The k and mu values are constructed on the fly. We use the monotonicity of
    k and mu with respect to kz to accelerate the bin search. The same can be used
    in real space where we only need to count the positive rz modes and double them.
    This works because we construct the `Xi(vec r)` by inverse Fourier transforming
    `P(vec k)`.

    Parameters
    ----------
    n1d : int
        size of the 3d array along x and y dimension.
    L : float
        box size of the simulation.
    kedges : array_like
        edges of the k wavenumbers or r separation bins.
    Nmu : int
        number of bins of mu, which ranges from 0 to 1.
    weights : array_like
        array of shape (n1d, n1d, n1d//2+1) containing the power spectrum modes.
    poles : array_like
        Legendre multipoles of the power spectrum or correlation function.
    dtype : np.dtype
        float type (32 or 64) to use in calculations.
    space : str
        options are Fourier space, `fourier`, which computes power spectrum,
        or configuration-space, `real`, which computes the correlation function.

    Returns
    -------
    weighted_counts : array_like
        mean power spectrum per (k, mu) wedge.
    counts : array_like
        number of modes per (k, mu) wedge.
    weighted_counts_poles : array_like
        mean power spectrum per k for each Legendre multipole.
    counts_poles
        number of modes per k.
    """
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

def project_3d_to_poles(k_bin_edges, raw_p3d, Lbox, poles):
    r"""
    Project 3D power spectrum into multipoles of the power spectrum.

    Parameters
    ---------
    k_bin_edges : array_like
        edges of the k wavenumbers.
    raw_p3d : array_like
        array containing the power spectrum modes.
    Lbox : float
        box size of the simulation.
    poles : array_like
        Legendre multipoles of the power spectrum or correlation function.

    Returns
    -------
    binned_poles : array_like
        mean power spectrum per k for each Legendre multipole.
    Npoles : array_like
        number of modes per k.
    """
    assert np.max(poles) <= 10, "numba implementation works up to ell = 10"
    nmesh = raw_p3d.shape[0]
    poles = np.array(poles)
    raw_p3d = np.asarray(raw_p3d)
    binned_p3d, N3d, binned_poles, Npoles = bin_kmu(nmesh, Lbox, k_bin_edges, Nmu=1, weights=raw_p3d, poles=poles)
    binned_poles *= Lbox**3
    return binned_poles, Npoles

@numba.njit(parallel=True, fastmath=True)
def expand_poles_to_3d(k_ell, P_ell, n1d, L, poles, dtype=np.float32):
    r"""
    Expand power spectrum multipoles to a 3D power spectrum evaluated at the fundamental modes of the box.
    Uses a custom version of linear interpolation, since np.interp is very slow when not vectorized.

    Parameters
    ----------
    k_ell : array_like
        wavenumbers at which multipoles of the power spectrum are evaluated.
    P_ell : array_like
        power spectrum multipoles to be interpolated at the 3D k-vector values.
    n1d : int
        size of the 3d array along x and y dimension.
    L : float
        box size of the simulation.
    poles : array_like
        Legendre multipoles of the power spectrum.

    Returns
    -------
    Pk : array_like
        array of shape (n1d, n1d, n1d//2+1) containing 3D power spectrum modes.
    """
    
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
    r"""
    Custom linear interpolation. Assumes `x` entries are equidistant and monotonically increasing.
    Assigns `y[0]` and `y[-1]` to the leftmost and rightmost edges, respectively.

    Parameters
    ----------
    xd : float
        x-value at which to evaluate function y(x).
    x : array_type
        equidistantly separated x values at which function y is provided.
    y : array_type
        y values at each x.

    Returns
    -------
    yd : float
        linearly interpolated value at `xd`.
    """
    if xd <= x[0]:
        return y[0]
    elif xd >= x[-1]:
        return y[-1]
    dx = x[1]-x[0]
    f = (xd-x[0])/dx
    fl = np.int64(f)
    yd = y[fl] + (f - fl) * (y[fl+1] - y[fl])
    return yd

@numba.njit(parallel=True, fastmath=True)
def get_smoothing(n1d, L, R, dtype=np.float32):
    r"""
    Construct Gaussian kernel of the form exp(-k^2 R^2/2) as a 3D Fourier field.

    Parameters
    ----------
    n1d : int
        size of the 3d array along x and y dimension.
    L : float
        box size of the simulation.
    R : float
        smoothing scale in units of Mpc/h provided [L] = Mpc/h.

    Returns
    -------
    Sk : array_like
        smoothing kernel of shape (n1d, n1d, n1d//2+1).
    """
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
    r"""
    Obtain delta*mu^2 field by multiplying delta by mu^2 in parallel.
    Note that `delta` here is a Fourier 3D field.

    Parameters
    ----------
    delta : array_like
        Fourier 3D field of shape (n1d, n1d, n1d//2+1).
    n1d : int
        size of the 3d array along x and y dimension.
    dtype_c : np.dtype
        complex dtype (64 or 128).
    dtype_f : np.dtype
        float dtype (32 or 64).

    Returns
    -------
    delta_mu2 : array_like
        array of shape (n1d, n1d, n1d//2+1) containing delta*mu^2.
    """
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
    r"""
    Transform 3D power spectrum into correlation function multipoles.
    Reads ASDF file locally to save memory.

    Parameters
    ----------
    pk_fn : str
        Name of ASDF file from which to read power spectrum.
    Lbox : float
        box size of the simulation.
    r_bins : array_like
        r separation bins.
    poles : array_like
        Legendre multipoles of the power spectrum or correlation function.
    key : str
        key name in the data structure of the ASDF file containing 3D power spectrum.

    Returns
    -------
    r_binc : array_like
        r separation bin centers.
    binned_poles : array_like
        correlation function multipoles.
    Npoles : array_like
        number of modes per r bin.
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

def get_k_mu_edges(Lbox, k_max, n_k_bins, n_mu_bins, logk):
    r"""
    Obtain bin edges of the k wavenumbers and mu angles.
    
    Parameters
    ----------
    Lbox : float
        box size of the simulation.
    k_max : float
        maximum k wavenumber.
    n_k_bins : int
        number of bins of k, which ranges from 0 to `k_max` if `logk == True` 
        and 2pi/L (incl.) to `k_max` if `logk == False`. 
    n_mu_bins : int
        number of bins of mu, which ranges from 0 to 1.
    logk : bool
        Logarithmic or linear k bins

    Returns
    -------
    k_bin_edges
        edges of the k wavenumbers.
    mu_bin_edges
        edges of the mu angles.   
    """
    # define k-binning
    if logk:
        # set minimum k to make sure we cover fundamental mode
        k_min = (1.-1.e-4)*2.*np.pi/Lbox
        k_bin_edges = np.geomspace(k_min, k_max, n_k_bins+1)
    else:
        k_bin_edges = np.linspace(0., k_max, n_k_bins+1)

    # define mu-binning
    mu_bin_edges = np.linspace(0., 1., n_mu_bins + 1)
    return k_bin_edges, mu_bin_edges

def calc_pk3d(field_fft, Lbox, k_bin_edges, mu_bin_edges, field2_fft=None, poles=[]):
    r"""
    Calculate the power spectrum of a given Fourier field.

    Parameters
    ----------
    field_fft : array_like
        Fourier 3D field.
    Lbox : float
        box size of the simulation.
    k_bin_edges : array_like
        edges of the k wavenumbers.
    mu_bin_edges : array_like
        edges of the mu angles.
    field2_fft : array_like, optional
        second Fourier 3D field, used in cross-correlation.
    poles : array_like, optional
        Legendre multipoles of the power spectrum or correlation function.        

    Returns
    -------
    p3d : array_like
        mean power spectrum per (k, mu) wedge.
    N3d : array_like
        number of modes per (k, mu) wedge.
    binned_poles : array_like
        mean power spectrum per k for each Legendre multipole.
    Npoles : array_like
        number of modes per k.
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
    binned_p3d, N3d, binned_poles, Npoles = bin_kmu(nmesh, Lbox, k_bin_edges, Nmu, raw_p3d, poles)

    # quantity above is dimensionless, multiply by box size (in Mpc/h)
    p3d = binned_p3d * Lbox**3
    if len(poles) > 0:
        binned_poles *= Lbox**3
    return p3d, N3d, binned_poles, Npoles

# @profile
def get_field(pos, Lbox, nmesh, paste, w=None, d=0.):
    r"""
    Construct real-space 3D field given particle positions.

    Parameters
    ----------
    pos : array_like
        particle positions of shape (N, 3).
    Lbox : float
        box size of the simulation.
    nmesh : int
        size of the 3d array along x and y dimension.
    paste :
        particle pasting approach (CIC or TSC).
    w : array_like, optional
        weights for each particle.
    d : float, optional
        uniform shift to particle positions.

    Returns
    -------
    field : array_like
        field containing pasted particles of shape (nmesh, nmesh, nmesh).
    """
    # check if weights are requested
    if w is not None:
        assert pos.shape[0] == len(w)
    pos = pos.astype(np.float32)
    field = np.zeros((nmesh, nmesh, nmesh), dtype=np.float32)
    if paste == 'TSC':
        if d != 0.:
            # TODO: could add an offset parameter to tsc_parallel
            tsc_parallel(pos + np.float32(d), field, Lbox, weights=w)
        else:
            tsc_parallel(pos, field, Lbox, weights=w)
    elif paste == 'CIC':
        if d != 0.:
            cic_serial(pos + np.float32(d), field, Lbox, weights=w)
        else:
            cic_serial(pos, field, Lbox, weights=w)
    if w is None: # in the zcv code the weights are already normalized, so don't normalize here
        field /= (pos.shape[0]/nmesh**3.) # same as passing "Value" to nbodykit (1+delta)(x) V(x)
        field -= 1. # leads to -1 in the complex field
    return field

@numba.njit(parallel=True, fastmath=True)
def shift_field_fft(field_fft, field_shift_fft, n1d, L, d, dtype=np.float32):
    r"""
    Computed interlaced field in Fourier space by combining original and shifted
    (by half a cell size) field.

    Parameters
    ----------
    field_fft : array_like
        Fourier 3D field.
    field_shift_fft : array_like
        shifted Fourier 3D field.
    n1d : int
        size of the 3d array along x and y dimension.
    L : float
        box size of the simulation.
    d : float
        uniform shift to particle positions.
    dtype : np.dtype
        float type (32 or 64) to use in calculations.

    Returns
    -------
    field_fft : array_like
        Modified original array.
    """
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

def get_interlaced_field_fft(pos, field, Lbox, nmesh, paste, w):
    r"""
    Calculate interlaced field from particle positions and return 3D Fourier field.

    pos : array_like
        particle positions of shape (N, 3)
    field : array_like
        field containing pasted particles of shape (nmesh, nmesh, nmesh).
    Lbox : float
        box size of the simulation.
    nmesh : int
        size of the 3d array along x and y dimension.
    paste :
        particle pasting approach (CIC or TSC).
    w : array_like, optional
        weights for each particle.

    Returns
    -------
    field_fft : array_like
        interlaced 3D Fourier field.
    """
    # cell width
    d = Lbox / nmesh

    # offset by half a cell
    field_shift = get_field(pos, Lbox, nmesh, paste, w, d=0.5*d)
    print("shift", field_shift.dtype, pos.dtype)
    del pos, w
    gc.collect()

    # fourier transform shifted field and sum them up
    field_fft = rfftn(field, workers=-1)
    field_shift_fft = rfftn(field_shift, workers=-1)
    shift_field_fft(field_fft, field_shift_fft, nmesh, Lbox, d)
    del field_shift_fft
    gc.collect()
    print("field fft", field_fft.dtype)
    return field_fft

# @profile
def get_field_fft(pos, Lbox, nmesh, paste, w, W, compensated, interlaced):
    r"""
    Calculate field from particle positions and return 3D Fourier field.

    pos : array_like
        particle positions of shape (N, 3)
    Lbox : float
        box size of the simulation.
    nmesh : int
        size of the 3d array along x and y dimension.
    paste :
        particle pasting approach (CIC or TSC).
    w : array_like
        weights for each particle.
    W : array_like
        TSC/CIC compensated filter in Fourier space.
    compensated : bool
        want to apply first-order compensated filter?
    interlaced : bool
        want to apply interlacing?

    Returns
    -------
    field_fft : array_like
        interlaced 3D Fourier field.
    """
    
    # get field in real space
    field = get_field(pos, Lbox, nmesh, paste, w)
    print("field, pos", field.dtype, pos.dtype)
    if interlaced:
        # get interlaced field
        field_fft = get_interlaced_field_fft(pos, field, Lbox, nmesh, paste, w)
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

def get_W_compensated(Lbox, nmesh, paste, interlaced):
    r"""
    Compute the TSC/CIC kernel convolution for a given set of wavenumbers.

    Parameters
    ----------
    Lbox : float
        box size of the simulation.
    nmesh : int
        size of the 3d array along x and y dimension.
    paste : str
        particle pasting approach (CIC or TSC).
    interlaced : bool
        want to apply interlacing?

    Returns
    -------
    W : array_like
        TSC/CIC compensated filter in Fourier space.
    """

    # cell width
    d = Lbox / nmesh

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
def calc_power(x1, y1, z1, nbins_k, nbins_mu, k_max, logk, Lbox, paste, nmesh, compensated = True, interlaced = True, w = None, x2 = None, y2 = None, z2 = None, w2 = None, poles=[]):
    r"""
    Compute the 3D power spectrum given particle positions by first painting them on a cubic
    mesh and then applying Fourier transforms and mode counting. Can output Legendre multipoles,
    (k, mu) wedges, or both.

    TODO: Return a dictionary.

    x1 : array_like
        particle positions in the x dimension.
    y1 : array_like
        particle positions in the y dimension.
    z1 : array_like
        particle positions in the z dimension.
    nbins_k : int
        number of bins of k, which ranges from 0 to `k_max` if `logk == True` 
        and 2pi/L (incl.) to `k_max` if `logk == False`. 
    nbins_mu : int
        number of bins of mu, which ranges from 0 to 1.
    k_max : float
        maximum k wavenumber.
    logk : bool
        Logarithmic or linear k bins
    Lbox : float
        box size of the simulation.
    paste : str
        particle pasting approach (CIC or TSC).
    nmesh : int
        size of the 3d array along x and y dimension.
    compensated : bool, optional
        want to apply first-order compensated filter? Default is True.
    interlaced : bool, optional
        want to apply interlacing? Default is True.
    w : array_like, optional
        weights for each particle.
    x2 : array_like, optional
        second set of particle positions in the x dimension.
    y2 : array_like, optional
        second set of particle positions in the y dimension.
    z2 : array_like, optional
        second set of particle positions in the z dimension.
    poles : 
        Legendre multipoles of the power spectrum or correlation function.

    Returns
    -------
    k_binc : array_like
        airthmetic bin centers of the mu angles
    mu_binc : array_like
        airthmetic bin centers of the k wavenumbers.
    p3d : array_like
        mean power spectrum per (k, mu) wedge.
    N3d : array_like
        number of modes per (k, mu) wedge.
    binned_poles : array_like
        mean power spectrum per k for each Legendre multipole.
    Npoles : array_like
        number of modes per k.
    """

    # get the window function
    if compensated:
        W = get_W_compensated(Lbox, nmesh, paste, interlaced)
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
    field_fft = get_field_fft(pos, Lbox, nmesh, paste, w, W, compensated, interlaced)
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
        field2_fft = get_field_fft(pos2, Lbox, nmesh, paste, w2, W, compensated, interlaced)
        del pos2
        gc.collect()
    else:
        field2_fft = None

    # calculate power spectrum
    k_bin_edges, mu_bin_edges = get_k_mu_edges(Lbox, k_max, nbins_k, nbins_mu, logk)
    p3d, N3d, binned_poles, Npoles = calc_pk3d(field_fft, Lbox, k_bin_edges, mu_bin_edges, field2_fft=field2_fft, poles=poles)
    if len(poles) == 0:
        binned_poles, Npoles = [], []
    
    # define bin centers
    k_binc = (k_bin_edges[1:] + k_bin_edges[:-1])*.5
    mu_binc = (mu_bin_edges[1:] + mu_bin_edges[:-1])*.5
    return k_binc, mu_binc, p3d, N3d, binned_poles, Npoles
