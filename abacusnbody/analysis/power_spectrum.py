r"""
The power spectrum module contains various useful tools for computing power
spectra (wedges, multipoles and beyond) in the cubic box.
"""

import gc
import warnings

import numpy as np
import numba
from astropy.table import Table
from scipy.fft import rfftn, irfftn, fftfreq

from .tsc import tsc_parallel
from .cic import cic_serial


__all__ = [
    'calc_power',
    'calc_pk_from_deltak',
    'pk_to_xi',
    'project_3d_to_poles',
    'get_k_mu_edges',
]

MAX_THREADS = numba.config.NUMBA_NUM_THREADS

# the first 20 factorials
FACTORIAL_LOOKUP_TABLE = np.array(
    [
        1,
        1,
        2,
        6,
        24,
        120,
        720,
        5040,
        40320,
        362880,
        3628800,
        39916800,
        479001600,
        6227020800,
        87178291200,
        1307674368000,
        20922789888000,
        355687428096000,
        6402373705728000,
        121645100408832000,
        2432902008176640000,
    ],
    dtype=np.int64,
)


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
    for i in range(2, x + 1):
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
    x = factorial(n) // (factorial(k) * factorial(n - k))
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
    sum = dtype(0.0)
    for k in range(n // 2 + 1):
        factor = dtype(n_choose_k(n, k) * n_choose_k(2 * n - 2 * k, n))
        if k % 2 == 0:
            sum += factor * x ** (dtype(0.5 * (n - 2 * k)))
        else:
            sum -= factor * x ** (dtype(0.5 * (n - 2 * k)))
    sum *= dtype(0.5**n)
    return sum


@numba.njit(parallel=True, fastmath=True)
def bin_kmu(
    n1d,
    L,
    kedges,
    muedges,
    weights,
    poles=np.empty(0, 'i8'),
    dtype=np.float32,
    fourier=True,
    nthread=MAX_THREADS,
):
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
    muedges : array_like
        edges of the mu bins. mu ranges from 0 to 1.
    weights : array_like
        array of shape (n1d, n1d, n1d//2+1) containing the power spectrum modes.
    poles : array_like
        Legendre multipoles of the power spectrum or correlation function.
    dtype : np.dtype
        float type (32 or 64) to use in calculations.
    fourier : bool
        options are Fourier space, True, which computes power spectrum,
        or configuration-space, False, which computes the correlation function.
    nthread : int, optional
        Number of numba threads to use

    Returns
    -------
    weighted_counts : ndarray of float
        mean power spectrum per (k, mu) wedge.
    counts : ndarray of int
        number of modes per (k, mu) wedge.
    weighted_counts_poles : ndarray of float
        mean power spectrum per k for each Legendre multipole.
    counts_poles : ndarray of int
        number of modes per k.
    weighted_counts_k : ndarray of float
        mean wavenumber per (k, mu) wedge.
    """

    numba.set_num_threads(nthread)

    kzlen = n1d // 2 + 1
    Nk = len(kedges) - 1
    Nmu = len(muedges) - 1
    if fourier:
        dk = 2.0 * np.pi / L
    else:
        dk = L / n1d
    kedges2 = ((kedges / dk) ** 2).astype(dtype)
    muedges2 = (muedges**2).astype(dtype)

    nthread = numba.get_num_threads()
    counts = np.zeros((nthread, Nk, Nmu), dtype=np.int64)
    weighted_counts = np.zeros((nthread, Nk, Nmu), dtype=dtype)
    Np = len(poles)
    if Np == 0:
        poles = np.empty(0, dtype=np.int64)  # so that compiler does not complain
    else:
        poles = poles.astype(np.int64)
    weighted_counts_poles = np.zeros((nthread, len(poles), Nk), dtype=dtype)
    weighted_counts_k = np.zeros((nthread, Nk, Nmu), dtype=dtype)

    # Loop over all k vectors
    for i in numba.prange(n1d):
        tid = numba.get_thread_id()
        i2 = i**2 if i < n1d // 2 else (i - n1d) ** 2
        for j in range(n1d):
            bk, bmu = 0, 0
            j2 = j**2 if j < n1d // 2 else (j - n1d) ** 2
            for k in range(kzlen):
                kmag2 = dtype(i2 + j2 + k**2)
                if kmag2 > 0:
                    invkmag2 = kmag2**-1
                    mu2 = dtype(k**2) * invkmag2
                else:
                    mu2 = dtype(0.0)  # matches nbodykit

                if kmag2 < kedges2[0]:
                    continue

                if kmag2 >= kedges2[-1]:
                    break

                while kmag2 > kedges2[bk + 1]:
                    bk += 1

                while mu2 > muedges2[bmu + 1]:
                    bmu += 1

                counts[tid, bk, bmu] += 1 if k == 0 else 2
                weighted_counts[tid, bk, bmu] += (
                    weights[i, j, k] if k == 0 else dtype(2.0) * weights[i, j, k]
                )
                weighted_counts_k[tid, bk, bmu] += (
                    np.sqrt(kmag2) * dk if k == 0 else dtype(2.0) * np.sqrt(kmag2) * dk
                )
                if Np > 0:
                    for ip in range(len(poles)):
                        pole = poles[ip]
                        if pole != 0:
                            pw = dtype(2 * pole + 1) * P_n(mu2, pole)
                            weighted_counts_poles[tid, ip, bk] += (
                                weights[i, j, k] * pw
                                if k == 0
                                else dtype(2.0) * weights[i, j, k] * pw
                            )

    counts = counts.sum(axis=0)
    weighted_counts = weighted_counts.sum(axis=0)
    weighted_counts_poles = weighted_counts_poles.sum(axis=0)
    weighted_counts_k = weighted_counts_k.sum(axis=0)
    counts_poles = counts.sum(axis=1)

    for ip, pole in enumerate(poles):
        if pole == 0:
            weighted_counts_poles[ip] = weighted_counts.sum(axis=1)

    for i in range(Nk):
        if Np > 0:
            if counts_poles[i] != 0:
                weighted_counts_poles[:, i] /= dtype(counts_poles[i])
        for j in range(Nmu):
            if counts[i, j] != 0:
                weighted_counts[i, j] /= dtype(counts[i, j])
                weighted_counts_k[i, j] /= dtype(counts[i, j])
    return (
        weighted_counts,
        counts,
        weighted_counts_poles,
        counts_poles,
        weighted_counts_k,
    )


@numba.njit(parallel=True, fastmath=True)
def bin_kppi(
    n1d,
    L,
    kedges,
    pimax,
    Npi,
    weights,
    dtype=np.float32,
    fourier=True,
    nthread=MAX_THREADS,
):
    r"""
    Compute mean and count modes in (kp, pi) bins for a 3D rfft mesh of shape (n1d, n1d, n1d//2+1)
    or a real mesh of shape (n1d, n1d, n1d).

    The kp and pi values are constructed on the fly. We use the monotonicity of
    pi with respect to kz to accelerate the bin search. The same can be used
    in real space where we only need to count the positive rz modes and double them.
    This works because we construct the `Xi(vec r)` by inverse Fourier transforming
    `P(vec k)`, so we preserve the symmetry. Note that Xi has dimensions of
    (nmesh, nmesh, nmesh).

    Parameters
    ----------
    n1d : int
        size of the 3d array along x and y dimension.
    L : float
        box size of the simulation.
    kedges : array_like
        edges of the k wavenumbers or r separation bins.
    pimax : float
        maximum value along the los for which we consider separations.
    Npi : int
        number of bins of pi, which ranges from 0 to pimax.
    weights : array_like
        array of shape (n1d, n1d, n1d//2+1) containing the power spectrum modes.
    dtype : np.dtype
        float type (32 or 64) to use in calculations.
    fourier : bool
        options are Fourier space, True, which computes power spectrum,
        or configuration-space, False, which computes the correlation function.
    nthread : int, optional
        Number of numba threads to use

    Returns
    -------
    weighted_counts : ndarray of float
        mean power spectrum per (kp, pi) bin.
    counts : ndarray of int
        number of modes per (kp, pi) bin.
    """

    numba.set_num_threads(nthread)

    kzlen = n1d // 2 + 1
    Nk = len(kedges) - 1
    if fourier:
        dk = 2.0 * np.pi / L
    else:
        dk = L / n1d
    kedges2 = ((kedges / dk) ** 2).astype(dtype)
    piedges2 = ((np.linspace(0.0, pimax, Npi + 1) / dk) ** 2).astype(dtype)

    nthread = numba.get_num_threads()
    counts = np.zeros((nthread, Nk, Npi), dtype=np.int64)
    weighted_counts = np.zeros((nthread, Nk, Npi), dtype=dtype)

    # Loop over all k vectors
    for i in numba.prange(n1d):
        tid = numba.get_thread_id()
        i2 = i**2 if i < n1d // 2 else (i - n1d) ** 2
        for j in range(n1d):
            bk, bpi = 0, 0  # kp not monotonic, but pi is monotonic
            j2 = j**2 if j < n1d // 2 else (j - n1d) ** 2
            kmag2 = dtype(i2 + j2)

            # skip until we reach bin of interest
            if kmag2 < kedges2[0]:
                continue

            # if we are out of bounds, no need to keep searching
            if kmag2 >= kedges2[-1]:
                break

            while kmag2 > kedges2[bk + 1]:
                bk += 1

            for k in range(kzlen):
                kz2 = k**2

                while kz2 > piedges2[bpi + 1]:
                    bpi += 1

                if kz2 >= piedges2[-1]:
                    break

                counts[tid, bk, bpi] += 1 if k == 0 else 2
                weighted_counts[tid, bk, bpi] += (
                    weights[i, j, k] if k == 0 else dtype(2.0) * weights[i, j, k]
                )

    counts = counts.sum(axis=0)
    weighted_counts = weighted_counts.sum(axis=0)

    for i in range(Nk):
        for j in range(Npi):
            if counts[i, j] != 0:
                weighted_counts[i, j] /= dtype(counts[i, j])
    return weighted_counts, counts


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
    assert np.max(poles) <= 10, 'numba implementation works up to ell = 10'
    nmesh = raw_p3d.shape[0]
    poles = np.asarray(poles)
    raw_p3d = np.asarray(raw_p3d)
    muedges = np.array([0.0, 1.0])
    binned_p3d, N3d, binned_poles, Npoles, k_avg = bin_kmu(
        nmesh, Lbox, k_bin_edges, muedges=muedges, weights=raw_p3d, poles=poles
    )
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

    assert np.abs((k_ell[1] - k_ell[0]) - (k_ell[-1] - k_ell[-2])) < 1.0e-6
    kzlen = n1d // 2 + 1
    numba.get_num_threads()
    Pk = np.zeros((n1d, n1d, kzlen), dtype=dtype)
    dk = dtype(2.0 * np.pi / L)
    k_ell = k_ell.astype(dtype)
    P_ell = P_ell.astype(dtype)

    # Loop over all k vectors
    for i in numba.prange(n1d):
        numba.get_thread_id()
        i2 = i**2 if i < n1d // 2 else (i - n1d) ** 2
        for j in range(n1d):
            j2 = j**2 if j < n1d // 2 else (j - n1d) ** 2
            for k in range(kzlen):
                kmag2 = dtype(i2 + j2 + k**2)
                if kmag2 > 0:
                    invkmag2 = kmag2**-1
                    mu2 = dtype(k**2) * invkmag2
                else:
                    mu2 = dtype(0.0)  # matches nbodykit
                for ip in range(len(poles)):
                    if poles[ip] == 0:
                        Pk[i, j, k] += linear_interp(
                            np.sqrt(kmag2) * dk, k_ell, P_ell[ip]
                        )
                    else:
                        Pk[i, j, k] += linear_interp(
                            np.sqrt(kmag2) * dk, k_ell, P_ell[ip]
                        ) * P_n(mu2, poles[ip])
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
    dx = x[1] - x[0]
    f = (xd - x[0]) / dx
    fl = np.int64(f)
    yd = y[fl] + (f - fl) * (y[fl + 1] - y[fl])
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
    kzlen = n1d // 2 + 1
    numba.get_num_threads()
    Sk = np.zeros((n1d, n1d, kzlen), dtype=dtype)
    dk = dtype(2.0 * np.pi / L)
    dk2 = dtype(dk**2)
    R2 = dtype(R**2)

    # Loop over all k vectors
    for i in numba.prange(n1d):
        numba.get_thread_id()
        i2 = i**2 if i < n1d // 2 else (i - n1d) ** 2
        for j in range(n1d):
            j2 = j**2 if j < n1d // 2 else (j - n1d) ** 2
            for k in range(kzlen):
                kmag2 = dtype(i2 + j2 + k**2)
                Sk[i, j, k] = np.exp(-kmag2 * dk2 * R2 / 2.0)
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
    kzlen = n1d // 2 + 1
    numba.get_num_threads()
    delta_mu2 = np.zeros((n1d, n1d, kzlen), dtype=dtype_c)

    # Loop over all k vectors
    for i in numba.prange(n1d):
        numba.get_thread_id()
        i2 = i**2 if i < n1d // 2 else (i - n1d) ** 2
        for j in range(n1d):
            j2 = j**2 if j < n1d // 2 else (j - n1d) ** 2
            for k in range(kzlen):
                kmag2 = dtype_f(i2 + j2 + k**2)
                if kmag2 > 0:
                    invkmag2 = kmag2**-1
                    mu2 = dtype_f(k**2) * invkmag2
                else:
                    mu2 = dtype_f(0.0)
                delta_mu2[i, j, k] = delta[i, j, k] * mu2
    return delta_mu2


def pk_to_xi(Pk, Lbox, r_bins, poles=[0, 2, 4]):
    r"""
    Transform 3D power spectrum into correlation function multipoles.

    Parameters
    ----------
    Pk : array_like
        3D power spectrum.
    Lbox : float
        box size of the simulation.
    r_bins : array_like
        r separation bins.
    poles : array_like
        Legendre multipoles of the power spectrum or correlation function.

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
    Xi = irfftn(Pk, workers=-1).real
    del Pk
    gc.collect()

    # define r bins
    r_binc = (r_bins[1:] + r_bins[:-1]) * 0.5

    # bin into xi_ell(r)
    nmesh = Xi.shape[0]
    poles = np.asarray(poles)
    muedges = np.array([0.0, 1.0])
    _, _, binned_poles, Npoles, r_avg = bin_kmu(
        nmesh, Lbox, r_bins, muedges=muedges, weights=Xi, poles=poles, fourier=False
    )
    binned_poles *= nmesh**3
    return r_binc, binned_poles, Npoles


def get_k_mu_edges(Lbox, k_max, kbins, mubins, logk):
    r"""
    Obtain bin edges of the k wavenumbers and mu angles.

    Parameters
    ----------
    Lbox : float
        box size of the simulation.
    k_max : float
        maximum k wavenumber.
    kbins : int or array_like
        An int indicating the number of bins of k, which ranges
        from 0 to `k_max` if `logk`, and 2pi/L to `k_max` if not.
        Or an array-like, which will be returned unchanged.
    mubins : int or array_like
        An int indicating the number of bins of mu, which ranges from 0 to 1.
        Or an array-like, which will be returned unchanged.
    logk : bool
        Logarithmic or linear k bins

    Returns
    -------
    k_bin_edges
        edges of the k wavenumbers.
    mu_bin_edges
        edges of the mu angles.
    """

    if isinstance(kbins, int):
        # define k-binning
        if logk:
            # set minimum k to make sure we cover fundamental mode
            k_min = (1.0 - 1.0e-4) * 2.0 * np.pi / Lbox
            kbins = np.geomspace(k_min, k_max, kbins + 1)
        else:
            kbins = np.linspace(0.0, k_max, kbins + 1)

    if isinstance(mubins, int):
        # define mu-binning
        mubins = np.linspace(0.0, 1.0, mubins + 1)

    return kbins, mubins


@numba.njit(parallel=True, fastmath=True)
def get_raw_power(field_fft, field2_fft=None):
    r"""
    Calculate the 3D power spectrum of a given Fourier field.

    Parameters
    ----------
    field_fft : array_like
        Fourier 3D field.

    Returns
    -------
    raw_p3d : array_like
        raw 3D power spectrum.
    """
    # calculate <deltak,deltak.conj>
    if field2_fft is not None:
        raw_p3d = (np.conj(field_fft) * field2_fft).real
    else:
        raw_p3d = np.abs(field_fft) ** 2
    return raw_p3d


def calc_pk_from_deltak(
    field_fft,
    Lbox,
    k_bin_edges,
    mu_bin_edges,
    field2_fft=None,
    poles=np.empty(0, 'i8'),
    squeeze_mu_axis=True,
    nthread=MAX_THREADS,
):
    r"""
    Calculate the power spectrum of a given Fourier field, with binning in (k,mu).
    Optionally computes Legendre multipoles in k bins.

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
    poles : np.ndarray, optional
        Legendre multipoles of the power spectrum or correlation function.
        Probably has to be a Numpy array, or Numba will complain.
    squeeze_mu_axis : bool, optional
        Remove the mu axis from the output arrays if it has length 1.
        Default: True
    nthread : int, optional
        Number of numba threads to use

    Returns
    -------
    power : array_like
        mean power spectrum per (k, mu) wedge.
    N_mode : array_like
        number of modes per (k, mu) wedge.
    binned_poles : array_like
        mean power spectrum per k for each Legendre multipole.
    N_mode_poles : array_like
        number of modes per k.
    k_avg : array_like
        mean wavenumber per (k, mu) wedge.
    """
    numba.set_num_threads(nthread)

    # get raw power
    raw_p3d = get_raw_power(field_fft, field2_fft)

    # power spectrum
    nmesh = raw_p3d.shape[0]
    power, N_mode, binned_poles, N_mode_poles, k_avg = bin_kmu(
        nmesh, Lbox, k_bin_edges, mu_bin_edges, raw_p3d, poles, nthread=nthread
    )

    # quantity above is dimensionless, multiply by box size (in Mpc/h)
    power *= Lbox**3
    if len(poles) > 0:
        binned_poles *= Lbox**3

    if squeeze_mu_axis and len(mu_bin_edges) == 2:
        power = power[:, 0]
        N_mode = N_mode[:, 0]
        k_avg = k_avg[:, 0]

    return dict(
        power=power,
        N_mode=N_mode,
        binned_poles=binned_poles,
        N_mode_poles=N_mode_poles,
        k_avg=k_avg,
    )


def get_field(
    pos, Lbox, nmesh, paste, w=None, d=0.0, nthread=MAX_THREADS, dtype=np.float32
):
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
    nthread : int, optional
        Number of numba threads to use
    dtype : np.dtype, optional
        Data type of the field

    Returns
    -------
    field : array_like
        field containing pasted particles of shape (nmesh, nmesh, nmesh).
    """
    # check if weights are requested
    if w is not None:
        assert pos.shape[0] == len(w)

    field = np.zeros((nmesh, nmesh, nmesh), dtype=dtype)
    paste = paste.upper()
    if paste == 'TSC':
        tsc_parallel(pos, field, Lbox, weights=w, nthread=nthread, offset=d)
    elif paste == 'CIC':
        warnings.warn(
            'Note that currently CIC pasting, unlike TSC, supports only a non-parallel implementation.'
        )
        if d != 0.0:
            cic_serial(pos + d, field, Lbox, weights=w)
        else:
            cic_serial(pos, field, Lbox, weights=w)
    else:
        raise ValueError(f'Unknown pasting method: {paste}')
    normalize_field(field, inplace=True, tot_weight=len(pos), nthread=nthread)
    return field


@numba.njit(parallel=True, fastmath=True)
def normalize_field(field, tot_weight=None, inplace=False, nthread=MAX_THREADS):
    """
    Normalize a cosmological density field to the overdensity convention:

    ``overdens = field / field.mean() - 1``

    If you know the total weight already (i.e. ``field.sum()``, you can pass that as
    the ``tot_weight`` argument to accelerate the computation.

    Parameters
    ----------
    field : array_like
        The field to normalize

    tot_weight : float, optional
        The total weight, i.e. ``field.sum()``

    inplace : bool, optional
        Whether to normalize in-place

    Returns
    -------
    overdens : np.ndarray
        The normalized overdensity field
    """

    numba.set_num_threads(nthread)

    dtype = field.dtype.type
    if tot_weight is None:
        # TODO parallel=True doesn't accept dtype
        tot_weight = field.sum()

    norm = dtype(field.size / tot_weight)
    if inplace:
        flatfield = field.reshape(-1)
        for i in numba.prange(len(flatfield)):
            flatfield[i] = flatfield[i] * norm - dtype(1.0)
    else:
        field = field * norm - dtype(1.0)
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
    kzlen = n1d // 2 + 1
    numba.get_num_threads()
    dk = dtype(2.0 * np.pi / L)
    d = dtype(d)
    norm = dtype(0.5 / n1d**3)
    fac = dtype(0.5 * d) * 1j

    # Loop over all k vectors
    for i in numba.prange(n1d):
        # tid = numba.get_thread_id()
        kx = dtype(i) * dk if i < n1d // 2 else dtype(i - n1d) * dk
        for j in range(n1d):
            ky = dtype(j) * dk if j < n1d // 2 else dtype(j - n1d) * dk
            for k in range(kzlen):
                kz = dtype(k) * dk
                field_fft[i, j, k] += field_shift_fft[i, j, k] * np.exp(
                    fac * (kx + ky + kz)
                )
                field_fft[i, j, k] *= norm


def get_interlaced_field_fft(
    pos, Lbox, nmesh, paste, w, nthread=MAX_THREADS, verbose=False
):
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

    # fourier transform shifted field and sum them up
    field = get_field(pos, Lbox, nmesh, paste, w)
    field_fft = rfftn(field, workers=nthread)
    del field
    gc.collect()

    # offset by half a cell
    field_shift = get_field(pos, Lbox, nmesh, paste, w, d=0.5 * d)
    field_shift_fft = rfftn(field_shift, workers=nthread)
    if verbose:
        print('shift', field_shift.dtype, pos.dtype)
    del field_shift
    del pos, w
    gc.collect()

    shift_field_fft(field_fft, field_shift_fft, nmesh, Lbox, d)
    del field_shift_fft
    gc.collect()
    if verbose:
        print('field fft', field_fft.dtype)
    return field_fft


def get_field_fft(
    pos,
    Lbox,
    nmesh,
    paste,
    w,
    W,
    compensated,
    interlaced,
    nthread=MAX_THREADS,
    verbose=False,
    dtype=np.float32,
):
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
    nthread : int, optional
        Number of numba threads to use
    verbose : bool, optional
        Print out debugging info
    dtype : np.dtype, optional
        Data type of the field

    Returns
    -------
    field_fft : array_like
        interlaced 3D Fourier field.
    """

    if interlaced:
        # get interlaced field
        field_fft = get_interlaced_field_fft(
            pos, Lbox, nmesh, paste, w, nthread=nthread
        )
    else:
        # get field in real space
        field = get_field(pos, Lbox, nmesh, paste, w, nthread=nthread, dtype=dtype)
        if verbose:
            print('field, pos', field.dtype, pos.dtype)

        # get Fourier modes from skewers grid
        inv_size = dtype(1 / field.size)
        field_fft = rfftn(field, overwrite_x=True, workers=nthread)
        _normalize(field_fft, inv_size, nthread=nthread)

    # apply compensation filter
    if compensated:
        assert W is not None
        field_fft /= (
            W[:, np.newaxis, np.newaxis]
            * W[np.newaxis, :, np.newaxis]
            * W[np.newaxis, np.newaxis, : (nmesh // 2 + 1)]
        )
    return field_fft


@numba.njit(parallel=True, fastmath=True)
def _normalize(field, a, nthread=MAX_THREADS):
    numba.set_num_threads(nthread)
    flatfield = field.reshape(-1)
    for i in numba.prange(len(flatfield)):
        flatfield[i] *= a


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
    k = (fftfreq(nmesh, d=d) * 2.0 * np.pi).astype(np.float32)  # h/Mpc

    # apply deconvolution
    paste = paste.upper()
    if interlaced:
        if paste == 'TSC':
            p = 3.0
        elif paste == 'CIC':
            p = 2.0
        else:
            raise ValueError(f'Unknown pasting method {paste}')
        W = np.sinc(0.5 * k / kN) ** p  # sinc def
    else:  # first order correction of interlacing (aka aliasing)
        s = np.sin(0.5 * np.pi * k / kN) ** 2
        if paste == 'TSC':
            W = (1 - s + 2.0 / 15 * s**2) ** 0.5
        elif paste == 'CIC':
            W = (1 - 2.0 / 3 * s) ** 0.5
        del s
    return W


def calc_power(
    pos,
    Lbox,
    kbins=None,
    mubins=None,
    k_max=None,
    logk=False,
    paste='TSC',
    nmesh=128,
    compensated=True,
    interlaced=True,
    w=None,
    pos2=None,
    w2=None,
    poles=None,
    squeeze_mu_axis=True,
    nthread=MAX_THREADS,
    dtype=np.float32,
):
    r"""
    Compute the 3D power spectrum given particle positions by first painting them on a
    cubic mesh and then applying Fourier transforms and mode counting. Outputs (k,mu)
    wedges by default; can also output Legendre multipoles.

    pos : array_like
        particle positions, shape (N,3)
    Lbox : float
        box size of the simulation.
    kbins : int, array_like, or None, optional
        An int indicating the number of bins of k, which ranges
        from 0 to `k_max` if `logk`, and 2pi/L to `k_max` if not.
        Or an array-like, which will be used as-is.
        Default is None, which sets `kbins` to `nmesh`.
    mubins : int, None, or array_like, optional
        An int indicating the number of bins of mu. mu ranges from 0 to 1.
        Or an array-like of bin edges, which will be used as-is.
        Default of None sets `mubins` to 1.
    k_max : float, optional
        maximum k wavenumber.
        Default is None, which sets `k_max` to k_Nyquist of the mesh.
    logk : bool, optional
        Logarithmic or linear k bins. Ignored if `kbins` is array-like.
        Default is False.
    paste : str, optional
        particle pasting approach (CIC or TSC). Default is 'TSC'.
    nmesh : int, optional
        size of the 3d array along x and y dimension. Default is 128.
    compensated : bool, optional
        want to apply first-order compensated filter? Default is True.
    interlaced : bool, optional
        want to apply interlacing? Default is True.
    w : array_like, optional
        weights for each particle.
    pos2 : array_like, optional
        second set of particle positions, shape (N,3)
    poles : None or list of int, optional
        Legendre multipoles of the power spectrum or correlation function.
        Default of None gives the monopole.
    squeeze_mu_axis : bool, optional
        Remove the mu axis from the output arrays if it has length 1.
        Default: True
    nthread : int, optional
        Number of numba threads to use
    dtype : np.dtype, optional
        Data type of the field

    Returns
    -------
    power : astropy.Table
        The power spectrum in an astropy Table of length ``nbins_k``. The columns are:

        - ``k_mid``: arithmetic bin centers of the k wavenumbers, shape ``(nbins_k,)``
        - ``k_avg``: mean wavenumber per (k, mu) wedge, shape ``(nbins_k,nbins_mu)``
        - ``mu_mid``: arithmetic bin centers of the mu angles, shape ``(nbins_k,nbins_mu)``
        - ``power``: mean power spectrum per (k, mu) wedge, shape ``(nbins_k,nbins_mu)``
        - ``N_mode``: number of modes per (k, mu) wedge, shape ``(nbins_k,nbins_mu)``

        If multipoles are requested via ``poles``, the table includes:

        - ``poles``: mean Legendre multipole coefficients, shape ``(nbins_k,len(poles))``
        - ``N_mode_poles``: number of modes per pole, shape ``(nbins_k,len(poles))``

        The ``meta`` field of the table will have metadata about the power spectrum.
    """
    if kbins is None:
        kbins = nmesh
    if k_max is None:
        k_max = np.pi * nmesh / Lbox
    return_mubins = mubins is not None
    if mubins is None:
        mubins = 1

    meta = dict(
        Lbox=Lbox,
        logk=logk,
        paste=paste,
        nmesh=nmesh,
        compensated=compensated,
        interlaced=interlaced,
        poles=poles,
        nthread=nthread,
        N_pos=len(pos),
        is_weighted=w is not None,
        field_dtype=dtype,
        squeeze_mu_axis=squeeze_mu_axis,
    )
    if pos2 is not None:
        meta['N_pos2'] = len(pos2)
        meta['is_weighted2'] = w2 is not None

    # get the window function
    if compensated:
        W = get_W_compensated(Lbox, nmesh, paste, interlaced)
    else:
        W = None

    # convert to fourier space
    field_fft = get_field_fft(
        pos,
        Lbox,
        nmesh,
        paste,
        w,
        W,
        compensated,
        interlaced,
        nthread=nthread,
        dtype=dtype,
    )

    # if second field provided
    if pos2 is not None:
        # convert to fourier space
        field2_fft = get_field_fft(
            pos2,
            Lbox,
            nmesh,
            paste,
            w2,
            W,
            compensated,
            interlaced,
            nthread=nthread,
            dtype=dtype,
        )
    else:
        field2_fft = None

    poles = np.asarray(poles or [], dtype=np.int64)

    # calculate power spectrum
    kbins, mubins = get_k_mu_edges(Lbox, k_max, kbins, mubins, logk)
    P = calc_pk_from_deltak(
        field_fft,
        Lbox,
        kbins,
        mubins,
        field2_fft=field2_fft,
        poles=poles,
        squeeze_mu_axis=squeeze_mu_axis,
        nthread=nthread,
    )

    # define bin centers
    k_binc = (kbins[1:] + kbins[:-1]) * 0.5
    mu_binc = (mubins[1:] + mubins[:-1]) * 0.5

    res = dict(
        k_min=kbins[:-1],
        k_max=kbins[1:],
        k_mid=k_binc,
        k_avg=P['k_avg'],
        power=P['power'],
        N_mode=P['N_mode'],
    )
    if len(poles) > 0:
        res.update(
            poles=P['binned_poles'].T,
            N_mode_poles=P['N_mode_poles'],
        )
    if return_mubins:
        res.update(
            mu_min=np.broadcast_to(mubins[:-1], res['power'].shape),
            mu_max=np.broadcast_to(mubins[1:], res['power'].shape),
            mu_mid=np.broadcast_to(mu_binc, res['power'].shape),
        )
    res = Table(res, meta=meta)

    return res
