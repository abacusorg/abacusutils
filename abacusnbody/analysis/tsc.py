import timeit
import warnings

import numba
import numpy as np

__all__ = ['tsc_parallel', 'partition_parallel']


def tsc_parallel(
    pos,
    densgrid,
    box,
    weights=None,
    nthread=-1,
    wrap=True,
    npartition=None,
    sort=False,
    coord=0,
    verbose=False,
    offset=0.0,
):
    """
    A parallel implementation of TSC mass assignment using numba. The algorithm
    partitions the particles into stripes of sufficient width that their TSC
    clouds don't overlap, and then does all the even-numbered stripes in
    parallel, followed by the odd-numbered ones.

    This method parallelizes well and can exceed, e.g., 500 million particles
    per second with 32 cores on a modern processor for a cache-resident grid
    size. That's 20 ms for 10^7 particles, which is number density 1e-3 in a
    2 Gpc/h box.

    The algorithm is expected to be bound by memory bandwidth, rather than
    CPU.  Sometimes using, e.g., half of all CPUs will be faster than using all
    of them.

    ``npartition`` is a tuning parameter.  Generally it should be at least
    ``2*nthread``, so that all threads have work to do in both passes.  Sometimes
    an even finer partitioning can produce a favorable ordering in memory of
    particles for TSC.  Sorting the particles within each stripe produces an
    even more favorable ordering, but the current implementation of sorting is
    slow enough that it's usually not worth it.

    Parameters
    ----------
    pos : ndarray of shape (n,3)
        The particles, in domain [0,box)

    densgrid : ndarray, tuple, or int
        Either an ndarray in which to write the density, or a tuple/int
        indicating the shape of the array to allocate. Can be 2D or 3D; ints
        are interpreted as 3D cubic grids. Anisotropic grids are also supported
        (nx != ny != nz).

    box : float
        The domain size. Positions are expected in domain [0,box) (but may be
        wrapped; see ``wrap``).

    weights : ndarray of shape (n,), optional
        Particle weights/masses.
        Default: None

    nthread : int, optional
        Number of threads, for both the parallel partition and the TSC.
        Values < 0 use ``numba.config.NUMBA_NUM_THREADS``, which is usually all
        CPUs.
        Default: -1

    wrap : bool, optional
        Apply an in-place periodic wrap to any out-of-bounds particle positions,
        bringing them back to domain [0,box).  This is on by default
        because it's generally fast compared to TSC.
        Default: True

    npartition : int, optional
        Number of stripes in which to partition the positions.  This is a
        tuning parameter (with certain constraints on the max value); a value
        of None will use the default of (no larger than) 2*nthread.
        Default: None

    sort : bool, optional
        Sort the particles along the ``coord`` coordinate within each partition
        stripe. This can affect performance.
        Default: False

    coord : int, optional
        The coordinate on which to partition. ``coord = 0`` means ``x``,
        ``coord = 1`` means ``y``, etc.
        Default: 0

    verbose : bool, optional
        Print some information about settings and timings.
        Default: False

    Returns
    -------
    dens : ndarray
        The density grid, which may be newly allocated, or the same as the
        input ``densgrid`` argument if that argument was an ndarray.
    """

    if nthread < 0:
        nthread = numba.config.NUMBA_NUM_THREADS
    if verbose:
        print(f'nthread={nthread}')

    numba.set_num_threads(nthread)
    if isinstance(densgrid, (int, np.integer)):
        densgrid = (densgrid, densgrid, densgrid)
    if isinstance(densgrid, tuple):
        densgrid = _zeros_parallel(densgrid)
    n1d = densgrid.shape[coord]

    if not npartition:
        if nthread > 1:
            # Can be equal to n1d//2, or less than or equal to n1d//3.
            # Must be even, and need not exceed 2*nthread.
            if 2 * nthread >= n1d // 2:
                npartition = n1d // 2
                npartition = 2 * (npartition // 2)
                if npartition < n1d // 2:
                    npartition = n1d // 3
            else:
                npartition = min(n1d // 3, 2 * nthread)
            npartition = 2 * (npartition // 2)  # must be even
        else:
            npartition = 1

    if npartition > n1d // 3 and npartition != n1d // 2 and nthread > 1:
        raise ValueError(
            f'npartition {npartition} must be less than'
            f' ngrid//3 = {n1d // 3} or equal to ngrid//2 = {n1d // 2}'
        )
    if npartition > 1 and npartition % 2 != 0 and nthread > 1:
        raise ValueError(f'npartition {npartition} not divisible by 2')
    if verbose and nthread > 1 and npartition < 2 * nthread:
        print(
            f'npartition {npartition} not large enough to use'
            f' all {nthread} threads; should be 2*nthread',
            stacklevel=2,
        )

    def _check_dtype(a, name):
        if a.itemsize > 4:
            warnings.warn(
                f'{name}.dtype={a.dtype} instead of np.float32. '
                'float32 is recommended for performance.',
            )

    _check_dtype(pos, 'pos')
    _check_dtype(densgrid, 'densgrid')
    if weights is not None:
        _check_dtype(weights, 'weights')

    if verbose:
        print(f'npartition={npartition}')

    wraptime = -timeit.default_timer()
    if wrap:
        # This could be on-the-fly instead of in-place, if needed
        _wrap_inplace(pos, box)
    wraptime += timeit.default_timer()
    if verbose:
        print(f'Wrap time: {wraptime:.4g} sec')

    if npartition > 1:
        parttime = -timeit.default_timer()
        ppart, starts, wpart = partition_parallel(
            pos,
            npartition,
            box,
            weights=weights,
            nthread=nthread,
            coord=coord,
            sort=sort,
        )
        parttime += timeit.default_timer()
        if verbose:
            print(f'Partition time: {parttime:.4g} sec')
    else:
        ppart = pos
        wpart = weights
        starts = np.array([0, len(pos)], dtype=np.int64)

    tsctime = -timeit.default_timer()
    _tsc_parallel(ppart, starts, densgrid, box, weights=wpart, offset=offset)
    tsctime += timeit.default_timer()

    if verbose:
        print(f'TSC time: {tsctime:.4g} sec')

    return densgrid


@numba.njit(parallel=True)
def _zeros_parallel(shape, dtype=np.float32):
    arr = np.empty(shape, dtype=dtype)

    for i in numba.prange(shape[0]):
        arr[i] = 0.0

    return arr


@numba.njit(parallel=True)
def _wrap_inplace(pos, box):
    for i in numba.prange(len(pos)):
        for j in range(3):
            if pos[i, j] >= box:
                pos[i, j] -= box
            elif pos[i, j] < 0:
                pos[i, j] += box


@numba.njit(parallel=True)
def _tsc_parallel(ppart, starts, dens, box, weights, offset):
    npartition = len(starts) - 1
    for i in numba.prange((npartition + 1) // 2):
        if weights is not None:
            wslice = weights[starts[2 * i] : starts[2 * i + 1]]
        else:
            wslice = None
        _tsc_scatter(
            ppart[starts[2 * i] : starts[2 * i + 1]],
            dens,
            box,
            weights=wslice,
            offset=offset,
        )
    if npartition > 1:
        for i in numba.prange((npartition + 1) // 2):
            if weights is not None:
                wslice = weights[starts[2 * i + 1] : starts[2 * i + 2]]
            else:
                wslice = None
            _tsc_scatter(
                ppart[starts[2 * i + 1] : starts[2 * i + 2]],
                dens,
                box,
                weights=wslice,
                offset=offset,
            )


@numba.njit(parallel=True, fastmath=True)
def partition_parallel(
    pos,
    npartition,
    boxsize,
    weights=None,
    coord=0,
    nthread=-1,
    sort=False,
):
    """
    A parallel partition.  Partitions a set of positions into ``npartition``
    pieces, using the ``coord`` coordinate (``coord=0`` partitions on ``x``, ``coord=1``
    partitions on ``y``, etc.).

    The particle copy stage is coded as a scatter rather than a gather.

    Note that this function is expected to be bound by memory bandwidth rather
    than CPU.

    Parameters
    ----------
    pos : ndarray of shape (n,3)
        The positions, in domain [0,boxsize)

    npartition : int
        The number of partitions

    boxsize : float
        The domain of the particles

    weights : ndarray of shape (n,), optional
        Particle weights.
        Default: None

    coord : int, optional
        The coordinate to partition on. 0 is x, 1 is y, etc.
        Default: 0 (x coordinate)

    nthread : int, optional
        Number of threads to parallelize over (using Numba threading).
        Values < 0 use ``numba.config.NUMBA_NUM_THREADS``, which is usually all
        CPUs.
        Default: -1

    sort : bool, optional
        Sort the particles on the ``coord`` coordinate within each partition.
        Can speed up subsequent TSC, but generally slow and not worth it.
        Default: False

    Returns
    -------
    partitioned : ndarray like ``pos``
        The particles, in partitioned order

    part_starts : ndarray, shape (npartition + 1,), dtype int64
        The index in ``partitioned`` where each partition starts

    wpart : ndarray or None
        The weights, in partitioned order; or None if ``weights`` not given.
    """

    if nthread < 0:
        nthread = numba.config.NUMBA_NUM_THREADS
    numba.set_num_threads(nthread)

    assert pos.shape[1] == 3

    # First pass: compute key and per-thread histogram
    dtype = pos.dtype.type
    inv_pwidth = dtype(npartition / boxsize)
    keys = np.empty(len(pos), dtype=np.int32)
    counts = np.zeros((nthread, npartition), dtype=np.int32)
    tstart = np.linspace(0, len(pos), nthread + 1).astype(np.int64)
    for t in numba.prange(nthread):
        for i in range(tstart[t], tstart[t + 1]):
            keys[i] = min(np.int32(pos[i, coord] * inv_pwidth), npartition - 1)
            counts[t, keys[i]] += 1

    # Compute start indices for parallel scatter
    pointers = np.empty(nthread * npartition, dtype=np.int64)
    pointers[0] = 0
    pointers[1:] = np.cumsum(counts.T)[:-1]
    pointers = np.ascontiguousarray(pointers.reshape(npartition, nthread).T)

    starts = np.empty(npartition + 1, dtype=np.int64)
    starts[:-1] = pointers[0]
    starts[-1] = len(pos)

    # Do parallel scatter, specializing for weights to help Numba
    psort = np.empty_like(pos)
    if weights is not None:
        wsort = np.empty_like(weights)
        for t in numba.prange(nthread):
            for i in range(tstart[t], tstart[t + 1]):
                k = keys[i]
                s = pointers[t, k]
                for j in range(3):
                    psort[s, j] = pos[i, j]
                wsort[s] = weights[i]
                pointers[t, k] += 1

        if sort:
            for i in numba.prange(npartition):
                part = psort[starts[i] : starts[i + 1]]
                iord = part[:, coord].argsort()
                part[:] = part[iord]
                weightspart = wsort[starts[i] : starts[i + 1]]
                weightspart[:] = weightspart[iord]
    else:
        wsort = None
        for t in numba.prange(nthread):
            for i in range(tstart[t], tstart[t + 1]):
                k = keys[i]
                s = pointers[t, k]
                for j in range(3):
                    psort[s, j] = pos[i, j]
                pointers[t, k] += 1

        if sort:
            for i in numba.prange(npartition):
                part = psort[starts[i] : starts[i + 1]]
                iord = part[:, coord].argsort()
                part[:] = part[iord]

    return psort, starts, wsort


@numba.njit
def _rightwrap(x, L):
    if x >= L:
        return x - L
    return x


@numba.njit(fastmath=True)
def _tsc_scatter(positions, density, boxsize, weights=None, offset=0.0):
    """
    TSC worker function. Expects particles in domain [0,boxsize).
    Supports 3D and 2D.
    """
    ftype = positions.dtype.type
    itype = np.int16
    threeD = density.ndim == 3
    gx = itype(density.shape[0])
    gy = itype(density.shape[1])
    if threeD:
        gz = itype(density.shape[2])

    inv_hx = ftype(gx / boxsize)
    inv_hy = ftype(gy / boxsize)
    if threeD:
        inv_hz = ftype(gz / boxsize)

    offset = ftype(offset)
    W = ftype(1.0)
    have_W = weights is not None

    HALF = ftype(0.5)
    P75 = ftype(0.75)
    for n in range(len(positions)):
        if have_W:
            W = ftype(weights[n])

        # convert to a position in the grid
        px = (positions[n, 0] + offset) * inv_hx
        py = (positions[n, 1] + offset) * inv_hy
        if threeD:
            pz = (positions[n, 2] + offset) * inv_hz

        # round to nearest cell center
        ix = itype(round(px))
        iy = itype(round(py))
        if threeD:
            iz = itype(round(pz))

        # calculate distance to cell center
        dx = ftype(ix) - px
        dy = ftype(iy) - py
        if threeD:
            dz = ftype(iz) - pz

        # find the tsc weights for each dimension
        wx = P75 - dx**2
        wxm1 = HALF * (HALF + dx) ** 2
        wxp1 = HALF * (HALF - dx) ** 2
        wy = P75 - dy**2
        wym1 = HALF * (HALF + dy) ** 2
        wyp1 = HALF * (HALF - dy) ** 2
        if threeD:
            wz = P75 - dz**2
            wzm1 = HALF * (HALF + dz) ** 2
            wzp1 = HALF * (HALF - dz) ** 2
        else:
            wz = ftype(1.0)

        # find the wrapped x,y,z grid locations of the points we need to change
        # negative indices will be automatically wrapped
        ixm1 = _rightwrap(ix - itype(1), gx)
        ixw = _rightwrap(ix, gx)
        ixp1 = _rightwrap(ix + itype(1), gx)
        iym1 = _rightwrap(iy - itype(1), gy)
        iyw = _rightwrap(iy, gy)
        iyp1 = _rightwrap(iy + itype(1), gy)
        if threeD:
            izm1 = _rightwrap(iz - itype(1), gz)
            izw = _rightwrap(iz, gz)
            izp1 = _rightwrap(iz + itype(1), gz)
        else:
            izw = itype(0)

        # change the 9 or 27 cells that the cloud touches
        density[ixm1, iym1, izw] += wxm1 * wym1 * wz * W
        density[ixm1, iyw, izw] += wxm1 * wy * wz * W
        density[ixm1, iyp1, izw] += wxm1 * wyp1 * wz * W
        density[ixw, iym1, izw] += wx * wym1 * wz * W
        density[ixw, iyw, izw] += wx * wy * wz * W
        density[ixw, iyp1, izw] += wx * wyp1 * wz * W
        density[ixp1, iym1, izw] += wxp1 * wym1 * wz * W
        density[ixp1, iyw, izw] += wxp1 * wy * wz * W
        density[ixp1, iyp1, izw] += wxp1 * wyp1 * wz * W

        if threeD:
            density[ixm1, iym1, izm1] += wxm1 * wym1 * wzm1 * W
            density[ixm1, iym1, izp1] += wxm1 * wym1 * wzp1 * W

            density[ixm1, iyw, izm1] += wxm1 * wy * wzm1 * W
            density[ixm1, iyw, izp1] += wxm1 * wy * wzp1 * W

            density[ixm1, iyp1, izm1] += wxm1 * wyp1 * wzm1 * W
            density[ixm1, iyp1, izp1] += wxm1 * wyp1 * wzp1 * W

            density[ixw, iym1, izm1] += wx * wym1 * wzm1 * W
            density[ixw, iym1, izp1] += wx * wym1 * wzp1 * W

            density[ixw, iyw, izm1] += wx * wy * wzm1 * W
            density[ixw, iyw, izp1] += wx * wy * wzp1 * W

            density[ixw, iyp1, izm1] += wx * wyp1 * wzm1 * W
            density[ixw, iyp1, izp1] += wx * wyp1 * wzp1 * W

            density[ixp1, iym1, izm1] += wxp1 * wym1 * wzm1 * W
            density[ixp1, iym1, izp1] += wxp1 * wym1 * wzp1 * W

            density[ixp1, iyw, izm1] += wxp1 * wy * wzm1 * W
            density[ixp1, iyw, izp1] += wxp1 * wy * wzp1 * W

            density[ixp1, iyp1, izm1] += wxp1 * wyp1 * wzm1 * W
            density[ixp1, iyp1, izp1] += wxp1 * wyp1 * wzp1 * W
