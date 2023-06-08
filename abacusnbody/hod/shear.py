import time

import numba
import numpy as np
import numpy.linalg as la
from numba import njit
from scipy.ndimage import gaussian_filter

# from nbodykit.lab import ArrayCatalog, FieldMesh
# from nbodykit.base.mesh import MeshFilter


@numba.vectorize
def rightwrap(x, L):
    if x >= L:
        return x - L
    return x

@njit
def dist(pos1, pos2, L=None):
    '''
    Calculate L2 norm distance between a set of points
    and either a reference point or another set of points.
    Optionally includes periodicity.
    Parameters
    ----------
    pos1: ndarray of shape (N,m)
        A set of points
    pos2: ndarray of shape (N,m) or (m,) or (1,m)
        A single point or set of points
    L: float, optional
        The box size. Will do a periodic wrap if given.
    Returns
    -------
    dist: ndarray of shape (N,)
        The distances between pos1 and pos2
    '''

    # read dimension of data
    N, nd = pos1.shape

    # allow pos2 to be a single point
    pos2 = np.atleast_2d(pos2)
    assert pos2.shape[-1] == nd
    broadcast = len(pos2) == 1

    dist = np.empty(N, dtype=pos1.dtype)

    i2 = 0
    for i in range(N):
        delta = 0.
        for j in range(nd):
            dx = pos1[i][j] - pos2[i2][j]
            if L is not None:
                if dx >= L/2:
                    dx -= L
                elif dx < -L/2:
                    dx += L
            delta += dx*dx
        dist[i] = np.sqrt(delta)
        if not broadcast:
            i2 += 1
    return dist

@njit(nopython=True, nogil=True)
def numba_tsc_3D(positions, density, boxsize, weights=np.empty(0)):
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
        wxm1 = .5*(.5 + dx)**2
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
        ixm1 = (ix - 1)
        ixw  = rightwrap(ix    , gx)
        ixp1 = rightwrap(ix + 1, gx)
        iym1 = (iy - 1)
        iyw  = rightwrap(iy    , gy)
        iyp1 = rightwrap(iy + 1, gy)
        if threeD:
            izm1 = (iz - 1)
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

def smooth_density(D, R, N_dim, Lbox):
    # cell size
    cell = Lbox/N_dim
    # smoothing scale
    R /= cell
    D_smooth = gaussian_filter(D, R)
    return D_smooth

# tophat
@njit(nopython=True)
def Wth(ksq, r):
    k = np.sqrt(ksq)
    w = 3*(np.sin(k*r)-k*r*np.cos(k*r))/(k*r)**3
    return w

# gaussian
@njit(nopython=True)
def Wg(k, r):
    return np.exp(-k*r*r/2.)


@njit(nopython=True)
def get_tidal(dfour, karr, N_dim, R):

    # initiate array
    tfour = np.zeros(shape=(N_dim, N_dim, N_dim, 3, 3),dtype=np.complex128)#complex)

    # computing tidal tensor
    for a in range(N_dim):
        for b in range(N_dim):
            for c in range(N_dim):
                if (a, b, c) == (0, 0, 0): continue

                ksq = karr[a]**2 + karr[b]**2 + karr[c]**2
                # smoothed density Gauss fourier
                #dksmo[a, b, c] = Wg(ksq)*dfour[a, b, c]
                # smoothed density TH fourier
                #dkth[a, b, c] = Wth(ksq)*dfour[a, b, c]
                # all 9 components
                tfour[a, b, c, 0, 0] = karr[a]*karr[a]*dfour[a, b, c]/ksq
                tfour[a, b, c, 1, 1] = karr[b]*karr[b]*dfour[a, b, c]/ksq
                tfour[a, b, c, 2, 2] = karr[c]*karr[c]*dfour[a, b, c]/ksq
                tfour[a, b, c, 1, 0] = karr[a]*karr[b]*dfour[a, b, c]/ksq
                tfour[a, b, c, 0, 1] = tfour[a, b, c, 1, 0]
                tfour[a, b, c, 2, 0] = karr[a]*karr[c]*dfour[a, b, c]/ksq
                tfour[a, b, c, 0, 2] = tfour[a, b, c, 2, 0]
                tfour[a, b, c, 1, 2] = karr[b]*karr[c]*dfour[a, b, c]/ksq
                tfour[a, b, c, 2, 1] = tfour[a, b, c, 1, 2]
                if R is not None:
                    tfour[a, b, c, :, :] *= Wth(ksq, R)
    return tfour

@njit(nopython=True)
def get_shear_nb(tidr, N_dim):
    shear = np.zeros(shape=(N_dim, N_dim, N_dim), dtype=np.float64)
    for a in range(N_dim):
        for b in range(N_dim):
            for c in range(N_dim):
                t = tidr[a, b, c]
                evals, evects = la.eig(t)
                # ascending
                idx = evals.argsort()
                evals = evals[idx]
                evects = evects[:, idx]
                l1 = evals[0]
                l2 = evals[1]
                l3 = evals[2]
                shear[a, b, c] = 0.5*((l2-l1)**2 + (l3-l1)**2 + (l3-l2)**2)
    return shear

def get_shear(dsmo, N_dim, Lbox, R=None):

    # fourier transform the density field
    dfour = np.fft.fftn(dsmo)

    # k values
    karr = np.fft.fftfreq(N_dim, d=Lbox/(2*np.pi*N_dim))

    # creating empty arrays for future use
    start = time.time()
    tfour = get_tidal(dfour, karr, N_dim, R)
    tidr = np.real(np.fft.ifftn(tfour, axes = (0, 1, 2)))
    print("finished tidal, took time", time.time() - start)

    # compute shear
    start = time.time()
    shear = np.sqrt(get_shear_nb(tidr, N_dim))
    print("finished shear, took time", time.time() - start)

    return shear
