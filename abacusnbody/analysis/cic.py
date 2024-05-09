import numpy as np
import numba
from numba import njit


@numba.vectorize
def rightwrap(x, L):
    if x >= L:
        return x - L
    return x


@njit(nogil=True)
def cic_serial(positions, density, boxsize, weights=None):
    """
    Compute density using the cloud-in-cell algorithm. Assumes cubic box
    """
    gx = np.uint32(density.shape[0])
    gy = np.uint32(density.shape[1])
    gz = np.uint32(density.shape[2])
    threeD = gz != 1
    W = 1.0
    have_W = weights is not None

    for n in range(len(positions)):
        if have_W:
            W = weights[n]

        # convert to a position in the grid
        px = (positions[n, 0] / boxsize) * gx  # used to say boxsize+0.5
        py = (positions[n, 1] / boxsize) * gy  # used to say boxsize+0.5
        if threeD:
            pz = (positions[n, 2] / boxsize) * gz  # used to say boxsize+0.5

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
        wx = 1.0 - np.abs(dx)
        if dx > 0.0:  # on the right of the center ( < )
            wxm1 = dx
            wxp1 = 0.0
        else:  # on the left of the center
            wxp1 = -dx
            wxm1 = 0.0
        wy = 1.0 - np.abs(dy)
        if dy > 0.0:
            wym1 = dy
            wyp1 = 0.0
        else:
            wyp1 = -dy
            wym1 = 0.0
        if threeD:
            wz = 1.0 - np.abs(dz)
            if dz > 0.0:
                wzm1 = dz
                wzp1 = 0.0
            else:
                wzp1 = -dz
                wzm1 = 0.0
        else:
            wz = 1.0

        # find the wrapped x,y,z grid locations of the points we need to change
        # negative indices will be automatically wrapped
        ixm1 = rightwrap(ix - 1, gx)
        ixw = rightwrap(ix, gx)
        ixp1 = rightwrap(ix + 1, gx)
        iym1 = rightwrap(iy - 1, gy)
        iyw = rightwrap(iy, gy)
        iyp1 = rightwrap(iy + 1, gy)
        if threeD:
            izm1 = rightwrap(iz - 1, gz)
            izw = rightwrap(iz, gz)
            izp1 = rightwrap(iz + 1, gz)
        else:
            izw = np.uint32(0)

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
