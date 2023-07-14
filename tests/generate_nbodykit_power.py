import numpy as np
from nbodykit.lab import ArrayCatalog, FFTPower


def CompensateTSC(w, v):
    """
    Return the Fourier-space kernel that accounts for the convolution of
    the gridded field with the TSC window function in configuration space

    .. note::
        see equations in
        `Jing et al 2005 <https://arxiv.org/abs/astro-ph/0409240>`_

    Parameters
    ----------
    w : list of arrays
        the list of "circular" coordinate arrays, ranging from
        :math:`[-\pi, \pi)`.
    v : array_like
        the field array
    """
    for i in range(3):
        wi = w[i]
        tmp = (np.sinc(0.5 * wi / np.pi) ) ** 3
        v = v / tmp
    return v

def CompensateCIC(w, v):
    """
    Return the Fourier-space kernel that accounts for the convolution of
    the gridded field with the CIC window function in configuration space

    .. note::
        see equations in
        `Jing et al 2005 <https://arxiv.org/abs/astro-ph/0409240>`_

    Parameters
    ----------
    w : list of arrays
        the list of "circular" coordinate arrays, ranging from
        :math:`[-\pi, \pi)`.
    v : array_like
        the field array
    """
    for i in range(3):
        wi = w[i]
        tmp = (np.sinc(0.5 * wi / np.pi) ) ** 2
        v = v / tmp
    return v


def CompensateTSCShotnoise(w, v):
    """
    Return the Fourier-space kernel that accounts for the convolution of
    the gridded field with the TSC window function in configuration space,
    as well as the approximate aliasing correction to the first order

    .. note::
        see equations in
        `Jing et al 2005 <https://arxiv.org/abs/astro-ph/0409240>`_

    Parameters
    ----------
    w : list of arrays
        the list of "circular" coordinate arrays, ranging from
        :math:`[-\pi, \pi)`.
    v : array_like
        the field array
    """
    for i in range(3):
        wi = w[i]
        s = np.sin(0.5 * wi)**2
        v = v / (1 - s + 2./15 * s**2) ** 0.5
    return v

def CompensateCICShotnoise(w, v):
    """
    Return the Fourier-space kernel that accounts for the convolution of
    the gridded field with the CIC window function in configuration space,
    as well as the approximate aliasing correction to the first order

    .. note::
        see equations in
        `Jing et al 2005 <https://arxiv.org/abs/astro-ph/0409240>`_

    Parameters
    ----------
    w : list of arrays
        the list of "circular" coordinate arrays, ranging from
        :math:`[-\pi, \pi)`.
    v : array_like
        the field array
    """
    for i in range(3):
        wi = w[i]
        s = np.sin(0.5 * wi)**2
        v = v / (1 - 2. / 3. * s) ** 0.5
    return v

def power_test_data():
    return dict(Lbox=1000.,
                **np.load("tests/data_power/test_pos.npz"),
                )

def generate_nbody(power_test_data, interlaced=False, compensated=False, paste='CIC'):

    # load data
    power_test_data = power_test_data()
    Lbox = power_test_data['Lbox']
    pos = power_test_data['pos']

    # specifications of the power spectrum computation
    nmesh = 72
    nbins_mu = 4
    kmin = 0.
    dk = 2.*np.pi/Lbox
    kmax = np.pi*nmesh/Lbox
    poles = [0, 2, 4]

    # file to save to
    comp_str = "_compensated" if compensated else ""
    int_str = "_interlaced" if interlaced else ""
    fn = f"tests/data_power/nbody_{paste}{comp_str}{int_str}.npz"

    # create mesh object
    cat = ArrayCatalog({'Position': pos.T})

    # convert to a MeshSource, apply compensation (same as nbodykit, but need to apply manually, as it fails on some nbodykit versions)
    mesh = cat.to_mesh(window=paste.lower(), Nmesh=nmesh, BoxSize=Lbox, interlaced=interlaced, compensated=False, position='Position')
    if compensated and interlaced:
        if paste == "TSC":
            compensation = CompensateTSC
        elif paste == "CIC":
            compensation = CompensateCIC
    elif compensated and not interlaced:
        if paste == "TSC":
            compensation = CompensateTSCShotnoise
        elif paste == "CIC":
            compensation = CompensateCICShotnoise
    if compensated:
        mesh = mesh.apply(compensation, kind='circular', mode='complex')

    # compute the 2D power
    r = FFTPower(mesh, mode='2d', kmin=kmin, dk=dk, kmax=kmax, Nmu=nbins_mu, los=[0,0,1], poles=poles)
    p_ell = r.poles
    k = p_ell['k']
    modes = p_ell['modes']
    P_ell = []
    for pole in poles:
        P_ell.append(p_ell[f'power_{pole:d}'])
    P_ell = np.array(P_ell)
    Pkmu = r.power
    #k = Pkmu['k']
    k = r.power.coords['k'] # bin centers (matches abacusutils preference)
    modes = Pkmu['modes']
    Pkmu = Pkmu['power']

    # save arrays
    np.savez(fn, k=k, power=Pkmu, modes=modes, power_ell=P_ell)

if __name__ == '__main__':
    for compensated in [True, False]:
        for interlaced in [True, False]:
            for paste in ['TSC', 'CIC']:
                generate_nbody(power_test_data, interlaced=interlaced, compensated=compensated, paste=paste)
