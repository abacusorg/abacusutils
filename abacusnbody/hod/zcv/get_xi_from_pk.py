import gc

import asdf
import numpy as np

from scipy.fft import ifftn
from scipy.special import legendre
from abacusnbody.analysis.power_spectrum import mean2d_numba_seq

def get_r_mu_box(L_hMpc, n_xy, n_z):
    """
    Compute the size of the k vector and mu for each mode. Assumes z direction is LOS
    """
    # cell width in (x,y) directions (Mpc/h)
    d_xy = L_hMpc / n_xy
    #k_xy = (fftfreq(n_xy, d=d_xy) * 2. * np.pi).astype(np.float32)
    r_xy = (np.arange(n_xy) * d_xy).astype(np.float32)

    # cell width in z direction (Mpc/h)
    d_z = L_hMpc / n_z
    #r_z = (fftfreq(n_z, d=d_z) * 2. * np.pi).astype(np.float32)
    r_z = (np.arange(n_z) * d_z).astype(np.float32)

    # h/Mpc
    x = r_xy[:, np.newaxis, np.newaxis]
    y = r_xy[np.newaxis, :, np.newaxis]
    z = r_z[np.newaxis, np.newaxis, :]
    x[x > L_hMpc/2.] -= L_hMpc
    y[y > L_hMpc/2.] -= L_hMpc
    z[z > L_hMpc/2.] -= L_hMpc
    r_box = np.sqrt(x**2 + y**2 + z**2)

    # define mu box
    mu_box = z/np.ones_like(r_box)
    mu_box[r_box > 0.] /= r_box[r_box > 0.]
    mu_box[r_box == 0.] = 0.

    # I believe the definition is that and it allows you to count all modes (agrees with nbodykit)
    mu_box = np.abs(mu_box)
    print("r box, mu box", r_box.dtype, mu_box.dtype)

    r_box = r_box.flatten()
    mu_box = mu_box.flatten()
    return r_box, mu_box

def pk_to_xi(power_tr_fn, r_bins, poles=[0, 2, 4], key='P_k3D_tr_tr'):

    # apply fourier transform to get 3D correlation
    f = asdf.open(power_tr_fn)
    Lbox = f['header']['Lbox']
    nmesh = f['header']['nmesh']
    n_perp = n_los = nmesh
    Pk = f['data'][key]
    Xi = ifftn(Pk, workers=-1).real.flatten()
    del Pk; gc.collect()

    # define r bins
    r_binc = (r_bins[1:]+r_bins[:-1])*.5

    # get r and mu box
    r_box, mu_box = get_r_mu_box(Lbox, n_perp, n_los)
    r_box = r_box.astype(np.float32)
    mu_box = mu_box.astype(np.float32)

    # bin into xi_ell(r)
    binned_poles = []
    Npoles = []
    ranges = ((r_bins[0], r_bins[-1]), (0., 1.+1.e-6)) # not doing this misses the +/- 1 mu modes in the first few bins
    nbins2d = (len(r_bins)-1, 1)
    nbins2d = np.asarray(nbins2d).astype(np.int64)
    ranges = np.asarray(ranges).astype(np.float64)
    for i in range(len(poles)):
        Ln = legendre(poles[i])
        print(r_box.dtype, Xi.dtype)
        binned_pole, Npole = mean2d_numba_seq(np.array([r_box, mu_box]), bins=nbins2d, ranges=ranges, logk=False, weights=Xi*Ln(mu_box)*(2.*poles[i]+1.))
        binned_poles.append(binned_pole)
    Npoles.append(Npole)
    Npoles = np.array(Npoles)
    binned_poles = np.array(binned_poles)

    xi_dict = {}
    xi_dict['r_binc'] = r_binc
    xi_dict['Npoles'] = Npoles
    xi_dict['binned_poles'] = binned_poles
    return xi_dict
