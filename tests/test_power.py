"""
Test the power spectrum module against nbodykit
"""
import numpy as np
from abacusnbody.hod.power_spectrum import calc_power

def test_power():
    # load data
    Lbox = 1000.
    data = np.load("tests/data_power/test_pos.npz")
    x = data['x']
    y = data['y']
    z = data['z']

    # specifications of the power spectrum computation
    paste = "CIC"
    nmesh = 576
    nbins_mu = 4
    logk = False
    k_hMpc_max = np.pi*nmesh/Lbox + 1.e-6 # so that the first bin includes +/- 2pi/L which nbodykit does for this choice of nmesh
    nbins_k = nmesh//2

    # loop over choices
    for interlaced in [False]:
        for compensated in [False]:
            for paste in ['CIC']:
                # compute power
                k_binc, mu_binc, Pkmu, Nkmu, binned_poles, Npoles = calc_power(x, y, z, nbins_k, nbins_mu, k_hMpc_max, logk,
                                                                               Lbox, paste, nmesh, compensated, interlaced)

                # load presaved nbodykit computation
                comp_str = "_compensated" if compensated else ""
                int_str = "_interlaced" if interlaced else ""
                fn = f"tests/data_power/nbody_{paste}{comp_str}{int_str}.npz"
                data = np.load(fn)
                k_nbody = data['k']
                Pkmu_nbody = data['power'].real
                Nkmu_nbody = data['modes']

                # loop over all mu values
                for i in range(Pkmu_nbody.shape[1]):
                    # compute the fractional difference [%] (note bin edges defined different)
                    frac_diff = np.abs(1.-(Pkmu_nbody[:, i]/Pkmu[:-1, i]).real)*100.

                    # several stats of that
                    mean_diff = np.nanmean(frac_diff)
                    max_diff = np.nanmax(frac_diff)
                    more_diff = np.sum(frac_diff > 1.)

                    # print them out
                    print("mean difference [%] = ", mean_diff)
                    print("max difference [%] = ", max_diff)
                    print("entries deviating by more than 1% = ", more_diff)

                    assert mean_diff < 0.1 # mean difference should be less than 0.1%
                    assert mean_diff < 5. # maximum difference shouldn't be more than 5%
                    assert more_diff/nbins_k < 0.035 # less than 3.5% of entries differing by more than 1%
