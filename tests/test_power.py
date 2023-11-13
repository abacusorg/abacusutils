"""
Test the power spectrum module against nbodykit
"""

from pathlib import Path

import numpy as np
import pytest

_curdir = Path(__file__).parent
DATA_POWER = _curdir / 'data_power'

@pytest.fixture
def power_test_data():
    return dict(Lbox=1000.,
                **np.load(DATA_POWER / "test_pos.npz"),
                )

@pytest.mark.parametrize('interlaced', [False,True], ids=['nointer','inter'])
@pytest.mark.parametrize('compensated', [False,True], ids=['nocomp','comp'])
@pytest.mark.parametrize('paste', ['CIC','TSC'])
def test_power(power_test_data, interlaced, compensated, paste):
    from abacusnbody.analysis.power_spectrum import calc_power

    # load data
    Lbox = power_test_data['Lbox']
    pos = power_test_data['pos']

    # specifications of the power spectrum computation
    nmesh = 72
    nbins_mu = 4
    logk = False
    k_hMpc_max = np.pi*nmesh/Lbox + 1.e-6 # so that the first bin includes +/- 2pi/L which nbodykit does for this choice of nmesh
    nbins_k = nmesh//2
    poles = (0,2,4)

    # compute power
    res = calc_power(pos, Lbox, nbins_k, nbins_mu, k_hMpc_max, logk,
                       paste, nmesh, compensated, interlaced, poles=poles,
                       )

    # check that the monopole and bandpower are equal
    assert np.allclose(res['poles'][:,0],
                       (res['power'] * res['N_mode']).sum(axis=1) / res['N_mode'].sum(axis=1),
    )

    # load presaved nbodykit computation
    comp_str = "_compensated" if compensated else ""
    int_str = "_interlaced" if interlaced else ""
    fn = DATA_POWER / f"nbody_{paste}{comp_str}{int_str}.npz"
    data = np.load(fn)
    # k_nbody = data['k']
    Pkmu_nbody = data['power'].real
    # Nkmu_nbody = data['modes']

    # loop over all mu values
    for i in range(Pkmu_nbody.shape[1]):
        # compute the fractional difference [%] (note bin edges defined different)
        frac_diff = np.abs(1.-(Pkmu_nbody[:, i]/res['power'][:-1, i]).real)*100.

        # several stats of that
        mean_diff = np.nanmean(frac_diff)
        max_diff = np.nanmax(frac_diff)
        more_diff = np.sum(frac_diff > 1.)

        # print them out
        print("mean difference [%] = ", mean_diff)
        print("max difference [%] = ", max_diff)
        print("entries deviating by more than 1% = ", more_diff)

        assert mean_diff < 0.15 # mean difference should be less than 0.15%
        assert mean_diff < 5. # maximum difference shouldn't be more than 5%
        assert more_diff/nbins_k < 0.035 # less than 3.5% of entries differing by more than 1%
