"""
This module tests `abacusnbody.hod` by running `prepare_sim` (the subsampling step)
and `AbacusHOD` to generate mock galaxies, and comparing the results to a reference
catalog.

To run the tests, use:
    $ pytest test_hod.py

To generate new reference, run:
    $ python test_hod.py
"""

from os.path import dirname, join as pjoin
import tempfile

import yaml
import pytest
import h5py
import numpy as np
from astropy.io import ascii

from common import check_close


TESTDIR = dirname(__file__)
REFDIR = pjoin(dirname(__file__), 'ref_hod')
EXAMPLE_SIM = pjoin(TESTDIR, 'Mini_N64_L32')
EXAMPLE_CONFIG = pjoin(TESTDIR, 'abacus_hod.yaml')
EXAMPLE_SUBSAMPLE_HALOS = pjoin(REFDIR, 
    'Mini_N64_L32/z0.000/halos_xcom_2_seed600_abacushod_oldfenv_MT_new.h5')
EXAMPLE_SUBSAMPLE_PARTS = pjoin(REFDIR, 
    'Mini_N64_L32/z0.000/particles_xcom_2_seed600_abacushod_oldfenv_MT_new.h5')
EXAMPLE_LRGS = pjoin(REFDIR, 
    'Mini_N64_L32/z0.000/galaxies_rsd/LRGs.dat')
EXAMPLE_ELGS = pjoin(REFDIR, 
    'Mini_N64_L32/z0.000/galaxies_rsd/ELGs.dat')


# @pytest.mark.xfail
def test_hod(tmp_path, reference_mode = False):
    '''Test loading a halo catalog
    '''
    from abacusnbody.hod import prepare_sim
    from abacusnbody.hod.abacus_hod import AbacusHOD

    config = yaml.safe_load(open(EXAMPLE_CONFIG))
    # inform abacus_hod where the simulation files are, relative to the cwd
    config['sim_params']['sim_dir'] = TESTDIR
    
    sim_params = config['sim_params']
    HOD_params = config['HOD_params']
    clustering_params = config['clustering_params']

    # reference mode
    if reference_mode:
        print("Generating new reference files...")
        prepare_sim.main(EXAMPLE_CONFIG, params=config)
        
        # additional parameter choices
        want_rsd = HOD_params['want_rsd']
        bin_params = clustering_params['bin_params']
        
        # create a new abacushod object
        newBall = AbacusHOD(sim_params, HOD_params, clustering_params)
        mock_dict = newBall.run_hod(newBall.tracers, want_rsd, write_to_disk = True, Nthread = 2)

    # test mode
    else:
        simname = config['sim_params']['sim_name'] # "AbacusSummit_base_c000_ph006"
        z_mock = config['sim_params']['z_mock']
        # all output dirs should be under tmp_path
        config['sim_params']['output_dir'] = pjoin(tmp_path, 'data_mocks_summit_new') + '/'
        config['sim_params']['subsample_dir'] = pjoin(tmp_path, "data_subs") + '/'
        config['sim_params']['scratch_dir'] = pjoin(tmp_path, "data_gals") + '/'
        savedir = config['sim_params']['subsample_dir'] + simname+"/z"+str(z_mock).ljust(5, '0') + '/'

        # check subsample file match
        prepare_sim.main(EXAMPLE_CONFIG, params = config)

        newhalos = h5py.File(savedir+'/halos_xcom_2_seed600_abacushod_oldfenv_MT_new.h5', 'r')['halos']
        temphalos = h5py.File(EXAMPLE_SUBSAMPLE_HALOS, 'r')['halos']
        for i in range(len(newhalos)):
            for j in range(len(newhalos[i])):
                assert check_close(newhalos[i][j], temphalos[i][j])
        newparticles = h5py.File(savedir+'/particles_xcom_2_seed600_abacushod_oldfenv_MT_new.h5', 'r')['particles']
        tempparticles = h5py.File(EXAMPLE_SUBSAMPLE_PARTS, 'r')['particles']
        for i in range(len(newparticles)):
            for j in range(len(newparticles[i])):
                assert check_close(newparticles[i][j], tempparticles[i][j])

        # additional parameter choices
        want_rsd = HOD_params['want_rsd']
        write_to_disk = HOD_params['write_to_disk']
        bin_params = clustering_params['bin_params']
        rpbins = np.logspace(bin_params['logmin'], bin_params['logmax'], bin_params['nbins'])
        pimax = clustering_params['pimax']
        pi_bin_size = clustering_params['pi_bin_size']
        
        # create a new abacushod object
        newBall = AbacusHOD(sim_params, HOD_params, clustering_params)
        
        # throw away run for jit to compile, write to disk
        mock_dict = newBall.run_hod(newBall.tracers, want_rsd, write_to_disk = True, Nthread = 2)
        savedir_gal = config['sim_params']['output_dir']\
            +"/"+simname+"/z"+str(z_mock).ljust(5, '0') +"/galaxies_rsd/LRGs.dat"
        data = ascii.read(EXAMPLE_LRGS)
        data1 = ascii.read(savedir_gal)
        for ekey in data.keys():
            assert check_close(data[ekey], data1[ekey])

        savedir_gal = config['sim_params']['output_dir']\
            +"/"+simname+"/z"+str(z_mock).ljust(5, '0') +"/galaxies_rsd/ELGs.dat"
        data = ascii.read(EXAMPLE_ELGS)
        data1 = ascii.read(savedir_gal)
        for ekey in data.keys():
            assert check_close(data[ekey], data1[ekey])


if __name__ == '__main__':
    with tempfile.TemporaryDirectory() as tmpdir:
        test_hod(tmpdir, reference_mode = False)
