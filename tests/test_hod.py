"""
This module tests `abacusnbody.hod` by running `prepare_sim` (the subsampling step)
and `AbacusHOD` to generate mock galaxies, and comparing the results to a reference
catalog.

To run the tests, use:
    $ pytest tests/test_hod.py

To generate new reference, run with reference_mode = True:
    $ python tests/test_hod.py
from base directory
"""

import tempfile
from os.path import dirname
from os.path import join as pjoin

import h5py
import numba
import yaml
from astropy.io import ascii
from common import check_close

# required for pytest to work (see GH #60)
numba.config.THREADING_LAYER='forksafe'

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
        # bin_params = clustering_params['bin_params']

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
            print(newhalos[i], temphalos[i])
            for j in range(len(newhalos[i])):
                assert check_close(newhalos[i][j], temphalos[i][j])
        newparticles = h5py.File(savedir+'/particles_xcom_2_seed600_abacushod_oldfenv_MT_new.h5', 'r')['particles']
        tempparticles = h5py.File(EXAMPLE_SUBSAMPLE_PARTS, 'r')['particles']
        for i in range(len(newparticles)):
            for j in range(len(newparticles[i])):
                assert check_close(newparticles[i][j], tempparticles[i][j])

        # additional parameter choices
        want_rsd = HOD_params['want_rsd']

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

        # smoke test for zcv
        config['sim_params']['sim_name'] = 'AbacusSummit_base_c000_ph006' # so that meta can find it
        config['sim_params']['z_mock'] = 0.8 # so that meta can find it
        config['HOD_params']['want_rsd'] = False # so that it doesn't run twice
        config['zcv_params']['zcv_dir'] = pjoin(TESTDIR, 'data_zcv')
        config['zcv_params']['tracer_dir'] = pjoin(tmp_path, 'zcv_tracer_data')
        mock_dict = newBall.run_hod(newBall.tracers, want_rsd = config['HOD_params']['want_rsd'], write_to_disk = False, Nthread = 2)
        del mock_dict['ELG']  # drop ELG since zcv works with a single tracer currently
        # zcv_dict =
        newBall.apply_zcv(mock_dict, config)

if __name__ == '__main__':
    with tempfile.TemporaryDirectory() as tmpdir:
        test_hod(tmpdir, reference_mode = False)
