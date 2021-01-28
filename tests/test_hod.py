import tempfile
import filecmp
import os.path

import pytest

EXAMPLE_SIM = os.path.join(os.path.dirname(__file__), 'Mini_N64_L32')

path2config = 'config/abacus_hod.yaml'

def test_loading(tmp_path):
    '''Test loading a halo catalog
    '''

    from abacusnbody.hod.AbacusHOD import load_sims

    # check subsample file match
    





    from abacusnbody.hod.AbacusHOD.abacus_hod import AbacusHOD

    config = yaml.load(open(path2config))
    sim_params = config['sim_params']
    HOD_params = config['HOD_params']
    power_params = config['power_params']
    
    # additional parameter choices
    want_rsd = HOD_params['want_rsd']
    write_to_disk = HOD_params['write_to_disk']
    bin_params = power_params['bin_params']
    rpbins = np.logspace(bin_params['logmin'], bin_params['logmax'], bin_params['nbins'])
    pimax = power_params['pimax']
    pi_bin_size = power_params['pi_bin_size']
    
    # create a new abacushod object
    newBall = AbacusHOD(sim_params, HOD_params, power_params)
    
    # throw away run for jit to compile, write to disk
    mock_dict = newBall.run_hod(newBall.tracers, want_rsd, write_to_disk = True)


    assert filecmp.cmp(HALOS_OUTPUT,tmp_path/'halos_test.txt')

