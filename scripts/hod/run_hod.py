#!/usr/bin/env python3
'''
This is a script for generating HOD mock catalogs.

Usage
-----
$ ./run_hod.py --help
'''

import os
import glob
import time

import yaml
import numpy as np
import argparse

from abacusnbody.hod.AbacusHOD.abacus_hod import AbacusHOD

DEFAULTS = {}
DEFAULTS['path2config'] = 'config/abacus_hod.yaml'

def main(path2config):

    # load the yaml parameters
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
    # xirppi = newBall.compute_xirppi(mock_dict, rpbins, pimax, pi_bin_size)

    # # run the fit 10 times for timing
    # for i in range(10):
    #     # example for sandy
    #     newBall.tracers['LRG']['alpha'] += 0
    #     print("alpha = ",newBall.tracers['LRG']['alpha'])
    #     start = time.time()
    #     mock_dict = newBall.run_hod(newBall.tracers, want_rsd, write_to_disk)
    #     print("Done iteration ", i, "took time ", time.time() - start)
        
class ArgParseFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    pass

if __name__ == "__main__":


    # parsing arguments
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=ArgParseFormatter)
    parser.add_argument('--path2config', help='Path to the config file', default=DEFAULTS['path2config'])
    args = vars(parser.parse_args())
    main(**args)
