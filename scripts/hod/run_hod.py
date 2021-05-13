#!/usr/bin/env python3
'''
This is a script for generating HOD mock catalogs.

Usage
-----
$ python ./run_hod.py --help
'''

import os
import glob
import time

import yaml
import numpy as np
import argparse

from abacusnbody.hod.abacus_hod import AbacusHOD

DEFAULTS = {}
DEFAULTS['path2config'] = 'config/abacus_hod.yaml'

def main(path2config):

    # load the yaml parameters
    config = yaml.load(open(path2config))
    sim_params = config['sim_params']
    HOD_params = config['HOD_params']
    clustering_params = config['clustering_params']
    
    # additional parameter choices
    want_rsd = HOD_params['want_rsd']
    write_to_disk = HOD_params['write_to_disk']
    bin_params = clustering_params['bin_params']
    rpbins = np.logspace(bin_params['logmin'], bin_params['logmax'], bin_params['nbins'] + 1)
    pimax = clustering_params['pimax']
    pi_bin_size = clustering_params['pi_bin_size']
    
    # create a new abacushod object
    newBall = AbacusHOD(sim_params, HOD_params, clustering_params)
    
    # throw away run for jit to compile, write to disk
    mock_dict = newBall.run_hod(newBall.tracers, want_rsd, write_to_disk = False, Nthread = 16)
    # mock_dict = newBall.gal_reader()
    start = time.time()
    xirppi = newBall.compute_xirppi(mock_dict, rpbins, pimax, pi_bin_size, Nthread = 32)
    print("Done xi, total time ", time.time() - start)
    # print(xirppi)
    # wp = newBall.compute_wp(mock_dict, rpbins, pimax, pi_bin_size)
    # print(wp)

    # run the fit 10 times for timing
    meantime = 0
    Ntest = 20
    for i in range(Ntest):
        print(i)
        # # run hod, ngal, xirppi
        # newBall.tracers['LRG']['alpha'] += 0.01
        # print("alpha = ",newBall.tracers['LRG']['alpha'])
        start = time.time()
        mock_dict = newBall.run_hod(newBall.tracers, want_rsd, write_to_disk, Nthread = 64)
        print("Done hod, took time ", time.time() - start)
        start = time.time()
        # ngal_dict = newBall.compute_ngal()
        # print("Done ngal, took time ", time.time() - start, ngal_dict)
        xirppi = newBall.compute_xirppi(mock_dict, rpbins, pimax, pi_bin_size, Nthread = 32)
        deltat = time.time() - start
        print("Done xi, total time ", deltat)
        meantime += deltat
    print("meantime ", meantime / Ntest)

class ArgParseFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    pass

if __name__ == "__main__":


    # parsing arguments
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=ArgParseFormatter)
    parser.add_argument('--path2config', help='Path to the config file', default=DEFAULTS['path2config'])
    args = vars(parser.parse_args())
    main(**args)
