#!/usr/bin/env python3
'''
This is a script for generating HOD mock catalogs.

MODIFIED BS abacus_hod.py and GRAND_HOD.py in anaconda3
/global/homes/b/boryanah/anaconda3/envs/desc/lib/python3.7/site-packages/abacusnbody/hod/GRAND_HOD.py

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
DEFAULTS['path2config'] = 'config/lc_hod.yaml'

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

    # run the HODs
    newBall = AbacusHOD(sim_params, HOD_params, clustering_params)
    mock_dict = newBall.run_hod(tracers=newBall.tracers, want_rsd=want_rsd, write_to_disk=write_to_disk, Nthread=16)

    # can change some parameter and run again to time
    zs = [0.1]
    for i in range(len(zs)):
        # create a new abacushod object
        sim_params['z_mock'] = zs[i]
        newBall = AbacusHOD(sim_params, HOD_params, clustering_params)
        start = time.time()
        mock_dict = newBall.run_hod(tracers=newBall.tracers, want_rsd=want_rsd, write_to_disk=False, Nthread=16)
        print("Done hod, took time ", time.time() - start)
        
        
class ArgParseFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    pass

if __name__ == "__main__":


    # parsing arguments
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=ArgParseFormatter)
    parser.add_argument('--path2config', help='Path to the config file', default=DEFAULTS['path2config'])
    args = vars(parser.parse_args())
    main(**args)
