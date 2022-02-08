#!/usr/bin/env python3
'''
This is a script for generating HOD mock catalogs.

Usage
-----
$ python ./run_lc_hod.py --help
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

    # run the HODs (note: the first time you call the function run_hod, the script takes a bit to compile)
    newBall = AbacusHOD(sim_params, HOD_params, clustering_params)
    mock_dict = newBall.run_hod(tracers=newBall.tracers, want_rsd=want_rsd, write_to_disk=write_to_disk, Nthread=16)
        
class ArgParseFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    pass

if __name__ == "__main__":

    # parsing arguments
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=ArgParseFormatter)
    parser.add_argument('--path2config', help='Path to the config file', default=DEFAULTS['path2config'])
    args = vars(parser.parse_args())
    main(**args)
