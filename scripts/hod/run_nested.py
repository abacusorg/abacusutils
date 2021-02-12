#! /usr/bin/env python

import os
import time
import sys

import numpy as np
import argparse
from dynesty import NestedSampler
import yaml
import dill
from scipy import stats

from likelihood import PowerData
from abacusnbody.hod.abacus_hod import AbacusHOD

DEFAULTS = {}
DEFAULTS['path2config'] = 'config/boss_lrg_wp.yaml'


def lnprob(p, param_mapping, param_tracer, Data, Ball):
    # read the parameters 
    for key in param_mapping.keys():
        mapping_idx = param_mapping[key]
        tracer_type = param_tracer[key]
        #tracer_type = param_tracer[params[mapping_idx, -1]]
        Ball.tracers[tracer_type][key] = p[mapping_idx]
        
    # pass them to the mock dictionary
    mock_dict = Ball.run_hod(Ball.tracers, Ball.want_rsd, Nthread = 64)

    clustering = Ball.compute_wp(mock_dict, Ball.rpbins, Ball.pimax, Ball.pi_bin_size, Nthread = 16)

    lnP = Data.compute_likelihood(theory_density)

    return lnP

# prior transform function
def prior_transform(u, params_hod, params_hod_initial_range):
    return stats.norm.ppf(u, loc = params_hod, scale = params_hod_initial_range)

def main(path2config):

    # load the yaml parameters
    config = yaml.load(open(path2config))
    sim_params = config['sim_params']
    HOD_params = config['HOD_params']
    clustering_params = config['clustering_params']
    data_params = config['data_params']
    dynesty_config_params = config['dynesty_config_params']
    fit_params = config['dynesty_fit_params']    
    
    # create a new abacushod object and load the subsamples
    newBall = AbacusHOD(sim_params, HOD_params, clustering_params)

    # read data parameters
    newData = wp_Data(data_params, HOD_params)

    # parameters to fit
    nparams = len(fit_params.keys())
    param_mapping = {}
    param_tracer = {}
    params = np.zeros((nparams, 2))
    for key in fit_params.keys():
        mapping_idx = fit_params[key][0]
        tracer_type = fit_params[key][-1]
        param_mapping[key] = mapping_idx
        param_tracer[key] = tracer_type
        params[mapping_idx, :] = fit_params[key][1:-1]

    # Make path to output
    if not os.path.isdir(os.path.expanduser(dynesty_config_params['path2output'])):
        try:
            os.makedirs(os.path.expanduser(dynesty_config_params['path2output']))
        except:
            pass
        
    # dynesty parameters
    nlive = dynesty_config_params['nlive']
    maxcall = dynesty_config_params['maxcall']
    method = dynesty_config_params['method']
    bound = dynesty_config_params['bound']

    # where to record
    prefix_chain = os.path.join(os.path.expanduser(dynesty_config_params['path2output']),
                                dynesty_config_params['chainsPrefix'])

    # initiate sampler
    found_file = os.path.isfile(prefix_chain+'.dill')
    if (not found_file) or (not dynesty_config_params['rerun']):

        # initialize our nested sampler
        sampler = NestedSampler(lnprob, prior_transform, nparams, 
            logl_args = [param_mapping, param_tracer, newData, newBall], 
            ptform_args = [params[:, 0], params[:, 1]], 
            nlive=nlive, sample = method, rstate = np.random.RandomState(dynesty_config_params['rseed']))
            # first_update = {'min_eff': 20})

    else:
        # load sampler to continue the run
        with open(prefix_chain+'.dill', "rb") as f:
            sampler = dill.load(f)
        sampler.rstate = np.load(prefix_chain+'_results.npz')['rstate']
    print("run sampler")

    sampler.run_nested(maxcall = maxcall)

    # save sampler itself
    with open(prefix_chain+'.dill', "wb") as f:
         dill.dump(sampler, f)
    res1 = sampler.results
    np.savez(prefix_chain+'_results.npz', res = res1, rstate = np.random.get_state())

class ArgParseFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=ArgParseFormatter)
    parser.add_argument('--path2config', dest='path2config', type=str, help='Path to config file.', default=DEFAULTS['path2config'])
    
    args = vars(parser.parse_args())    
    main(**args)

