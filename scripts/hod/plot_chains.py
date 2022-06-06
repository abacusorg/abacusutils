import os

import numpy as np
import matplotlib.pyplot as plt
import argparse
import getdist
from getdist import plots, MCSamples
import yaml

DEFAULTS = {}
DEFAULTS['path2config'] = 'config/abacus_hod.yaml'

def get_samples(outfile, par_names, w_rat, n_par, b_iter):
    marg_chains = np.loadtxt(outfile)
    # uncomment for when your chains have been complete
    #marg_chains = marg_chains[w_rat*n_par*b_iter:]
    marg_chains = marg_chains[3*marg_chains.shape[0]//4:]
    hsc = MCSamples(samples=marg_chains, names=par_names)
    return hsc

def main(path2config):
    # read parameters
    config = yaml.load(open(path2config))
    fit_params = config['fit_params']
    ch_params = config['ch_config_params']

    # parameters
    n_iter = ch_params['sampleIterations']
    w_rat = ch_params['walkersRatio']
    b_iter = ch_params['burninIterations']
    par_names = fit_params.keys()
    lab_names = par_names
    n_par = len(par_names)

    # what are we plotting
    HOD_pars = par_names
    filename = "triangle_test.png"
    dir_chains = ch_params['path2output']

    # walkers ratio, number of params and burn in iterations
    marg_outfile = os.path.join(dir_chains, (ch_params['chainsPrefix']+".txt"))

    # read the samples after removing burnin
    marg_hsc = get_samples(marg_outfile, par_names, w_rat, n_par, b_iter)

    # Triangle plot
    g = plots.getSubplotPlotter()
    g.settings.legend_fontsize = 20
    g.settings.scaling_factor = 0.1
    g.triangle_plot([marg_hsc], params=HOD_pars)
    plt.savefig(filename)
    plt.close()

class ArgParseFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=ArgParseFormatter)
    parser.add_argument('--path2config', dest='path2config', type=str, help='Path to config file.', default=DEFAULTS['path2config'])
    
    args = vars(parser.parse_args())    
    main(**args)

