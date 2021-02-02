import os

import numpy as np

class PowerData(object):
    """
    Dummy object for calculating a likelihood
    """
    def __init__(self, data_params):
        """
        Constructor of the power spectrum data
        """
        # load the power spectrum for all tracer combinations
        power = {}
        for key in data_params['tracer_combos'].keys():
            power[key] = np.load(data_params['tracer_combos'][key]['path2power'])['xi']
        self.power = power

        # load the covariance matrix for all tracer combinations
        icov = {}
        for key in data_params['tracer_combos'].keys():
            cov = np.load(data_params['tracer_combos'][key]['path2cov'])['xicov']
            icov[key] = np.linalg.inv(cov)
        self.icov = icov


    def compute_likelihood(self, theory):
        """
        Computes the likelihood using information from the context
        """
        # Calculate a likelihood up to normalization
        lnprob = 0.
        for key in self.power.keys():
            delta = (self.power[key] - theory[key]).flatten()
            lnprob += np.einsum('i,ij,j',delta, self.icov[key], delta)
        lnprob *= -0.5

        # Return the likelihood
        print(" <><> Likelihood evaluated, lnprob = ",lnprob)
        return lnprob
