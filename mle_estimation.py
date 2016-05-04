'''
Created on Nov 16, 2015
Class which does MLE estimation for perfect Data.
For real Binning see POPRES-Analysis
@author: Harald
'''
l0 = 0.05  # Minimum block length which is reported


import numpy as np
import math
from statsmodels.base.model import GenericLikelihoodModel
from scipy.special import kv as kv

    
def single_pair(l_vec, r, C, sigma):
    '''Estimates the likelihood for a single pair. Assert len l_vec>0'''
    return np.sum(np.log([C * r ** 2 / (2 * l * sigma ** 2) * kv(2, np.sqrt(2 * l) * r / sigma) for l in l_vec]))
    
def pairwise_ll(l, r, C, sigma):
    '''Full Pairwise Likelihood function for data.'''
    print("C: %.5f" % C)
    print("Sigma: %.4f" % sigma)
    
    if C <= 0 or sigma <= 0:
        return np.zeros_like(l)  # If Parameters do not make sense.
    else:
        pr_noshare = -C * r / (np.sqrt(2 * l0) * sigma) * kv(1, np.sqrt(2 * l0) * r / sigma)  # Standard vector of no-sharing
        f_share = np.array([single_pair(l[i], r[i], C, sigma) if l[i] != 0 else 0.0 for i in range(0, len(r))])  # For vector send it to single_pair
        res = pr_noshare[:, 0] + f_share
        return res.astype(np.float)

def pairwise_ll01(l, r, C, sigma):
    '''Pairwise Likelihood function only caring about block or not'''
    print("C: %.5f" % C)
    print("Sigma: %.4f" % sigma)
    if C <= 0 or sigma <= 0:
        return np.zeros_like(l)  # If Parameters do not make sense.
    else:
        lambd = (C * r / (np.sqrt(2 * l0) * sigma) * kv(1, np.sqrt(2 * l0) * r / sigma))  # Probability of sharing, vectorized.
        pr_noshare = np.exp(-lambd)  # Probabilities of no sharing
        l = [len(l[i]) if l[i] != 0 else 0 for i in range(0, len(r))]  # Number of shared blocks
        pr_share = np.array([(lambd[i] ** l[i]) / math.factorial(l[i]) if l[i] != 0 else 1 for i in range(0, len(r))])
        res = pr_noshare[:, 0] * pr_share  # Bring together the two terms
        return res.astype(np.float)
    
class MLE_estimation(GenericLikelihoodModel):
    def __init__(self, endog, exog, **kwds):
        super(MLE_estimation, self).__init__(endog, exog, **kwds)

    def nloglikeobs(self, params):
        C = params[0]
        sigma = params[1]
        p_ll = pairwise_ll(self.endog, self.exog, C, sigma)
        nll = -p_ll  # First is length of shared block, second is pairwise distance
        return nll

    def fit(self, start_params=None, maxiter=10000, maxfun=5000, **kwds):
        # we have one additional parameter and we need to add it for summary
        if start_params == None:
            start_params = np.array([0.02, 2])
        return super(MLE_estimation, self).fit(start_params=start_params,
                                     maxiter=maxiter, maxfun=maxfun,
                                     **kwds)
