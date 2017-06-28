'''
Created on Nov 16, 2015
Class which does MLE estimation
@author: Harald
'''
# l0 = 8.0  # Minimum block length which is reported
'''Here everything is measured in centi morgan!!'''

import numpy as np
import math
# from scipy import stats
# import statsmodels.api as sm
from statsmodels.base.model import GenericLikelihoodModel
from scipy.special import kv as kv

    
def single_pair(l_vec, r, C, sigma):
    '''Estimates the likelihood for a single pair. Assert len l_vec>0'''
    return np.sum(np.log([C * r ** 2 / (2.0 * l / 100.0 * sigma ** 2) * kv(2, np.sqrt(2.0 * l / 100.0) * r / sigma) for l in l_vec]))
    
def pairwise_ll(l, exog, l0, C, sigma):
    '''Full Pairwise Likelihood function for data.'''
    r = exog[:, 0]  # Distance between populations
    pw_nr = exog[:, 1]  # Number of pairwise Comparisons

    print("C: %.5f" % C)
    print("Sigma: %.4f" % sigma)
    
    if C <= 0 or sigma <= 0:
        return np.ones(len(l)) * (-np.inf)  # If Parameters do not make sense return infinitely negative likelihood
    else:
        pr_noshare_lg = (-C * r / (np.sqrt(2.0 * l0 / 100.0) * sigma) * kv(1, np.sqrt(2.0 * l0 / 100.0) * r / sigma)) * pw_nr  # Standard vector of no-sharing
        f_share_lg = np.array([single_pair(l[i], r[i], C, sigma) if (len(l[i]) > 0) else 1.0 for i in range(0, len(r))])  # For vector send it to single_pair
        res = pr_noshare_lg + f_share_lg
        return res.astype(np.float)

def single_pair_growth(l_vec, r, C, sigma, mu):
    '''Pairwise Log Likelihood function for data with exponential growth.'''
    a = np.sum(np.log([power_block(l) for l in l_vec]))  # Update for power BEFORE CHANGING TO EFFECTIVE RECOMBINATION RATE
    l_eff = l_vec - mu / 2.0  # Update to effective recombination rate in face of growth
    # True probability
    b = np.sum(np.log([C * r ** 2 / (2.0 * l / 100.0 * sigma ** 2) * kv(2, np.sqrt(2.0 * l / 100.0) * r / sigma) for l in l_eff]))
    return(a + b)

def single_pair_dd(l_vec, r, C, sigma):
    '''Pairwise Log Likelihood function for data with doomsday growth.'''
    a = np.sum(np.log([power_block(l) for l in l_vec]))  # Update for power BEFORE CHANGING TO EFFECTIVE RECOMBINATION RATE
    # True probability
    b = np.sum(np.log([C * r ** 3 / (4.0 * np.sqrt(2) * (l / 100 * sigma ** 2) ** (3.0 / 2.0)) * kv(3, np.sqrt(2.0 * l / 100.0) * r / sigma) for l in l_vec]))
    return(a + b)
       
def pairwise_ll_growth(l, exog, l0, C, sigma, mu):
    '''Full Pairwise Log(!) Likelihood function for data.'''
    r = exog[:, 0]  # Distance between populations
    pw_nr = exog[:, 1]  # Number of pairwise Comparisons

    print("C: %.5f" % C)
    print("Sigma: %.4f" % sigma)
    print("Mu: %.6f " % mu)
    
    if C <= 0 or sigma <= 0 or mu / 100.0 >= 2.0 * l0:
        return np.ones(len(l)) * (-np.inf)  # If Parameters do not make sense return infinitely negative likelihood
    else:
        le = l0 - mu / 2.0  # Update to effective recombination rate in face of pop growth
        pr_noshare_lg = (-C * r / (np.sqrt(2.0 * le / 100.0) * sigma) * kv(1, np.sqrt(2.0 * le / 100.0) * r / sigma)) * pw_nr  # Standard vector of no-sharing
        f_share_lg = np.array([single_pair_growth(l[i], r[i], C, sigma, mu) if (len(l[i]) > 0) else 1.0 for i in range(0, len(r))])  # For vector send it to single_pair
        power_noshare = power_block(l0)
        res = pr_noshare_lg * power_noshare + f_share_lg
        # print(res)
        print("Total likelihood: %.3f" % np.sum(res))
        return res.astype(np.float)   
         

def pairwise_ll01(l, r, l0, C, sigma):
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

def pairwise_ll_dd(l, exog, l0, C, sigma):
    '''Full Pairwise Log(!) Likelihood function for data. For Doomsday population growth'''
    r = exog[:, 0]  # Distance between populations
    pw_nr = exog[:, 1]  # Number of pairwise Comparisons

    print("C: %.5f" % C)
    print("Sigma: %.4f" % sigma)
    
    if C <= 0 or sigma <= 0:
        return np.ones(len(l)) * (-np.inf)  # If Parameters do not make sense return infinitely negative likelihood
    else:
        pr_noshare_lg = (-C * (r ** 2) / (4 * (l0 / 100) * sigma ** 2) * kv(2, np.sqrt(2.0 * l0 / 100.0) * r / sigma)) * pw_nr  # Standard vector of no-sharing
        f_share_lg = np.array([single_pair_dd(l[i], r[i], C, sigma) if (len(l[i]) > 0) else 1.0 for i in range(0, len(r))])  # For vector send it to single_pair
        power_noshare = power_block(l0)
        res = pr_noshare_lg * power_noshare + f_share_lg
        # print(res)
        print("Total log likelihood: %.3f" % np.sum(res))
        return res.astype(np.float) 
        
#########################################################################
    
class MLE_estimation(GenericLikelihoodModel):
    '''This is a class to do Maximum Likelihood mle_multi_run
    It needs the pairwise likelihood of the data'''
    def __init__(self, endog, exog, min_len=3.0, **kwds):
        super(MLE_estimation, self).__init__(endog, exog, **kwds)
        self.l0 = min_len  # In cM!
        
    def nloglikeobs(self, params):
        C = params[0]
        sigma = params[1]
        p_ll = pairwise_ll(self.endog, self.exog, self.l0, C, sigma)
        nll = -p_ll
        return nll

    def fit(self, start_params=None, maxiter=10000, maxfun=5000, **kwds):
        # we have one additional parameter and we need to add it for summary
        if start_params == None:
            start_params = np.array([0.1, 100.0])
        return super(MLE_estimation, self).fit(start_params=start_params,
                                     maxiter=maxiter, maxfun=maxfun,
                                     **kwds)
        
class MLE_estimation_growth(GenericLikelihoodModel):
    '''This is a class to do Maximum Likelihood mle_multi_run
    and incorporates exponential population growth
    It needs the pairwise likelihood of the data'''
    def __init__(self, endog, exog, min_len=3.0, **kwds):
        super(MLE_estimation_growth, self).__init__(endog, exog, **kwds)
        self.l0 = min_len  # In cM!
        
    def nloglikeobs(self, params):
        C = params[0]  # Absolute Parameter
        sigma = params[1]  # Dispersal parameter
        mu = params[2]  # Growth parameter
        p_ll = pairwise_ll_growth(self.endog, self.exog, self.l0, C, sigma, mu)
        nll = -p_ll
        return nll

    def fit(self, start_params=None, maxiter=10000, maxfun=5000, **kwds):
        # we have one additional parameter and we need to add it for summary
        if start_params == None:
            start_params = np.array([0.02, 75.0, 4.0])
        return super(MLE_estimation_growth, self).fit(start_params=start_params,
                                     maxiter=maxiter, maxfun=maxfun,
                                     **kwds)

class MLE_estimation_dd(GenericLikelihoodModel):  # Doomsday growth
    '''This is a class to do Maximum Likelihood mle_multi_run
    and incorporates exponential population growth
    It needs the pairwise likelihood of the data'''
    def __init__(self, endog, exog, min_len=3.0, **kwds):
        super(MLE_estimation_dd, self).__init__(endog, exog, **kwds)
        self.l0 = min_len  # In cM!
        
    def nloglikeobs(self, params):
        C = params[0]  # Absolute Parameter
        sigma = params[1]  # Dispersal parameter
        p_ll = pairwise_ll_dd(self.endog, self.exog, self.l0, C, sigma)
        nll = -p_ll
        return nll

    def fit(self, start_params=None, maxiter=10000, maxfun=5000, **kwds):
        # we have one additional parameter and we need to add it for summary
        if start_params == None:
            start_params = np.array([0.002, 75.0])
        return super(MLE_estimation_dd, self).fit(start_params=start_params,
                                     maxiter=maxiter, maxfun=maxfun,
                                     **kwds)
############################################################################


def power_block(x):
    '''Gives the power to detect block of length x. Works for vector as well.'''
    y = 1 - 1.0 / (1 + 0.077 * x * x * np.exp(0.54 * x))
    return y 

