# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 11:30:15 2017
This class contains the Migration Matrices.
@author: raphael
"""

import numpy as np
import scipy.sparse as sparse

def migration_matrix(L, parameters, balance='homogeneous', iterates=1, custom_func=None):
    '''Return the Migration Matrix. 
    balance: What mode of the migration matrix to use
    custom_func: Give custom function'''
    assert(len(parameters)==4)
    sigma2 = parameters[0:2].astype(float)
    pop_sizes = parameters[2:4].astype(float)
    
    sigma2 = np.maximum(sigma2.astype(float), [0, 0])
    if (np.amax(sigma2) >= .5):
        iterates = np.ceil(np.amax(sigma2) / .45)
    sigma2 = sigma2 / iterates  # make sure sigma2 is true variance of migration
    
    if (balance=='isotropic'):
        assert(L%2==0)
        mid=L/2
        M = migration_forward_isotropic(L, sigma2)
    elif (balance=='symmetric'):
        assert(L%2==1)
        mid=(L+1)/2
        M = migration_forward_symmetric(L, sigma2)
    elif (balance=='barrier'):
        assert(L%2==0)
        mid=(L+1)/2
        raise NotImplementedError("Implement Barrier Migration Matrix!!")
        #M = migration_forward_barrier(L, sigma2)
    elif balance=='homogeneous':
        #print 'homogeneous'
        M = migration_forward_homogeneous(L, sigma2)
        mid = (L+L%2)/2
        
    elif balance=='custom':
        M = custom_func(L, parameters)
        
    else:
        raise ValueError("Enter Valid Migration Mode!!")
    
    # if (L%2==0):
    #     population_sizes = sparse.diags(np.tile(np.repeat(pop_sizes, mid), L))
    #     inv_pop_sizes = sparse.diags(np.tile(np.repeat(1/pop_sizes, mid), L))
    # else:
    #     population_sizes = sparse.diags(np.tile(np.repeat(pop_sizes, mid)[:-1], L))
    #     inv_pop_sizes = sparse.diags(np.tile(np.repeat(1/pop_sizes, mid)[:-1], L))
    
    # M = inv_pop_sizes*M.transpose()*population_sizes
    M.setdiag(1 - np.array(M.sum(0))[0, ])
    
    return M ** iterates

def migration_forward_isotropic(L, sigma2):
    # create the forward migration matrix
    mid = L / 2
    horizontal_left = np.concatenate((np.repeat(.5 * sigma2, mid)[1:], [0]))  # horizontal migration to the left
    horizontal_right = np.concatenate((np.repeat(.5 * sigma2, mid)[:-1], [0]))  # horizontal migration to the right
    vertical = np.repeat(.5 * sigma2, mid)
    
    diag_left = np.tile(horizontal_left, L)[:-1]
    diag_right = np.tile(horizontal_right, L)[:-1]
    diag_vert = np.tile(vertical, L - 1)
    return sparse.diags([diag_left, diag_right, diag_vert, diag_vert], [1, -1, L, -L])

def migration_forward_symmetric(L, sigma2):
    mid = (L+1)/2
    
    horizontal = np.tile(np.concatenate((np.repeat(.5 * sigma2, mid-1), [0])), L)[:-1]
    vertical = np.tile(np.repeat(.5 * sigma2, mid)[:-1], L-1)
    return sparse.diags([horizontal, horizontal, vertical, vertical], [1, -1, L, -L])

def migration_forward_homogeneous(L, sigma2):
    sigma2=sigma2[0]
    horizontal = np.tile(np.concatenate((np.repeat(.5*sigma2, L)[:-1], [0])), L)[:-1]
    vertical = np.tile(np.repeat(.5 * sigma2, L), L-1)
    return sparse.diags([horizontal, horizontal, vertical, vertical], [1, -1, L, -L])
