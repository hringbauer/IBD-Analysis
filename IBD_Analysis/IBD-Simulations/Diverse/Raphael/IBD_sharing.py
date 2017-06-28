# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 22:11:37 2017

@author: raphael
"""

import numpy as np
import scipy.sparse as sparse

'''
Creates a migration kernel of size L^2 x L^2 for a population living on a square grid of size L
sigma is a size 2 np.array containing the migration rates for the two regions
the function first creates a nearest-neighbour migration kernel, and then iterates it (parameter called 'iterates')
'''
def Migration_matrix(grid_size, sigma, iterates=1):
    L = grid_size + grid_size % 2 - 1  # make sure grid is odd
    mid = (L + 1) / 2
    sigma = np.maximum(sigma.astype(float), [0, 0])
    if (np.amax(sigma) >= 1):
        iterates = np.ceil(np.amax(sigma) / .45)
    sigma = sigma / iterates  # make sure sigma is true variance of migration
    
    diag1 = np.tile(np.concatenate((np.repeat(.5 * sigma, mid - 1), [0])), L)[:-1]  # horizontal migration
    diag2 = np.tile(np.repeat(.5 * sigma, mid)[:-1], L - 1)  # vertical migration
    M = sparse.diags([diag1, diag1, diag2, diag2], [1, -1, L, -L])
    M.setdiag(1 - np.array(M.sum(0))[0, ])  # probability of staying put
    M = M ** iterates
    
    return M

'''
Computes the IBD sharing density
position should contain the positions of samples on the grid, as np.array([[x1, y1], [x2, y2]]) etc
bin_lengths should be a np.array containing the different bin lengths
max_generations is the stopping point for the integral over generations back in time
the function returns an (l, k, k) array, where l is the nb of bin lengths and k the number of samples
'''
def IBD_sharing(positions, bin_lengths, sigma, population_sizes, grid_size, max_generation=500, iterates=1):
    L = grid_size + grid_size % 2 - 1  # make sure grid is odd
    # mid = (L+1)/2
    M = Migration_matrix(grid_size, sigma, iterates)  # create migration matrix
    bin_lengths = bin_lengths.astype(float)
    
    sample_size = np.size(positions) / 2
    coordinates = positions[:, 0] + L * positions[:, 1]  # convert (x,y) coordinates to (x+L*y) coordinates
    # Kernel will give the spread of ancestry on the grid at each generation back in time
    # it starts as a dirac mass at the sampling position
    Kernel = sparse.csc_matrix(([1] * sample_size, (coordinates, np.arange(sample_size))), shape=(L ** 2, sample_size))
    
    inv_pop_sizes = sparse.diags(np.repeat((1 / population_sizes.astype(float)), (L ** 2 + 1) / 2)[:-1], 0)
    
    coalescence = []
    density = np.zeros((np.size(bin_lengths), sample_size, sample_size))
    
    for t in np.arange(max_generation):  # sum over all generations
        coalescence = Kernel.transpose() * inv_pop_sizes * Kernel  # coalescence probability at generation t
        blocks = 4 * t ** 2 * np.exp(-2.0 * bin_lengths * t)  # number of blocks of the right length at generation t
        density += np.multiply(coalescence.toarray(), blocks[:, np.newaxis, np.newaxis])  # multiply the two
        Kernel = M * Kernel  # update the kernel
    
    return density
