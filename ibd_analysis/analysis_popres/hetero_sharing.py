# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 22:11:37 2017
@author: raphael
"""

import numpy as np
import scipy.sparse as sparse
import scipy.spatial.distance as dist
from scipy.stats import hmean
import matplotlib.pyplot as plt
from migration_matrix import migration_matrix

def ibd_sharing(coordinates, L, step, bin_lengths, sigma, population_sizes, pw_growth_rate=0, balance='isotropic',
                max_generation=200):
    '''
    Compute the IBD sharing density.
    positions: Should contain the positions of samples on the grid as np.array([[x1, y1], [x2, y2]]) etc
    bin_lengths: should be a np.array containing the different bin lengths
    max_generations is the stopping point for the integral over generations back in time
    G: Length of the Genome (in Morgan)
    grid_max: Maximum Size of the Grid.
    Returns an (l, k, k) array, where l is the nb of bin lengths and k the number of samples
    '''
    mid = L / 2
    # print L, step, sigma
    # print "Creating migration matrix..."
    M = migration_matrix(L, np.concatenate(((sigma / step) ** 2, population_sizes)), balance)  # create migration matrix
    # print "migration matrix created."
    # print step**2*variance(M[mid+mid*L,:].todense().reshape((L,L)))
    bin_lengths = bin_lengths.astype(float)
    
    # sample_size = np.size(coordinates, 0)
    sample_size = np.size(coordinates, 1) 
    
    # Kernel will give the spread of ancestry on the grid at each generation back in time
    Kernel = coordinates
    
    inv_pop_sizes = sparse.diags(np.tile(np.repeat(.5 / population_sizes.astype(float), mid), L))
    # print inv_pop_sizes.todense()
    
    coalescence = []
    density = np.zeros((np.size(bin_lengths), sample_size, sample_size))
    
    for t in np.arange(max_generation):  # sum over all generations
        # print("Generation: %i" % t)
        coalescence = Kernel.transpose() * inv_pop_sizes * Kernel  # coalescence probability at generation t
        blocks = 4 * t ** (2 + pw_growth_rate) * np.exp(-2.0 * bin_lengths * t)  # number of blocks of the right length at generation t
        density += np.multiply(coalescence.toarray(), blocks[:, np.newaxis, np.newaxis])  # multiply the two
        Kernel = M * Kernel  # update the kernel
    
    # print "Computing of IBD sharing complete."
    # need to divide by step**2 in the Discretisation of the spatial integral
    return density / step ** 2

def prepare_coordinates(longitudes, latitudes, prior_sigma, coarse=.1):
    cartesian = map_projection(longitudes, latitudes)
    step, L = grid_fit(cartesian, prior_sigma, coarse)
    L = L + L % 2
    # print step, L
    coordinates = barycentric_coordinates(cartesian, L, step, L / 2)
    return coordinates, step, L

def prepare_step_size(cartesian, prior_sigma, coarse=.1):
    '''Calculates the step size given some coordinates.
    Cartesian: nx2 Array of cartesian coordinates'''
    step, L = grid_fit(cartesian, prior_sigma, coarse)
    L = L + L % 2
    coordinates = barycentric_coordinates(cartesian, L, step, L / 2)
    return coordinates, step, L

def prepare_coordinates_new(coordinates, barrier_location, prior_sigma, coarse=.1, projection=False,
                            step=0, L=0):
    '''
    calculates the step size and prepares the coordinates for inference
    if projection=True, coordinates should be longitudes-latitudes and will be projected
    '''
    if (projection == True):
        coordinates = map_projection(coordinates[:, 0], coordinates[:, 1])
    
    coordinates = centering_positions(coordinates, barrier_location)
    step1, L1 = grid_fit(coordinates, prior_sigma, coarse)
    
    # Overwrite Discretization Values if given.
    if L > 0:
        L1 = L     
    if step > 0:
        step1 = step
        
    L1 = L1 + L1 % 2
    coordinates = barycentric_coordinates(coordinates, L1, step1, L1 / 2)
    return coordinates, step1, L1

def plot_barycentric_coordinates(coordinates):
    L = np.sqrt(coordinates.shape[0])
    X = np.tile(np.arange(L), L)
    Y = np.repeat(np.arange(L), L)
    
    positions_x = np.sum(np.multiply(X, np.transpose(coordinates.todense())), axis=1)
    positions_y = np.sum(np.multiply(Y, np.transpose(coordinates.todense())), axis=1)
    
    plt.scatter(positions_x, positions_y)
    plt.show()

def sharing_density(bin_lengths, coordinates, L, step, parameters):
    sigma = parameters[0:2]
    population_sizes = parameters[2:4]
    pw_growth_rate = parameters[4]
    return ibd_sharing(coordinates, L, step, bin_lengths, sigma, population_sizes, pw_growth_rate)

def grid_fit(positions, sigma, coarse=.25, max_iterate=10):
    '''
    Find optimal spatial discretization for the computation
    '''
    # 1/coarse sets the mean number of grid points between pairs of samples
    # (harmonic mean gives more weight to close pairs)
    # max_iterates is the maximum number of times we will have to iterate the matrix M
    step = np.maximum(coarse * hmean(dist.pdist(positions)), np.max(sigma) / np.sqrt(.45 * max_iterate))
    # step=100  # To overwrite step for the moment for testing.
    # take L large enough that all points are at least 10 sigmas or at least 10 squares from the edges
    L = 2 * np.int(np.ceil((np.maximum(10 * np.max(sigma), 10 * step) + np.max(np.abs(positions), (0, 1))) / step))
    return step, L

def barycentric_coordinates(positions, L, step, offset):
    '''
    Convert coordinates in R^2 to barycentric coordinates in Z^2
    '''
    sample_size = np.size(positions, 0)
    left = np.floor(positions.astype(float) / step)  # position of lower left point closest to sampling position
    alpha = positions / step - left  # weight of points
    
    left = left + offset
    low_left = left[:, 0] + L * left[:, 1]  # convert (x,y) coordinates to (x+L*y) coordinates
    coordinates = np.concatenate((low_left, low_left + 1, low_left + L, low_left + L + 1))
    weights = np.concatenate(((1 - alpha[:, 0]) * (1 - alpha[:, 1]), alpha[:, 0] * (1 - alpha[:, 1]), (1 - alpha[:, 0]) * alpha[:, 1], alpha[:, 0] * alpha[:, 1]))
    
    bary_coordinates = sparse.csc_matrix((weights, (coordinates, np.tile(np.arange(sample_size), 4))), shape=(L ** 2, sample_size))
    return bary_coordinates

def centering_positions(positions, barrier_location):
    '''
    Change of coordinates so that the barrier is at x=0
    '''
    center = barrier_location[0]
    angle = barrier_location[1]
    c = np.cos(angle)
    s = np.sin(angle)
    rotation_matrix = np.array([[c, -s], [s, c]])
    return np.matmul(positions - center, rotation_matrix)

def map_projection(lon_vec, lat_vec):
    '''
    Winkel projection with standard parallel at mean latitude of the sample
    argument is (n,2) array with longitude as first column and latitude as second column
    returns cartesian coordinates in kilometers
    '''
    lon_lat_positions = np.column_stack((lon_vec, lat_vec))
    earth_radius = 6367.0  # radius at 46N
    lon_lat_positions = np.pi * lon_lat_positions / 180.0  # convert to radian
    mean_lat = np.mean(lon_lat_positions[:, 1])
    X = earth_radius * lon_lat_positions[:, 0] * .5 * (np.cos(mean_lat) + np.cos(lon_lat_positions[:, 1]))
    Y = earth_radius * lon_lat_positions[:, 1]
    return np.column_stack((X, Y))

if __name__ == "__main__":
    coords = np.array([[15, 10], [15, 15], [10, 10], [10, 15], [20, 20]])
    print(coords)
    step, L = grid_fit(coords, sigma=1, coarse=0.1)
    print(step)
    print(L)
    bc = barycentric_coordinates(coords, L, step, L / 2)
    # bc = barycentric_coordinates(coords, 100, 5.0, 50)
    print(bc)
    sample_size = np.size(bc, 1)
    print(sample_size)
    print ibd_sharing(bc, L, step, np.array([0.05, 0.01]), sigma=np.array([1.0, 1.0]), population_sizes=np.array([10, 10]), pw_growth_rate=0,
                max_generation=200)
    # print ibd_sharing(bc, L, step, np.array([0.05, 0.01]), sigma=np.array([1.0, 1.0]), population_sizes=np.array([10, 10]), pw_growth_rate=0,
    #            max_generation=200)
    
    

