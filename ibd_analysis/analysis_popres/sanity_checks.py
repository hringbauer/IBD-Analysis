# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 17:11:04 2017
@author: raphael
"""

import numpy as np
import scipy.sparse as sparse
from scipy.integrate import quad
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
from hetero_sharing import migration_matrix, ibd_sharing, map_projection
from scipy.special import kv as kv
from geopy.distance import vincenty

def G(t, x):
    return np.exp(-x ** 2 / (2 * t)) / np.sqrt(2 * np.pi * t)

def H(t, x):
    return np.abs(x) * np.exp(-x ** 2 / (2 * t)) / np.sqrt(2 * np.pi * t)

def f1(l, tau, t, x1, x2, y1, y2, beta, sigma1, sigma2):
    sigmay = .5 * (sigma1 + sigma2) + np.sign(y1) * .5 * (sigma1 - sigma2)
    return 2 * (1 + np.sign(y1) * beta) * H(tau, (np.max(y1, 0) + np.max(x1, 0)) / sigma1 + (1 + beta) * l / sigma1) * H(t - tau, (np.max(-y1, 0) + np.max(-x1, 0)) / sigma2 + (1 - beta) * l / sigma2) * G(tau * sigma1 ** 2 + (t - tau) * sigma2 ** 2, y2 - x2) / sigmay ** 2

def f2(tau, t, x1, x2, y1, y2, beta, sigma1, sigma2):
    return quad(f1, 0, np.inf, args=(tau, t, x1, x2, y1, y2, beta, sigma1, sigma2))[0]

def density(t, x1, x2, y1, y2, beta, sigma1, sigma2):
    sigmay = .5 * (sigma1 + sigma2) + np.sign(y1) * .5 * (sigma1 - sigma2)
    A = .5 * (np.sign(y1 * x1) + 1) * (G(sigmay ** 2 * t, x1 - y1) - G(sigmay ** 2 * t, x1 + y1)) * G(sigmay ** 2 * t, y2 - x2)
    B = quad(f2, 0, t, args=(t, x1, x2, y1, y2, beta, sigma1, sigma2))[0]
    return A + B

def one_dim_density(t, x, y, beta, sigma1, sigma2):
    sigmay = .5 * (sigma1 + sigma2) + np.sign(y) * .5 * (sigma1 - sigma2)
    sigmax = .5 * (sigma1 + sigma2) + np.sign(x) * .5 * (sigma1 - sigma2)
    gama = ((1 + beta) / sigma1 - (1 - beta) / sigma2) / ((1 + beta) / sigma1 + (1 - beta) / sigma2)
    return (G(t, x / sigmax - y / sigmay) + gama * np.sign(y) * G(t, np.abs(x) / sigmax + np.abs(y) / sigmay)) / sigmay

def variance_func(G):
    L = np.size(G, 0)
    X = np.tile(np.arange(L), (L, 1))
    Y = np.transpose(X)
    mean = np.array([np.sum(np.multiply(X, G), (0, 1)), np.sum(np.multiply(Y, G), (0, 1))])
    return np.sum(np.multiply((X - mean[0]) ** 2 + (Y - mean[1]) ** 2, G), (0, 1))

def gauss(t, x, y):
    return np.exp(-(x.astype(float) ** 2 + y.astype(float) ** 2) / (2 * t)) / (2 * np.pi * t)

def bessel(bin_lengths, distance, sigma, N, beta=0):
    return 2 ** (-1.5 * beta) * (distance / (sigma * np.sqrt(bin_lengths))) ** (2 + beta) * kv(2 + beta, np.sqrt(2 * bin_lengths) * distance / sigma) / (8 * N * np.pi * sigma ** 2)

def cylindrical(lat, lon):
    earth_radius = 6367.0
    lat = np.pi * lat / 180.0
    lon = np.pi * lon / 180.0
    mean_lat = np.mean(lat)
    X = earth_radius * lat * np.cos(mean_lat)
    Y = earth_radius * lon
    return np.column_stack((X, Y))
    
def test_projection(file_centroids='../country_centroids.csv'):
    ''' test for the winkel projection '''    
    centroids = np.loadtxt(file_centroids, dtype='string', delimiter=',')[1:, :]
    
    longitude = centroids[:, 2].astype('float')
    latitude = centroids[:, 1].astype('float')
    labels = centroids[:, 0]
    N = np.size(labels)
    
    pw_vincenty = np.zeros((N, N))
    for i in np.arange(N):
        for j in np.arange(i, N):
            pw_vincenty[i, j] = vincenty((latitude[i], longitude[i]), (latitude[j], longitude[j])).meters / 1000.0
    pw_vincenty = pw_vincenty[np.triu_indices(N, k=1)]
    
    #xy_positions = map_projection(np.column_stack((longitude, latitude)))
    #pw_dist = dist.pdist(xy_positions)
    
    plt.figure()
    #plt.scatter(pw_vincenty, pw_cylindrical, c='blue', alpha=.5, label='distances in cylindrical projection')
    #plt.scatter(pw_vincenty, pw_dist, c='red', label='distances in Winkel projection')
    plt.plot(pw_vincenty, pw_vincenty, c='black', label='great circle distances')
    plt.xlabel('Great circle distances computed by vincenty')
    plt.legend(loc="upper left")
    plt.show()
    
def display_populations(file_centroids='../country_centroids.csv', barrier=np.array([[1000, 5326], .06 * np.pi])):
    centroids = np.loadtxt(file_centroids, dtype='string', delimiter=',')[1:, :]
    
    longitude = centroids[:, 2].astype('float')
    latitude = centroids[:, 1].astype('float')
    labels = centroids[:, 0]
    N = np.size(labels)
    
    center = barrier[0]
    angle = barrier[1]
        
    xy_positions = map_projection(np.column_stack((longitude, latitude)))
    
    x = np.arange(400, 1500)
    y = np.tan(.5 * np.pi + angle) * (x - center[0]) + center[1]
    
    plt.figure()
    plt.scatter(xy_positions[:, 0], xy_positions[:, 1])
    for i in np.arange(N):
        plt.annotate(labels[i], (xy_positions[i, 0], xy_positions[i, 1]))
    plt.plot(x, y, c='red')
    
def test_1d(L=150, sigma=np.array([1, .5]), pop_sizes=np.array([1,1]), t=80, x0=-7):
    '''Test of analytic formula for the density of 1d skew Bm against law of first coordinate'''
    beta = ((sigma[1]*pop_sizes[1])**2 - (sigma[0]*pop_sizes[0])**2) / np.sum((sigma*pop_sizes)**2)
    mid = L / 2
    position = sparse.csc_matrix(([1], ([mid + x0 + L * mid], [0])), shape=(L ** 2, 1))
    
    X = np.tile(np.arange(L), (L, 1))
    Y = np.transpose(X)
    
    M = migration_matrix(L, sigma, pop_sizes)
    Green = position
    for _ in np.arange(t):
        Green = M * Green
    
    Green = Green.todense()
    Green = Green.reshape((L, L))
    
    marginal = np.array(np.sum(Green, 0))[0]  # Calculate the marginal Density
    
    skew_Bm = one_dim_density(t, x0, np.arange(L) - mid, beta, np.sqrt(sigma[1]), np.sqrt(sigma[0]))
    
    print("Sanity Check: Total sum Marg. density: %.4f" % np.sum(marginal))
    print("Sanity Check: Total sum Skew BM: %.4f" % np.sum(skew_Bm))
    print("Sum of absolute Differences: %.4g" % np.sum(np.abs(marginal - skew_Bm)))
    
    # Plot by Harald:
    x_vec = np.arange(L) - mid
    plt.figure()
    plt.plot(x_vec, marginal, color="cyan", alpha=0.8, label="Marginal Density: Simulated", linewidth=2)
    plt.plot(x_vec, skew_Bm, color="crimson", alpha=0.8, label="Skew B. Motion Theory", linewidth=2)
    plt.vlines(0, 0, max(skew_Bm), linewidth=2, label="Interface")
    plt.legend(loc="upper right")
    plt.xlabel("$\Delta$ x-Axis") 
    plt.ylabel("PDF")
    plt.show()

    # comparing variance of 2nd coordinate to expected variance of 2d skew Bm
    
    conditional = np.multiply(Green, 1 / marginal)
    
    variance = np.array(np.sum(np.multiply((Y - mid) ** 2, conditional), 0))[0]
    
    def conditional_variance(z, s, x, y, t, beta, sigma1, sigma2):
        sigmaz = .5 * (sigma1 + sigma2) + np.sign(z) * .5 * (sigma1 - sigma2)
        return sigmaz ** 2 * np.multiply(one_dim_density(s, x, z, beta, sigma1, sigma2), one_dim_density(t - s, z, y, beta, sigma1, sigma2)) / one_dim_density(t, x, y, beta, sigma1, sigma2)
    
    def f3(s, x, y, t, beta, sigma1, sigma2):
        return np.sum(conditional_variance(.5 * (x + y) + np.arange(L) - mid, s, x, y, t, beta, sigma1, sigma2))
    
    def expected_variance(x, y, t, beta, sigma1, sigma2):
        return quad(f3, 0, t, args=(x, y, t, beta, sigma1, sigma2))[0]
    
    true_variance = np.zeros(L)
    for y in np.arange(L):
        # print y
        true_variance[y] = expected_variance(x0, y - mid, t, beta, np.sqrt(sigma[1]), np.sqrt(sigma[0]))
    
    # displaying the relative difference in the variance, wheighted by the pdf the first coordinate
    print np.sum(marginal * np.abs(true_variance - variance) / variance)
    
    print("Calculate expected Variance!")
    plt.figure()
    plt.plot(np.arange(L), variance)
    plt.plot(np.arange(L), true_variance)
    plt.show()

def compare_bessel(sigma=np.array([1, 1]), d_e=np.array([1, 1]), beta=0):
    '''Test that IBD sharing in homogeneous case fits with bessel decay'''
    bin_lengths = np.arange(.04, 0.2, .001)

    # positions=np.array([[0,0],[0,5],[0,10],[0,15]])
    positions = np.array([[0, 0], [0, 10]])
    distance = np.sqrt(np.sum((positions[0, :] - positions[1, :]) ** 2))  # Calculate Pairwise Distance
    # distance=dist.pdist(positions)
    
    density = ibd_sharing(positions, bin_lengths, sigma, d_e, beta)
    density = density[:, 0, 1]
    # density = np.concatenate((density[0,0,1:4], density[0,1,2:4], density[0,2,3:4]))
    
    print("Simulated Decay: ")
    print density
    
    decay = bessel(bin_lengths, distance, np.mean(sigma), np.mean(d_e), beta=beta)
    
    print("Bessel Decay: ")
    print decay
    
    # plt.scatter(distance, density/decay)
    
    # diff=np.log(np.abs(density-decay))
    
    # plt.plot(bin_lengths, diff)
    
    plt.figure()
    plt.plot(bin_lengths, density, "ro", label="Simulated", linewidth=3, alpha=0.8)
    plt.plot(bin_lengths, decay, "bo", label="Bessel", linewidth=3, alpha=0.8)
    plt.legend()
    plt.yscale("log")
    plt.ylabel("Exp. Bl. Nr in dl / dl")
    plt.xlabel("Block Length [M]")
    plt.show()
    
def compare_bessel_distance(sigma=np.array([1, 1]), d_e=np.array([1, 1]), beta=0, L=0.05):
    '''Compare Bessel Decay against Distance'''
    # Calculate Position Vector
    positions = np.array([[0, i] for i in range(20)])
    # Calculate Pairwise Distances:
    distances = np.array([np.sqrt(np.sum((positions[0, :] - pos) ** 2)) for pos in positions])  
    
    # Calculate Density Vector:
    density = ibd_sharing(positions, np.array([L, ]), sigma, d_e, beta)
    dens0 = density[0, 0, :]  # Sharing with first Sample
    
    decay = bessel(L, distances, np.mean(sigma), np.mean(d_e), beta=beta)
    
    print("Simulated: ")
    print(dens0)
    print("Bessel:")
    print(decay)
    plt.figure()
    plt.plot(distances, dens0, "ro", label="Simulated", linewidth=3, alpha=0.8)
    plt.plot(distances, decay, "bo", label="Bessel", linewidth=3, alpha=0.8)
    plt.legend()
    plt.yscale("log")
    plt.ylabel("Exp. Bl. Nr in dl / dl")
    plt.xlabel("Distance")
    plt.show()
    
def compare_gaussian():
    '''Test that homogeneous migration converges to gaussian.
    Tests Green Function'''
    L = 100
    t = 120
    sigma = np.array([.2, .2])
    pop_sizes = np.array([1, 1])
    
    M = migration_matrix(L, sigma, pop_sizes)
    position = sparse.csc_matrix(([1], ([50 + L * 50], [0])), shape=(L ** 2, 1))
    
    G = position
    for _ in np.arange(t):
        G = M * G
    
    G = G.todense()
    G = G.reshape((L, L))
    
    print variance_func(G) / (2 * t)
    
    X = np.tile(np.arange(L), (L, 1))
    Y = np.transpose(X)
    
    continuous = gauss(sigma[0] * t, X - 50, Y - 50)
    
    print variance_func(continuous) / (2 * t)
    
    print("Relative Deviation Discrete and Gaussian: %.4f" % (np.max(np.abs(G - continuous)) / np.max(G)))



test_1d(L=200, t=150, pop_sizes=np.array([2, 1]))   # Compare marginal Density
# compare_bessel()  # Compare Bessel Decay with Block Length
# compare_bessel_distance()  # Compare Bessel Decay with Distance
# compare_gaussian()
# test_projection()
# display_populations()
