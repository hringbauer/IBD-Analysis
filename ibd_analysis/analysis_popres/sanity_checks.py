# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 17:11:04 2017
@author: raphael
"""

import numpy as np
import scipy.sparse as sparse
from scipy.integrate import quad
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
from hetero_sharing import migration_matrix, ibd_sharing
from scipy.special import kv as kv

def G(t,x):
    return np.exp(-x**2/(2*t))/np.sqrt(2*np.pi*t)

def H(t,x):
    return np.abs(x)*np.exp(-x**2/(2*t))/np.sqrt(2*np.pi*t)

def f1(l, tau, t, x1, x2, y1, y2, beta, sigma1, sigma2):
    sigmay = .5*(sigma1+sigma2) + np.sign(y1)*.5*(sigma1-sigma2)
    return 2*(1+np.sign(y1)*beta)*H(tau, (np.max(y1,0)+np.max(x1,0))/sigma1 + (1+beta)*l/sigma1)*H(t-tau, (np.max(-y1, 0)+np.max(-x1,0))/sigma2 + (1-beta)*l/sigma2)*G(tau*sigma1**2 + (t-tau)*sigma2**2,y2-x2)/sigmay**2

def f2(tau, t, x1, x2, y1, y2, beta, sigma1, sigma2):
    return quad(f1, 0, np.inf, args=(tau, t, x1, x2, y1, y2, beta, sigma1, sigma2))[0]

def density(t, x1, x2, y1, y2, beta, sigma1, sigma2):
    sigmay = .5*(sigma1+sigma2) + np.sign(y1)*.5*(sigma1-sigma2)
    A = .5*(np.sign(y1*x1)+1)*(G(sigmay**2*t,x1-y1) - G(sigmay**2*t,x1+y1))*G(sigmay**2*t,y2-x2)
    B = quad(f2, 0, t, args=(t, x1, x2, y1, y2, beta, sigma1, sigma2))[0]
    return A+B

def one_dim_density(t, x, y, beta, sigma1, sigma2):
    sigmay = .5*(sigma1+sigma2) + np.sign(y)*.5*(sigma1-sigma2)
    sigmax = .5*(sigma1+sigma2) + np.sign(x)*.5*(sigma1-sigma2)
    gama = ((1+beta)/sigma1 - (1-beta)/sigma2)/((1+beta)/sigma1 + (1-beta)/sigma2)
    return (G(t,x/sigmax-y/sigmay) + gama*np.sign(y)*G(t,np.abs(x)/sigmax + np.abs(y)/sigmay))/sigmay

def variance_func(G):
    L=np.size(G,0)
    X=np.tile(np.arange(L),(L,1))
    Y=np.transpose(X)
    mean=np.array([np.sum(np.multiply(X,G),(0,1)), np.sum(np.multiply(Y,G),(0,1))])
    return np.sum(np.multiply((X-mean[0])**2+(Y-mean[1])**2,G),(0,1))

def gauss(t, x, y):
    return np.exp(-(x.astype(float)**2+y.astype(float)**2)/(2*t))/(2*np.pi*t)

def bessel(bin_lengths, distance, sigma, N, beta=0):
    return 2**(-1.5*beta)*(distance/(sigma*np.sqrt(bin_lengths)))**(2+beta)*kv(2+beta,np.sqrt(2*bin_lengths)*distance/sigma)/(8*N*np.pi*sigma**2)

def test_1d(L=151, sigma=np.array([1,.5]), t=80, x0=-7):
    '''Test of analytic formula for the density of 1d skew Bm against law of first coordinate'''
    beta=(sigma[1]-sigma[0])/np.sum(sigma)
    mid=(L-1)/2
    position=sparse.csc_matrix(([1], ([mid+x0+L*mid], [0])), shape=(L**2, 1))
    
    X=np.tile(np.arange(L),(L,1))
    Y=np.transpose(X)
    
    M=migration_matrix(L, sigma)
    Green=position
    for _ in np.arange(t):
        Green=M*Green
    
    Green=Green.todense()
    Green=Green.reshape((L,L))
    
    marginal=np.array(np.sum(Green,0))[0] # Calculate the marginal Density
    
    skew_Bm=one_dim_density(t, x0, np.arange(L)-mid, beta, np.sqrt(sigma[1]), np.sqrt(sigma[0]))
    
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
    
    conditional=np.multiply(Green,1/marginal)
    
    variance=np.array(np.sum(np.multiply((Y-mid)**2,conditional),0))[0]
    
    def conditional_variance(z, s, x, y, t, beta, sigma1, sigma2):
        sigmaz = .5*(sigma1+sigma2) + np.sign(z)*.5*(sigma1-sigma2)
        return sigmaz**2*np.multiply(one_dim_density(s, x, z, beta, sigma1, sigma2),one_dim_density(t-s, z, y, beta, sigma1, sigma2))/one_dim_density(t, x, y, beta, sigma1, sigma2)
    
    def f3(s, x, y, t, beta, sigma1, sigma2):
        return np.sum(conditional_variance(.5*(x+y) + np.arange(L)-mid, s, x, y, t, beta, sigma1, sigma2))
    
    def expected_variance(x, y, t, beta, sigma1, sigma2):
        return quad(f3, 0, t, args=(x, y, t, beta, sigma1, sigma2))[0]
    
    true_variance=np.zeros(L)
    for y in np.arange(L):
        # print y
        true_variance[y] = expected_variance(x0, y-mid, t, beta, np.sqrt(sigma[1]), np.sqrt(sigma[0]))
    
    # displaying the relative difference in the variance, wheighted by the pdf the first coordinate
    print np.sum(marginal*np.abs(true_variance-variance)/variance)
    
    print("Calculate expected Variance!")
    plt.figure()
    plt.plot(np.arange(L), variance)
    plt.plot(np.arange(L), true_variance)
    plt.show()

def compare_bessel():
    '''Test that IBD sharing in homogeneous case fits with bessel decay'''
    bin_lengths=np.arange(.01, 2, .001)
    #bin_lengths=np.array([.1])
    sigma=np.array([1,1])
    d_e = np.array([1,1])
    beta=1
    #positions=np.array([[0,0],[0,5],[0,10],[0,15]])
    positions=np.array([[0,0],[0,8]])
    distance=np.sqrt(np.sum((positions[0,:]-positions[1,:])**2))
    #distance=dist.pdist(positions)
    
    density=ibd_sharing(positions, bin_lengths, sigma, d_e, beta)
    density=density[:,0,1]
    #density = np.concatenate((density[0,0,1:4], density[0,1,2:4], density[0,2,3:4]))
    
    print("Simulated Decay: ")
    print density
    
    decay=bessel(bin_lengths, distance, np.mean(sigma), np.mean(d_e))
    
    print("Bessel Decay: ")
    print decay
    
    #plt.scatter(distance, density/decay)
    
    #diff=np.log(np.abs(density-decay))
    
    #plt.plot(bin_lengths, diff)
    
    plt.figure()
    plt.plot(bin_lengths, density, "ro", label="Simulated", linewidth=3, alpha=0.8)
    plt.plot(bin_lengths, decay, "bo", label="Bessel", linewidth=3, alpha=0.8)
    plt.legend()
    plt.yscale("log")
    plt.ylabel("Exp. Bl. Nr in dl / dl")
    plt.xlabel("Block Length [M]")
    plt.show()
    
def compare_gaussian():
    '''Test that homogeneous migration converges to gaussian'''
    L=101
    t=120
    sigma=np.array([.2,.2])
    
    M=migration_matrix(L,sigma)
    position=sparse.csc_matrix(([1], ([50+L*50], [0])), shape=(L**2, 1))
    G=position
    for _ in np.arange(t):
        G=M*G
    
    G=G.todense()
    G=G.reshape((L,L))
    
    print variance_func(G)/(2*t)
    
    X=np.tile(np.arange(L),(L,1))
    Y=np.transpose(X)
    
    continuous=gauss(sigma[0]*t,X-50,Y-50) # Two Dimensional Function
    
    print variance_func(continuous)/(2*t)
    
    print("Relative Deviation Discrete and Gaussian: %.4f" % (np.max(np.abs(G-continuous))/np.max(G)))




# test_1d()
# compare_bessel()
compare_gaussian()
