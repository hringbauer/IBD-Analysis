'''
Created on Mar 19, 2015
Package similar to main, but with the purpose to do multiple runs for statistical analysis and save the data.
Parameters are taken from grid.py and mle_estim_error.py!
@author: Harald Ringbauer
'''

from grid import factory_Grid
from analysis import Analysis, torus_distance
from random import shuffle
from itertools import combinations
from bisect import bisect_right
from math import pi
from scipy.special import kv as kv  # Import Bessel functions of second kind

import cPickle as pickle
import numpy as np
import matplotlib.pyplot as plt

t = 200  # Generation time for a single run #t=200
nr_runs = 20  # How many runs
sample_sizes = (100, 270, 440, 625)
distances = [[2, 10], [10, 20], [20, 30], [30, 40], [40, 50], [50, 60]]  # Distances to use for binning
# distances = [[1, 5], [5, 10], [10, 15], [15, 20], [20, 25], [25, 30]]  # Distances to use for binning
intervals = [[4, 5], [5, 6.5], [6.5, 8], [8, 12]]  # Bins for the block length binning

def single_run():
    ''' Do a single run, parameters are saved in grid'''
    grid = factory_Grid()  # Set grid
    grid.reset_grid()  # Delete everything
    grid.set_samples()  
    grid.update_t(t)  # Do the actual run
    
    data = Analysis(grid)  # Do Data-Analysis
    data.fit_expdecay(show=False)
    sigma = data.sigma_estimate
    block_nr = len(grid.IBD_blocks)   
    return(sigma, block_nr)  
           
def analysis_run():
    grid = factory_Grid()
    parameters = (grid.sigma, grid.gridsize, grid.sample_steps, grid.dispmode)
    results = np.zeros((nr_runs, 2))  # Container for the data
    
    '''Runs the statistical analysis'''
    for i in range(0, nr_runs):
        print("Doing run: %.1f" % i)
        results[i, :] = single_run()  # Do the run and save the results 
    
    print("RUN COMPLETE!!")
    pickle.dump((results, parameters), open("Data1/stats_demes.p", "wb"))  # Pickle the data
    # save_name=raw_input("Save to what filename?")  
    # pickle.dump((results, parameters), open(save_name, "wb"))  # Pickle the data  
    print("SAVED")

def run_var_samp(save_name):
    '''Runs simulations for various sample sizes and saves estimates and parameters.'''
    grid = factory_Grid()
    results = np.zeros((len(sample_sizes), nr_runs, 2))  # Container for the data
    sample_steps = grid.sample_steps 
    
    '''Actual runs:'''
    row = 0
    for k in sample_sizes:
        position_list = [(i + sample_steps / 2, j + sample_steps / 2, 0) for i in range(0, grid.gridsize, sample_steps) for j in range(0, grid.gridsize, sample_steps)]
        # position_list = [(i + sample_steps / 2, j  + sample_steps / 2, 0) for i in range(0, sample_steps * k, sample_steps) for j in range(0, sample_steps * k, sample_steps)]


        for i in range(0, nr_runs):
            print("Doing run: %.1f for %.0f samples" % (i, k))
            grid.reset_grid()  # Delete everything
            shuffle(position_list)  # Randomize position List
            print(position_list)
            grid.set_chromosome(position_list[:k])  # Set the samples
            grid.update_t(t)  # Do the actual run
            if grid.dispmode == "demes":
                grid.update_IBD_blocks_demes(5)  # Update position for deme analysis!!
    
            data = Analysis(grid)  # Do Data-Analysis
            data.fit_expdecay(show=False)
            sigma = data.sigma_estimate
            block_nr = len(grid.IBD_blocks)
            results[row, i, :] = (sigma, block_nr)
        row += 1  # Go one down in the results_row
            
        print("RUN COMPLETE!!")
    parameters = (grid.sigma, grid.gridsize, sample_sizes, grid.dispmode)
    pickle.dump((results, parameters), open(save_name, "wb"))  # Pickle the data
    print("SAVED")   
    
def empirical_IBD_list(save_name):
    '''Generate empirical IBD-list. Nr. of run times'''
    results = []  # Container for the data
    
    '''Actual runs:'''
    for i in range(nr_runs):
        print("Doing run: %i" % i)
        grid = factory_Grid(growing=1)  # No growing grid
        grid.reset_grid()
        grid.set_samples()
        grid.update_t(t)  # Do the actual run
        # if grid.dispmode == "demes":
            # grid.update_IBD_blocks_demes(5)  # Update position for deme analysis!!
        pair_dist, pair_IBD, pair_nr = grid.give_lin_IBD(bin_pairs=True)  # Get the binned IBD-lists
        results.append([pair_dist, pair_IBD, pair_nr])  # Save the empirical data
        
    parameters = (grid.sigma, grid.gridsize, grid.sample_steps, grid.dispmode)
    pickle.dump((results, parameters), open(save_name, "wb"))  # Pickle the data
    print("SAVED")   

def get_normalization_factor(dist_bins, grid_size, sample_steps):
    '''Calculates Normalization factor for binned distances and starting grid'''
    position_list = [(i + sample_steps / 2, j + sample_steps / 2, 0) for i in range(0, grid_size, sample_steps) for j in range(0, grid_size, sample_steps)]
    k = len(position_list)
    print(k * (k - 1) / 2)
    distance_bins = np.zeros(len(dist_bins) - 1)  # Create bins for every element in List; len(bins)=len(counts)+1
    dist_bins[-1] += 0.000001  # Hack to make sure that the distance exactly matching the max are counted
                
    # Calculate Distance for every possible pair to get proper normalization factor:
    for (x, y) in combinations(np.arange(len(position_list)), r=2):
        dist = torus_distance(position_list[x][0], position_list[x][1], position_list[y][0], position_list[y][1], grid_size)   
        j = bisect_right(dist_bins, dist)
        if j < len(dist_bins) and j > 0:  # So it actually falls into somewhere
            distance_bins[j - 1] += 1
    return distance_bins

def get_normalization_lindata(dist_bins, pair_dist, pair_nr):
    '''Gets the Nr. of pairs in each distance bin from linearized IBD-data'''
    bl_nr = np.zeros(len(dist_bins))  # Creates zero array
    for i in range(len(dist_bins)):
        dist = dist_bins[i]
        bl_nr[i] += np.sum([pair_nr[j] for j in range(len(pair_dist)) 
                if dist[0] <= pair_dist[j] < dist[1]])
    return bl_nr
                    
def into_bins(pair_dist, pair_IBD, intervals, distances):
    '''Return Matrix of blocks Nr in each interval and distance bin.'''
    res = np.zeros((len(intervals), len(distances))).astype(np.int)  # Create empty Int-Array.
    for i in range(len(intervals)):
        for j in range(len(distances)):
            intv = intervals[i]
            dist = distances[j]
            for d in range(len(pair_dist)):  # Iterate over every pair
                if dist[0] <= pair_dist[d] < dist[1]:
                    res[i, j] += np.sum([1 for block in pair_IBD[d] if (intv[0] <= block <= intv[1])])
    return np.array(res)  # Return numpy array

def bessel_longer_l0(sigma, r, l0, b, D):
    '''The formula for block sharing longer than l0. Measured in cM
    If r vector return vector. b is growth rate parameter'''
    l0 = l0 / 100.0  # Change to Morgan
    G = 1.5  # Chromosome-Length
    C = G / (D * pi * sigma ** 2)  # First calculate the constant; D_e=1
    y = C * 2 ** ((-5 - 3 * b) / 2.0) * (r / (np.sqrt(l0) * sigma)) ** (1 + b) * kv(1 + b, np.sqrt(2.0 * l0) * r / sigma)
    # y = C * r / (sigma * np.sqrt(2 * l0)) * kv(1, np.sqrt(2.0 * l0) * r / sigma)
    return y

def bessel_l(sigma, r, l, b, D):
    '''Plots the exact Besseldecay for a block of length l. If r vector return  vector'''
    l = l / 100.0  # Change to Morgan
    G = 1.5  # Chromosome-Length
    C = G / (D * pi * sigma ** 2)  # First calculate the constant; D_e=1
    y = C * 2 ** (-3 - 3 * b / 2.0) * (r / (np.sqrt(l) * sigma)) ** (2 + b) * kv(2 + b, np.sqrt(2.0 * l) * r / sigma)
    # y = C * r / (sigma * np.sqrt(2 * l0)) * kv(1, np.sqrt(2.0 * l0) * r / sigma)
    return (y / 100.0)  # Return density in cM
    
def get_theory_sharing(intervals, distances, sigma, b, D):
    '''Gives back the theory sharing for the given Distance- and 
    block length intervals'''
    res = np.zeros((len(intervals), len(distances)))  # Array for results
    for i in range(len(intervals)):
        l = intervals[i]
        int_len = l[1] - l[0]
        for j in range(len(distances)):
            d1 = distances[j]
            dists = np.linspace(d1[0], d1[1], 20)  # 20 Intervals for calculation
            
            shr_bin = bessel_longer_l0(sigma, dists, l[0], b, D) - bessel_longer_l0(sigma, dists, l[1], b, D)  # Get the sharing per length bin
            res[i, j] = np.mean(shr_bin) / int_len  # Mean sharing per cM averaged over dist bins
    return res
       
def analyze_emp_IBD_list(save_name, show=True, b=0, D=1): 
    '''Plots summary of the empirical-IBD-list'''  
    (results, parameters) = pickle.load(open(save_name, "rb"))  # Data2/file.p
    print(parameters)
    print(len(results))
    sigma, _, _, _ = parameters
    dist_means = np.array([np.mean(i) for i in distances])  # Mean distances
    
    
    bl_nr = np.zeros((len(results), len(intervals), len(distances))).astype(np.int)  # Container for block matrices
    for i in range(len(results)):
        pair_dist, pair_IBD, pair_nr = results[i]  # Unpack the result arrays.
        # bl_list = results[i]
        # Generate Block-Info
        # block_info = [[b[1], torus_distance(b[2][0], b[2][1], b[3][0], b[3][1], grid_size)] for b in bl_list]
        bl_nr[i, :, :] = into_bins(pair_dist, pair_IBD, intervals, distances)  # Get the empirical data matrix
    # print(pair_dist)   
    # print(pair_nr) 

    nr_pairs_bins = get_normalization_lindata(distances, pair_dist, pair_nr)  # Get Nr. factor: Pairs of inds
    int_len = np.array([i[1] - i[0] for i in intervals])  # Length of the intervals - to normalize for that
    mean_shr = np.mean(bl_nr, axis=0)
    sts = np.std(bl_nr, axis=0)
    emp_shr = mean_shr / nr_pairs_bins
    emp_shr = emp_shr / int_len[:, None]
    emp_sts = sts / nr_pairs_bins
    emp_sts = emp_sts / int_len[:, None]
    
    # print(emp_shr)
    
    thr_shr = get_theory_sharing(intervals, distances, sigma, b, D)  # Get predicted sharing
    # print(thr_shr)
     
    '''Now do the plotting'''
    if show == True:
        f, axarr = plt.subplots(2, 2, sharex=True, sharey=True)  # Create sub-plots
            
        for i in range(4):  # Loop through interval list
            curr_plot = axarr[i / 2, i % 2]  # Set current plot
            curr_plot.set_yscale('log')  # Set Log-Scale
            interval = intervals[i] 
            # int_len = interval[1] - interval[0]  # Calculate the length of an interval
            curr_plot.set_ylim([1.0 / 10 ** 6, 1.0 / 10])
            curr_plot.set_xlim([0, 70])
                    
            l2, = curr_plot.semilogy(dist_means, thr_shr[i, :], 'r-.', linewidth=2)  # Plot of exact fit
            l1 = curr_plot.errorbar(dist_means, emp_shr[i, :], yerr=emp_sts[i, :], fmt='bo')
                # curr_plot.set_ylim([min(y) / 3, max(y) * 3])
            curr_plot.set_title("Interval: " + str(interval) + " cM")
            
            # curr_plot.annotate("Blocks: %.0f" % self.total_bl_nr, xy=(0.4, 0.7), xycoords='axes fraction', fontsize=16)
        f.text(0.5, 0.02, r'Distance [$\sigma$]', ha='center', va='center', fontsize=20)
        f.text(0.025, 0.5, 'IBD-blocks per pair and cM', ha='center', va='center', rotation='vertical', fontsize=20)
        f.legend((l1, l2), ('Simulated IBD-sharing', 'Theory'), loc=(0.7, 0.36))
        plt.tight_layout()
    
        plt.show()
    dist_means = dist_means / sigma  # Normalize The distance to Dispersal units
    return((dist_means, thr_shr, emp_shr, emp_sts))

def analyze_mult_emp_lists(save_names):
    emp_shr = []
    emp_sts = []
    for name in save_names:  # Load the data
        dist_means, thr_shr, shr, sts = analyze_emp_IBD_list(name, show=False)
        emp_shr.append(shr)
        emp_sts.append(sts)
     
       
    # Do the actual plot: (Multiple Windows)
#     f, axarr = plt.subplots(2, 2, sharex=True, sharey=True)  # Create sub-plots
#     for i in range(4):  # Loop through interval list
#         curr_plot = axarr[i / 2, i % 2]  # Set current plot
#         curr_plot.set_yscale('log')  # Set Log-Scale
#         interval = intervals[i] 
#         curr_plot.set_ylim([1.0 / 10 ** 6, 1.0 / 10])
#         curr_plot.set_xlim([0, 30])
#                 
#         l0, = curr_plot.semilogy(dist_means, thr_shr[i, :], 'r-', linewidth=2)  # Plot of exact fit
#         l1 = curr_plot.errorbar(dist_means - 1, emp_shr[0][i, :], yerr=emp_sts[0][i, :], fmt='yo', linewidth=2)
#         l2 = curr_plot.errorbar(dist_means - 0.5, emp_shr[1][i, :], yerr=emp_sts[1][i, :], fmt='go', linewidth=2)
#         l3 = curr_plot.errorbar(dist_means, emp_shr[2][i, :], yerr=emp_sts[2][i, :], fmt='ko', linewidth=2)
#         l4 = curr_plot.errorbar(dist_means + 0.5, emp_shr[3][i, :], yerr=emp_sts[3][i, :], fmt='mo', linewidth=2)
#         l5 = curr_plot.errorbar(dist_means + 1, emp_shr[4][i, :], yerr=emp_sts[4][i, :], fmt="co", linewidth=2)
#             # curr_plot.set_ylim([min(y) / 3, max(y) * 3])
#         curr_plot.set_title("Interval: " + str(interval) + " cM", fontsize=20)
#         curr_plot.tick_params(axis='x', labelsize=20)
#         curr_plot.tick_params(axis='y', labelsize=20)
#         
#         # curr_plot.annotate("Blocks: %.0f" % self.total_bl_nr, xy=(0.4, 0.7), xycoords='axes fraction', fontsize=16)
#     f.text(0.5, 0.02, r'Distance [$\sigma$]', ha='center', va='center', fontsize=28)
#     f.text(0.025, 0.5, 'IBD-blocks per pair and cM', ha='center', va='center', rotation='vertical', fontsize=28)
#     f.legend((l0, l1, l2, l3, l4, l5), ('Theory', 'DiscSim', 'Laplace', 'Normal', 'Uniform', 'Demes'), loc=(0.83, 0.23))
#     plt.tight_layout()
#     plt.show()

    plt.figure()
    plt.yscale('log')  # Set Log-Scale)
    plt.ylim([1.0 / 10 ** 6, 1.0 / 10])
    plt.xlim([0, 30])
    
    styles = ['r-', 'r--', 'r-.', 'r:']
    for i in range(len(intervals)):
        interval = intervals[i]
                
        plt.semilogy(dist_means, thr_shr[i, :], styles[i], linewidth=3, label=str(interval) + " cM")  # Plot of exact fit
        l1 = plt.errorbar(dist_means - 1, emp_shr[0][i, :], yerr=emp_sts[0][i, :], fmt='yo', linewidth=3)
        l2 = plt.errorbar(dist_means - 0.5, emp_shr[1][i, :], yerr=emp_sts[1][i, :], fmt='go', linewidth=3)
        l3 = plt.errorbar(dist_means, emp_shr[2][i, :], yerr=emp_sts[2][i, :], fmt='ko', linewidth=3)
        l4 = plt.errorbar(dist_means + 0.5, emp_shr[3][i, :], yerr=emp_sts[3][i, :], fmt='mo', linewidth=3)
        l5 = plt.errorbar(dist_means + 1, emp_shr[4][i, :], yerr=emp_sts[4][i, :], fmt="co", linewidth=3)
            # curr_plot.set_ylim([min(y) / 3, max(y) * 3])
        plt.tick_params(axis='x', labelsize=20)
        plt.tick_params(axis='y', labelsize=20)
        
        # curr_plot.annotate("Blocks: %.0f" % self.total_bl_nr, xy=(0.4, 0.7), xycoords='axes fraction', fontsize=16)
    plt.xlabel(r'Distance [$\sigma$]', fontsize=28)
    plt.ylabel('IBD-blocks per pair and cM', fontsize=28)
    f1 = plt.legend((l1, l2, l3, l4, l5), ('DiscSim', 'Laplace', 'Normal', 'Uniform', 'Demes'), loc=(0.75, 0.67))
    # Create a legend for the first line.
    plt.gca().add_artist(f1)  # Add the legend manually to the current Axes.
    
    plt.legend(loc=(0.1, 0.1))
    plt.tight_layout()
    plt.show()


    
def analyze_var_density():   
    '''Analyze varying density data_sets'''  
    b = [1, -1, 0]  # Growth rate parameters
    D = [200, 1, 10]
    save_names = ["growing20.p", "declining20.p", "const20.p"]
    # distances = [[2, 10], [10, 20], [20, 30], [30, 40], [40, 50], [50, 100]]  # Distances to use for binning
    # dists = np.linspace(1, 80, 200)  # x-Values for Theory
    emp_shr = []
    emp_sts = []
    thr_shrs = []
    
    for i in range(len(save_names)):  # Load the data
        name = save_names[i]
        dist_means, thr_shr, shr, sts = analyze_emp_IBD_list(name, show=False, b=b[i], D=D[i])
        thr_shrs.append(thr_shr)
        emp_shr.append(shr)
        emp_sts.append(sts)
     
       
    # Do the actual plot:
    f, axarr = plt.subplots(2, 2, sharex=True, sharey=True)  # Create sub-plots
    for i in range(4):  # Loop through interval list
        curr_plot = axarr[i / 2, i % 2]  # Set current plot
        curr_plot.set_yscale('log')  # Set Log-Scale
        interval = intervals[i] 
        curr_plot.set_ylim([1.0 / 10 ** 7, 1.0 / 10])
        curr_plot.set_xlim([0, 30])
                
        l0, = curr_plot.semilogy(dist_means, thr_shrs[0][i, :], 'r-', linewidth=2)  # Plot of exact fit
        curr_plot.errorbar(dist_means, emp_shr[0][i, :], yerr=emp_sts[0][i, :], fmt='ro')
        
        # l2 = curr_plot.semilogy(dists, bessel_l(1.0, dists, np.sqrt(interval[0]*interval[1]), 1, 200), 'b-')
        # l2 = curr_plot.errorbar(dist_means - 0.3, emp_shr[1][i, :], yerr=emp_sts[1][i, :], fmt='go')
        l1, = curr_plot.semilogy(dist_means, thr_shrs[1][i, :], 'b-', linewidth=2)  # Plot of exact fit
        curr_plot.errorbar(dist_means, emp_shr[1][i, :], yerr=emp_sts[1][i, :], fmt='bo')
 
        l2, = curr_plot.semilogy(dist_means, thr_shrs[2][i, :], 'm-', linewidth=2)  # Plot of exact fit
        curr_plot.errorbar(dist_means, emp_shr[2][i, :], yerr=emp_sts[2][i, :], fmt='mo')
        curr_plot.set_title("Interval: " + str(interval) + " cM", fontsize=20)
        curr_plot.tick_params(axis='x', labelsize=20)
        curr_plot.tick_params(axis='y', labelsize=20)
        
        # curr_plot.annotate("Blocks: %.0f" % self.total_bl_nr, xy=(0.4, 0.7), xycoords='axes fraction', fontsize=16)
    f.text(0.5, 0.02, r'Distance [$\sigma$]', ha='center', va='center', fontsize=28)
    f.text(0.025, 0.5, 'IBD-blocks per pair and cM', ha='center', va='center', rotation='vertical', fontsize=28)
    f.legend((l0, l2, l1), ('Growing', 'Constant', 'Declining'), loc=(0.8, 0.35))
    plt.tight_layout()
    plt.show()   
        
    

def analyze_var_samp():
    '''Analyze saved results in Data2 produced by run_var_samp'''
    (results_l, parameters_l) = pickle.load(open("Data2/laplace1.p", "rb"))
    (results_n, parameters_n) = pickle.load(open("Data2/normal1.p", "rb"))  # @UnusedVariable
    (results_u, parameters_u) = pickle.load(open("Data2/uniform1.p", "rb"))  # @UnusedVariable
    (results_d, parameters_d) = pickle.load(open("Data2/demes3.p", "rb"))  # @UnusedVariable
    (results_c, parameters_c) = pickle.load(open("Data2/discsim2.p", "rb"))  # @UnusedVariable
    
    (results_l1, _) = pickle.load(open("Data2/laplace.p", "rb"))
    (results_n1, _) = pickle.load(open("Data2/normal.p", "rb"))  # @UnusedVariable
    (results_u1, _) = pickle.load(open("Data2/uniform.p", "rb"))  # @UnusedVariable
    (results_d1, _) = pickle.load(open("Data2/demes.p", "rb"))  # @UnusedVariable
    (results_c1, _) = pickle.load(open("Data2/DISCSIM.p", "rb"))  # @UnusedVariable
    
    sample_sizes = parameters_l[2]
    sigma_estimates_l = np.concatenate((results_l[:, :, 0], results_l1[:, :, 0]), axis=1)
    sigma_estimates_n = np.concatenate((results_n[:, :, 0], results_n1[:, :, 0]), axis=1)
    sigma_estimates_u = np.concatenate((results_u[:, :, 0], results_u1[:, :, 0]), axis=1)
    sigma_estimates_d = results_d[:, :, 0]
    sigma_estimates_c = np.concatenate((results_c[:, :, 0], results_c1[:, :, 0]), axis=1)

    mean_sigma_est_l = np.mean(sigma_estimates_l, 1)
    mean_sigma_est_n = np.mean(sigma_estimates_n, 1)
    mean_sigma_est_u = np.mean(sigma_estimates_u, 1)
    mean_sigma_est_d = np.mean(sigma_estimates_d, 1)
    mean_sigma_est_c = np.mean(sigma_estimates_c, 1)
    
    print("Sample size: %.1f" % np.size(sigma_estimates_u, 1))
    
    std_sigma_est_l = np.std(sigma_estimates_l, 1)
    std_sigma_est_n = np.std(sigma_estimates_n, 1)
    std_sigma_est_u = np.std(sigma_estimates_u, 1)
    std_sigma_est_d = np.std(sigma_estimates_d, 1)
    std_sigma_est_c = np.std(sigma_estimates_c, 1)
    
    plt.figure()
    plt.errorbar(np.array(sample_sizes) - 10, mean_sigma_est_d, yerr=std_sigma_est_d, fmt='ko', label="Demes", linewidth=2)
    plt.errorbar(np.array(sample_sizes) - 5, mean_sigma_est_u, yerr=std_sigma_est_u, fmt='bo', label="Uniform", linewidth=2)
    plt.errorbar(sample_sizes, mean_sigma_est_l, yerr=std_sigma_est_l, fmt='mo', label="Laplace", linewidth=2)
    plt.errorbar(np.array(sample_sizes) + 5, mean_sigma_est_n, yerr=std_sigma_est_n, fmt='ro', label="Normal", linewidth=2)
    plt.errorbar(np.array(sample_sizes) + 10, mean_sigma_est_c, yerr=std_sigma_est_c, fmt='yo', label="DiscSim", linewidth=2)
    plt.xlabel("Nr of samples", fontsize=20)
    plt.ylabel(r"$\mathbf{\bar{\sigma}}$", fontsize=30)
    plt.tick_params(axis='x', labelsize=15)
    plt.tick_params(axis='y', labelsize=15)
    plt.axhline(2, color='g', label="True Value", linewidth=2)
    plt.legend()
    plt.show()

def analyze_stats():
    load_u = pickle.load(open("Data1/stats_uniform.p", "rb"))[0][:, 0]
    load_n = pickle.load(open("Data1/stats_normal.p", "rb"))[0][:, 0]
    load_l = pickle.load(open("Data1/stats_laplace.p", "rb"))[0][:, 0]
    load_d = pickle.load(open("Data1/stats_demes.p", "rb"))[0][:, 0]
    load_c = pickle.load(open("./Data1/disc_cstats.p", "rb"))[0][:, 0] / 1000.0
    
    val_u = 1.987
    val_n = 2.02
    val_l = 2.02
    val_d = 2
    val_c = 2
    
#     plt.figure()
#     plt.hist(load_c, bins=20, alpha=0.5)
#     plt.axvline(2, color='r', linestyle='dashed', linewidth=3)
#     plt.xlabel("Estimation")
#     plt.ylabel("Number")
#     plt.show()     
    print(load_c.mean())
    print("Rel. Bias %.6f" % (load_c.mean() / val_c - 1))
    print("CV %.4f" % (load_c.std() / load_c.mean()))
    
    # Plot Stuff
    f, axarr = plt.subplots(3, 2, sharey=True, figsize=(5, 10))
    # axarr.set_title("Estimated Dispersal rate")
    axarr[0, 0].hist(load_u * 2 / val_u, bins=20, alpha=0.5)
    axarr[0, 0].set_title('Uniform')
    axarr[0, 0].axvline(2, color='r', linestyle='dashed', linewidth=3)
    axarr[0, 0].axvline(np.mean(load_u), color='g', linestyle='dashed', linewidth=3)
    
    axarr[0, 1].hist(load_n * 2 / val_n, bins=20, alpha=0.5)
    axarr[0, 1].set_title('Normal')
    axarr[0, 1].axvline(2, color='r', linestyle='dashed', linewidth=3)
    axarr[0, 1].axvline(np.mean(load_n), color='g', linestyle='dashed', linewidth=3)
    
    axarr[1, 0].hist(load_l * 2 / val_l, bins=20, alpha=0.5)
    axarr[1, 0].set_title('Laplace')
    axarr[1, 0].axvline(2, color='r', linestyle='dashed', linewidth=3)
    axarr[1, 0].axvline(np.mean(load_l), color='g', linestyle='dashed', linewidth=3)
    
    axarr[1, 1].hist(load_d * 2 / val_d, bins=20, alpha=0.5)
    axarr[1, 1].set_title('Demes')
    axarr[1, 1].axvline(2, color='r', linestyle='dashed', linewidth=3)
    axarr[1, 1].axvline(np.mean(load_d), color='g', linestyle='dashed', linewidth=3)
    axarr[1, 1].set_xlim(1.85, 2.15001)
        
    axarr[2, 0].hist(load_c * 2 / val_c, bins=20, alpha=0.5)
    axarr[2, 0].set_title('DISC-SIM')
    axarr[2, 0].axvline(2, color='r', linestyle='dashed', linewidth=3)
    axarr[2, 0].axvline(np.mean(load_c), color='g', linestyle='dashed', linewidth=3)
    
    # for i in axarr[]:
        # i.set_xlim
        
        
    plt.delaxes(axarr[2, 1])
    f.text(0.5, 0.04, 'Dispersal Estimates', ha='center')
    f.text(0.04, 0.5, 'Counts', va='center', rotation='vertical')
    # Fine-tune figure; hide x ticks for top plots and y ticks for right plots
    # plt.setp([a.get_xticklabels() for a in axarr[0, :]], visible=False)
    # plt.setp([a.get_yticklabels() for a in axarr[:, 1]], visible=False)
    plt.show()

def run_var_samp1(file_name):
    '''Runs MLE-estimates for various sample sizes and saves estimates and CIs.'''
    grid = factory_Grid()  # Create an empty Grid.
    results = np.zeros((len(sample_sizes), nr_runs, 6))  # Container for the data
    parameters = (grid.sigma, grid.gridsize, sample_sizes, grid.dispmode)
    sample_steps, grid_size = grid.sample_steps, grid.gridsize
    
    '''Actual runs:'''
    row = 0
    for k in sample_sizes:
        position_list = [(i + sample_steps / 2, j + sample_steps / 2, 0) for i in range(0, grid_size, sample_steps) for j in range(0, grid_size, sample_steps)]
        # position_list = [(i + sample_steps / 2, j  + sample_steps / 2, 0) for i in range(0, sample_steps * k, sample_steps) for j in range(0, sample_steps * k, sample_steps)]

        for i in range(0, nr_runs):
            print("Doing run: %.1f for %.0f samples" % (i, k))
            grid = factory_Grid()
            grid.reset_grid()  # Delete everything
            shuffle(position_list)  # Randomize position List
            grid.set_samples(position_list[:k])  # Set the samples
            grid.update_t(t)  # Do the actual run
            if grid.dispmode == "demes":
                grid.update_IBD_blocks_demes(5)  # Update position for deme analysis!!
            
            # Do the maximum Likelihood estimation
            mle_ana = grid.create_MLE_object(bin_pairs=True)  # Create the MLE-object
            mle_ana.create_mle_model("constant", grid.chrom_l, [1, 2])
            mle_ana.mle_analysis_error()
            
            d_mle, sigma_mle = mle_ana.estimates[0], mle_ana.estimates[1] 
            ci_s = mle_ana.ci_s
            results[row, i, :] = (ci_s[1][0], ci_s[1][1], ci_s[0][0], ci_s[0][1], sigma_mle, d_mle)
        row += 1  # Go one down in the results_row
            
        print("RUN COMPLETE!!")
    pickle.dump((results, parameters), open(file_name, "wb"))  # Pickle the data
    print("SAVED") 

            
def analyze_var_samp1(file_name):
    '''Analyze the results of the MLE-estimates for various sample size'''
    (results, parameters) = pickle.load(open(file_name, "rb"))
    print("Parameters used for Simulations: \n")
    print(parameters)
    
    result = 3  # Position of result to analyze
    ci_lengths = results[result, :, 1] - results[result, :, 0]
    
    sigmas_mles = results[result, :, 4]
    d_mles = results[result, :, 5]

    print("Mean CI length: %.4f" % np.mean(ci_lengths))
    print("Mean sigma estimates: %.4f" % np.mean(sigmas_mles))
    print("Standard Deviations sigma: %.4f" % np.std(sigmas_mles))
    print("Mean D_e: %.4f" % (np.mean(d_mles)))
    print("Standard Deviations D_e: %.4f" % np.std(d_mles))
    
    k = len(results[:, 0, 0])
    # Calculate Confidence Intervalls:
    ci_lengths_s = [results[i, :, 1] - results[i, :, 0] for i in range(k)]
    ci_lengths_d = [results[i, :, 3] - results[i, :, 2] for i in range(k)]
    
    # Calculate Empirical Confidence Intervals:
    ci_lengths_s1 = [np.percentile(results[i, :, 4], 97.5) - np.percentile(results[i, :, 4], 2.5) 
                     for i in range(k)]
    ci_lengths_d1 = [np.percentile(results[i, :, 5], 97.5) - np.percentile(results[i, :, 5], 2.5) 
                     for i in range(k)]
    
    print("\n Mean Length of est. Confidence Intervals (Sigma/D)")
    print(np.mean(ci_lengths_s, axis=1))
    print(np.mean(ci_lengths_d, axis=1))
    
    print("\n Empirical Confidence Intervals:")
    print(ci_lengths_s1)
    print(ci_lengths_d1)
    
    # Now do the correlation of estimates:
    print("Correlation of Estimates")
    print([np.corrcoef(results[i, :, 4], results[i, :, 5])[0, 1] for i in range(k)])
    
    # Plot Sigma Estimate
    plt.figure()
    x_dist = np.linspace(0, 3, num=len(sigmas_mles))
    ist = np.argsort(results[0, :, 4])
    plt.plot(0 + x_dist, results[0, ist, 4], 'mo', label="MLE")
    plt.vlines(0 + x_dist, results[0, ist, 0], results[0, ist, 1], 'r', label="Confidence Interval")
    
    ist = np.argsort(results[1, :, 4])
    plt.vlines(3.5 + x_dist, results[1, ist, 0], results[1, ist, 1], 'r')
    plt.plot(3.5 + x_dist, results[1, ist, 4], 'mo')
    
    ist = np.argsort(results[2, :, 4])
    plt.vlines(7 + x_dist, results[2, ist, 0], results[2, ist, 1], 'r')
    plt.plot(7 + x_dist, results[2, ist, 4], 'mo')
    
    ist = np.argsort(results[3, :, 4])
    plt.vlines(10.5 + x_dist, results[3, ist, 0], results[3, ist, 1], 'r')
    # plt.scatter(11 + x_dist, results[3, :, 0], c='b')
    plt.plot(10.5 + x_dist, results[3, ist, 4], 'mo')
    plt.xlabel("Sample Size", fontsize=20)
    plt.ylabel("Estimated " + r"$\mathbf{\sigma}$", fontsize=20)
    plt.xticks([1.5, 5, 8.5, 12], ["100", "270", "440", "625"], fontsize=20)
    plt.hlines(2, -0.5, 14, label="True " + r"$\mathbf{\sigma}$", color='k', linewidth=2)
    plt.legend()
    plt.ylim([0, 3.5])
    plt.show()
    
    # Plot density estimate
    plt.figure()
    x_dist = np.linspace(0, 3, num=len(sigmas_mles))
    ist = np.argsort(results[0, :, 5])
    plt.plot(0 + x_dist, results[0, ist, 5], 'mo', label="MLE")
    plt.vlines(0 + x_dist, results[0, ist, 2], results[0, ist, 3], 'r', label="Confidence Interval")
    
    ist = np.argsort(results[1, :, 5])
    plt.vlines(3.5 + x_dist, results[1, ist, 2], results[1, ist, 3], 'r')
    plt.plot(3.5 + x_dist, results[1, ist, 5], 'mo')
    
    ist = np.argsort(results[2, :, 5])
    plt.vlines(7 + x_dist, results[2, ist, 2], results[2, ist, 3], 'r')
    plt.plot(7 + x_dist, results[2, ist, 5], 'mo')
    
    ist = np.argsort(results[3, :, 5])
    plt.vlines(10.5 + x_dist, results[3, ist, 2], results[3, ist, 3], 'r')
    plt.plot(10.5 + x_dist, results[3, ist, 5], 'mo')
    
    plt.xlabel("Sample Size", fontsize=20)
    plt.ylabel("Estimated " + r"$\mathbf{D}$", fontsize=20)
    plt.xticks([1.5, 5, 8.5, 12], ["100", "270", "440", "625"], fontsize=20)
    plt.hlines(1, -0.5, 14, label="True " + r"$\mathbf{D}$", color='k', linewidth=2)
    plt.legend()
    plt.ylim([0, 2.5])
    plt.show()

def parameter_estimates(file_name, k=625):
    '''Runs MLE-estimates for various growth paramters and saves estimates and CIs.'''
    grid = factory_Grid(1)  # Create an empty Grid.
    start_params = [5, 1]
    results = np.zeros((nr_runs, 9))  # Container for the data    # In case of power growth estimates
    parameters = (grid.sigma, grid.gridsize, sample_sizes, grid.dispmode)
    
    '''Do the actual runs:'''

    for i in range(0, nr_runs):
        print("Doing run: %.1f" % i)
        grid = factory_Grid(growing=True)
        grid.reset_grid()  # Delete everything
        grid.set_random_samples(k)
        grid.update_t(t)  # Do the actual run
        if grid.dispmode == "demes":
            grid.update_IBD_blocks_demes(5)  # Update position for deme analysis!!
            # If one wanted to fit the classic estimates
            # data = Analysis(grid)  # Do Data-Analysis
            # data.fit_expdecay(show=False)  # Do the classic fit
            # sigma_classic = data.sigma_estimate
            
            # Do the maximum Likelihood estimation
        mle_ana = grid.create_MLE_object(bin_pairs=True)  # Create the MLE-object
        mle_ana.create_mle_model("constant", grid.chrom_l, start_params)
        mle_ana.mle_analysis_error()
        
        if len(start_params) == 2:  # In case start_params are too short append stuff
            mle_ana.estimates = np.append(mle_ana.estimates, 0)
            ci_s=np.zeros([3,2])    # Hack to get the confidence interval vector to right length
            ci_s[:2,:]=mle_ana.ci_s
            mle_ana.ci_s =ci_s 
        
        d_mle, sigma_mle, b = mle_ana.estimates[0], mle_ana.estimates[1], mle_ana.estimates[2]
        ci_s = mle_ana.ci_s
        results[i, :] = (ci_s[1][0], ci_s[1][1], ci_s[0][0], ci_s[0][1], ci_s[2][0], ci_s[2][1], sigma_mle, d_mle, b)            
        
    print("RUN COMPLETE!!")
    pickle.dump((results, parameters), open(file_name, "wb"))  # Pickle the data
    print("SAVED") 
    
def analyze_var_growth():
    '''Analyzes the estimates for various growth scenarios generated
    with parameter estimates'''
    (results, parameters) = pickle.load(open("growing625.p", "rb"))
    results_gr, _ = pickle.load(open("declining625.p", "rb"))
    results_const, _ = pickle.load(open("const625.p", "rb"))
    
    
    print(len(results))
    print("Parameters used for Simulations: \n")
    print(parameters)

    sigmas_mles = results[:, 6]
    d_mles = results[:, 7]

    print(parameters)
    print("Mean MLE estimates: %.4f" % np.mean(sigmas_mles))
    print("Standard Deviations MLE: %.4f" % np.std(sigmas_mles))
    print("Mean D_e: %.4f" % (np.mean(d_mles)))
    
    # Plot Sigma Estimate
    plt.figure()
    x_dist = np.linspace(0, 3, num=len(sigmas_mles))
    ist = np.argsort(results[:, 6])
    plt.plot(0 + x_dist, results[ist, 6], 'mo', label="MLE")
    plt.vlines(0 + x_dist, results[ist, 0], results[ist, 1], 'r', label="Confidence Interval")
    
    ist = np.argsort(results_const[:, 6])
    plt.plot(3.5 + x_dist, results_const[ist, 6], 'mo')
    plt.vlines(3.5 + x_dist, results_const[ist, 0], results_const[ist, 1], 'r')
    
    ist = np.argsort(results_gr[:, 6])
    plt.plot(7 + x_dist, results_gr[ist, 6], 'mo')
    plt.vlines(7 + x_dist, results_gr[ist, 0], results_gr[ist, 1], 'r')
    

    plt.ylabel("Estimated " + r"$\mathbf{\sigma}$", fontsize=20)
    plt.xticks([1.5, 5, 8.5], ["Growing", "Constant", "Declining"], fontsize=20)
    plt.hlines(1, -0.5, 10.5, label="True " + r"$\mathbf{\sigma}$", color='k', linewidth=2)
    plt.legend(loc="lower right")
    plt.ylim([0, 1.5])
    plt.show()
    
    
    t = np.linspace(5, 75, 100)
    f, axarr = plt.subplots(3, sharey=True, sharex=True)
    # axarr.set_title("Estimated Dispersal rate")
    
    for i in range(len(results)):
        C, b = results[i, 7], results[i, 8]
        axarr[0].plot(t, C * t ** (-b), 'k-', linewidth=1, alpha=0.7)
    axarr[0].plot(t, 200 * t ** (-1), 'r-', linewidth=2.5)
    
    for i in range(len(results)):
        C, b = results_const[i, 7], results_const[i, 8]
        axarr[1].plot(t, C * t ** (-b), 'k-', linewidth=1, alpha=0.7)
    axarr[1].plot(t, 10 * t ** 0.0, 'r-', linewidth=2.5)
    
    for i in range(len(results)):
        C, b = results_gr[i, 7], results_gr[i, 8]
        l1, = axarr[2].plot(t, C * t ** (-b), 'k-', linewidth=1, alpha=0.7)
    l2, = axarr[2].plot(t, t, 'r-', linewidth=2.5)
    f.legend((l1, l2), ('Estimates', 'True'), loc=(0.63, 0.8))

    f.text(0.5, 0.04, 'Generations back', ha='center', fontsize=20)
    f.text(0.04, 0.5, 'Est. population Density', va='center', rotation='vertical', fontsize=20)
    plt.ylim([0, 80])
    plt.xlim([5, 60])
    # plt.tight_layout()
    plt.show()

def fit_wrong_model():
    '''Fits a constant model to the other two scenarios and estimates the mean sigma based on it.'''
    
    (results, parameters) = pickle.load(open("declining625w.p", "rb"))
    results_gr, _ = pickle.load(open("growing625w.p", "rb"))
    results_const, _ = pickle.load(open("constant625w.p", "rb")) 
    
    
    print(len(results))
    print("Parameters used for Simulations: \n")
    print(parameters)
    # print(results)

    sigmas_mles = results[:, 6]
    d_mles = results[:, 7]

    print(parameters)
    print("Mean MLE estimates: %.4f" % np.mean(sigmas_mles))
    print("Standard Deviations MLE: %.4f" % np.std(sigmas_mles))
    print("Mean D_e: %.4f" % (np.mean(d_mles)))
    
    # Plot Sigma Estimate
    plt.figure()
    x_dist = np.linspace(0, 3, num=len(sigmas_mles))
    ist = np.argsort(results[:, 6])
    plt.plot(0 + x_dist, results[ist, 6], 'mo', label="MLE")
    plt.vlines(0 + x_dist, results[ist, 0], results[ist, 1], 'r', label="Confidence Interval")
    
    ist = np.argsort(results_const[:, 6])
    plt.plot(3.5 + x_dist, results_const[ist, 6], 'mo')
    plt.vlines(3.5 + x_dist, results_const[ist, 0], results_const[ist, 1], 'r')
    
    ist = np.argsort(results_gr[:, 6])
    plt.plot(7 + x_dist, results_gr[ist, 6], 'mo')
    plt.vlines(7 + x_dist, results_gr[ist, 0], results_gr[ist, 1], 'r')
    

    plt.ylabel("Estimated " + r"$\mathbf{\sigma}$", fontsize=20)
    plt.xticks([1.5, 5, 8.5], ["Growing", "Constant", "Declining"], fontsize=20)
    plt.hlines(1, -0.5, 10.5, label="True " + r"$\mathbf{\sigma}$", color='k', linewidth=2)
    plt.legend(loc="lower right")
    plt.ylim([0, 1.5])
    plt.show()
    
    # Plot the pop densities
    t = np.linspace(5, 75, 100)
    f, axarr = plt.subplots(3, sharey=True, sharex=True)
    # axarr.set_title("Estimated Dispersal rate")
    
    for i in range(len(results)):
        C, b = results[i, 7], 0.0
        axarr[0].plot(t, C * t ** (-b), 'k-', linewidth=1, alpha=0.7)
    axarr[0].plot(t, 200 * t ** (-1), 'r-', linewidth=2.5)
    
    for i in range(len(results)):
        C, b = results_const[i, 7], 0.0
        axarr[1].plot(t, C * t ** (-b), 'k-', linewidth=1, alpha=0.7)
    axarr[1].plot(t, 10 * t ** 0.0, 'r-', linewidth=2.5)
    
    for i in range(len(results)):
        C, b = results_gr[i, 7], 0.0
        l1, = axarr[2].plot(t, C * t ** (-b), 'k-', linewidth=1, alpha=0.7)
    l2, = axarr[2].plot(t, t, 'r-', linewidth=2.5)
    f.legend((l1, l2), ('Estimates', 'True'), loc=(0.63, 0.8))

    f.text(0.5, 0.04, 'Generations back', ha='center', fontsize=20)
    f.text(0.04, 0.5, 'Est. population Density', va='center', rotation='vertical', fontsize=20)
    plt.ylim([0, 80])
    plt.xlim([5, 60])
    # plt.tight_layout()
    plt.show()
       
if __name__ == '__main__':
    inp = input("What do you want to do? \n (1) Run Analysis \n (2) Load Analysis\n (3) Run for varying sample size" 
    "\n (4) Analyze varying sample size\n (5) Empirical Block-Lists\n (6) Analyze multiple Models\n "
    "(7) Multiple MLE Runs\n (8) Analyze Multiple MLE Runs\n (9) Compare multiple models \n "
    "(10) Parameter Estimates \n (11) Analyze Estimates Var. Growth \n (12) Fit wrong demographic model\n"
    " (0) Analyze var. density\n")
    if inp == 1:
        analysis_run()
    elif inp == 2:
        analyze_stats()
    elif inp == 3:
        save_name = raw_input("What do you want to save to?\n")
        run_var_samp(save_name)
    elif inp == 4:
        analyze_var_samp()
    elif inp == 5:
        save_name = raw_input("What do you want to save to?\n")
        empirical_IBD_list(save_name)
    elif inp == 6:
        analyze_emp_IBD_list("discsim20.p")  # laplace100.p deme100 #test123
    elif inp == 7:
        run_var_samp1("mle_runs100.p")
    elif inp == 8:
        analyze_var_samp1("mle_runs100.p")
    elif inp == 9: 
        save_lists = ["discsim20.p", "laplace20.p", "uniform20.p", "normal20.p", "deme20.p"]
        analyze_mult_emp_lists(save_lists)
    elif inp == 10:
        parameter_estimates("constant625w.p", 625)
    elif inp == 11:
        analyze_var_growth()
    elif inp == 12:
        fit_wrong_model()
    elif inp == 0:
        analyze_var_density()
        
        
