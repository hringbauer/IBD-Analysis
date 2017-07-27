'''
Created on 21.10.2014
This is the class for Analysis and Data Visualisation
@author: hringbauer
'''

import sys
sys.path.append('../analysis_popres/')
import os
import numpy as np
import bisect
import itertools
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors

from matplotlib.widgets import Slider
from math import sqrt
from collections import Counter
from scipy.misc import factorial  # @UnresolvedImport
from scipy.optimize import curve_fit
from scipy.stats import gaussian_kde
from scipy.stats import binned_statistic
from scipy.special import kv as kv
# from blockpiece import Multi_Bl
from mle_estimation import MLE_estimation  # Fitting without error
from analysis_popres.mle_estim_error import MLE_estim_error
#from mle_estim_error import MLE_estim_error
#from mle_estim_error import MLE_estim_error  # Import the MLE-estimation scheme from POPRES analysis
from statsmodels.stats.moment_helpers import cov2corr
from matplotlib import collections  as mc  # For plotting lines


class Analysis(object):
    '''Object for analyzing the data produced by the Grid object, contains methods for visualization and data analysis. 
    At initialisation it acquires the data of it.'''
    grid_snapshot = []
    gridsize = 0
    IBD_blocks = []
    start_list = []
    IBD_results = ()
    IBD_treshold = 4
    t = 0
    rec_rate = 100
    sigma_estimate = 0.0
    chromosome_l = 0.0
    estimates = []
    stds = []
    
    def __init__(self, grid):  # Initializes the analysis object and saves the chromosomes object it is operating on
        self.IBD_blocks = grid.IBD_blocks
        # self.grid_snapshot = grid.grid
        self.gridsize = grid.gridsize
        self.start_list = grid.start_list
        self.t = grid.t
        self.rec_rate = grid.rec_rate
        self.IBD_treshold = grid.IBD_treshold
        self.chromosome_l = grid.chrom_l
   
    def plot_chromosome_slider(self):
        '''Do a slider plot for the correlograms'''
        fig = plt.figure()
        ax = plt.subplot(111)
        fig.subplots_adjust(left=0.25, bottom=0.25)
 
        # select first image
        [x, y, c, size] = self.extractblockdata(0)
        size = [300 * s for s in size]
        ax.set_xlim([-20, 20])
        ax.set_ylim([-20, 20])
        ax.text(-5, 16, "Generation: 0")
        # ax.set_xlabel('Distance Class')
        # ax.set_ylabel('Heterozygosity')
        ax.scatter(x, y, c=c, s=size, alpha=0.5)
        # save("signal123", ext="png", close=False, verbose=True)

      
        # define slider
        axcolor = 'lightgoldenrodyellow'
        bx = fig.add_axes([0.25, 0.1, 0.65, 0.03], axisbg=axcolor)
        
        slider = Slider(bx, 'Ancestral generation: ', 0, len(self.chromosomes.history1) - 1, valinit=0, valfmt='%i')
        
     
        def update(val):
            ax.cla()
            [x, y, c, size] = self.extractblockdata(int(val))
            size = [300 * s for s in size]
            ax.scatter(x, y, c=c, s=size, alpha=0.5)
            ax.set_xlim([-20, 20])
            ax.set_ylim([-20, 20])
            fig.canvas.draw()

        
        slider.on_changed(update)
        plt.show()
        
    def IBD_analysis(self, n_bins=10, show=True, blocks=0):
        '''Analyze the IBD_blocks saved in the given IBD-block list. 
        Group the blocks via np.hist in bins and calculate mean distance and probabilities'''
        # Create array with the pairwise distance of all detected elements
        
        if blocks == 0:
            blocks = self.IBD_blocks
        pair_distance = [torus_distance(element[2][0], element[2][1], element[3][0], element[3][1], self.gridsize) for element in blocks]
                  
        # Plot the result in a histogram:
        counts, bins, patches = plt.hist(pair_distance, n_bins, facecolor='g', alpha=0.9)  # @UnusedVariable

        if show == True:
            plt.xlabel('Distance')
            plt.ylabel('Number of shared blocks')
            plt.title('Histogram of IBD')
            plt.grid(True)
            plt.show()
            
        # Find proper normalization factors:
        distance_bins = np.zeros(len(bins) - 1)  # Create bins for every element in List; len(bins)=len(counts)+1
        bins[-1] += 0.000001  # Hack to make sure that the distance exactly matching the max are counted
#         for i in self.start_list[1:]:  # Calculate distance to the first element in start_list for all elements to get proxy for number of comparisons
#             dist = torus_distance(i[0], i[1], self.start_list[0][0], self.start_list[0][1], self.gridsize)
#                 
#             j = bisect.bisect_right(bins, dist)
#             if j < len(bins) and j > 0:  # So it actually falls into somewhere
#                 distance_bins[j - 1] += 1
                
        # Calculate Distance for every possible pair to get proper normalization factor:
        for (x, y) in itertools.combinations(np.arange(len(self.start_list)), r=2):
            dist = torus_distance(self.start_list[x][0], self.start_list[x][1], self.start_list[y][0], self.start_list[y][1], self.gridsize)   
            j = bisect.bisect_right(bins, dist)
            if j < len(bins) and j > 0:  # So it actually falls into somewhere
                distance_bins[j - 1] += 1
        
        distance_mean, _, _ = binned_statistic(pair_distance, pair_distance, bins=n_bins, statistic='mean')  # Calculate mean distances for distance bins
        # distance_mean=(bins[1:]+bins[:-1])/2.0
        
        distance_mean = distance_mean[counts != 0]  # Remove bins with no values / MAYBE UPDATE?
        distance_bins = distance_bins[counts != 0]
        counts = counts[counts != 0]
        distance_mean[distance_mean == 0] = 1  # In deme case, to account for possibility of bins with dist=0    
        
        # Poisson-Error:
        error = np.sqrt(counts)
        
        results = [counts[i] / distance_bins[i] for i in range(0, len(counts))]
        error = [error[i] / distance_bins[i] for i in range(0, len(counts))]  # STD
        
        self.IBD_results = (distance_mean, results, error)   
        
    def plot_expdecay(self, logy=True):
        '''Plot the IBD results; The boolean logy determines if log_scale is used'''
        self.IBD_analysis()
        distance_mean, results, _ = self.IBD_results
            
        if logy == False:
            plt.plot(distance_mean, results, 'bo', label="Simulation")
            plt.xlabel('Distance')
            plt.ylabel('Shared blocks per pair')
            plt.title('Probability of sharing a block')
            plt.grid(True)
            plt.show()
        
        if logy == True:
            plt.semilogy(distance_mean, results, 'bo', label="Simulation")
            plt.xlabel('Distance')
            plt.ylabel('Shared blocks per pair')
            plt.title('Probability of sharing a block')
            plt.grid(True)
            plt.show()
            
    def fit_expdecay(self, show=True):
        '''Fit the exponential decay and bessel decay'''
        self.IBD_analysis(show=show)
        x, y, error = self.IBD_results  # Load the data for fitting
        
        # C, r = fit_exp_linear(x, y)  # Fit with exponential fit
        # sigma_estimate = np.sqrt(2 * self.IBD_treshold / self.rec_rate) / (-r)
        # fit = [C * np.exp(r * t) for t in x]
        # print("Exponential fit: \nC: %.4G \nr: %.4G" % (C, r))
              
        parameters, cov_matrix = curve_fit(bessel_decay, x, y, absolute_sigma=True, sigma=error)  # @UnusedVariable p0=(C / 10.0, -r)
        std_param = np.sqrt(np.diag(cov_matrix))  # Get the standard deviation of the results
        C1, r1 = parameters  # Fit with curve_fit:
        sigma_estimate1 = np.sqrt(2 * self.IBD_treshold / self.rec_rate) / (r1)  # Fit sigma
        sigma_std1 = sigma_estimate1 * std_param[1] / r1 
        self.sigma_estimate = sigma_estimate1  # Save sigma
        print("Exact Bessel fit: \nC: %.4G \nr: %.4G" % (C1, r1))
        
        if show == True:  # Do a plot of the fit:
            x_plot = np.linspace(min(x), max(x), 10000)
            plt.figure()
            plt.yscale('log')
            plt.errorbar(x, y, yerr=error, fmt='go', label="Simulated IBD-Data", linewidth=2)
            # plt.semilogy(x, fit, 'y-.', label="Fitted exponential decay")  # Plot of exponential fit
            plt.semilogy(x_plot, bessel_decay(x_plot, C1, r1), 'r-.', label="Fitted Bessel decay", linewidth=2)  # Plot of exact fit
            plt.xlabel('Initial Distance', fontsize=25)
            plt.ylabel('Shared blocks >%0.f cM per pair' % self.IBD_treshold, fontsize=25)
            plt.tick_params(axis='x', labelsize=15)
            plt.tick_params(axis='y', labelsize=15)
            plt.legend(prop={'size':25})
            # Show sigma:
            # plt.annotate(r'$\bar{\sigma_1}=%.3G$' % sigma_estimate , xy=(0.1, 0.04), xycoords='axes fraction', fontsize=25)
            plt.annotate(r'$\bar{\sigma}=%.4G \pm %.2G$' % (sigma_estimate1, sigma_std1) , xy=(0.1, 0.12), xycoords='axes fraction', fontsize=30)
            plt.show()
        
    
    def fit_specific_length(self, interval, show=True):
        '''Fit Bessel decay for blocks of specific length'''
        block_list = [block for block in self.IBD_blocks if interval[0] <= block[1] <= interval[1]]  # Update the block-List
        
        self.IBD_analysis(show=show, blocks=block_list)
        
        x, y, error = self.IBD_results  # Load the data for fitting
        parameters, cov_matrix = curve_fit(bessel_decay2, x, y, absolute_sigma=True, sigma=error)  # @UnusedVariable
        std_param = np.sqrt(np.diag(cov_matrix))  # Get the standard deviation of the results
        C1, r1 = parameters  # Fit with curve_fit:
        sigma_estimate1 = np.sqrt(2 * interval[0] / self.rec_rate) / (r1)  # Fit sigma
        sigma_std1 = sigma_estimate1 * std_param[1] / r1 
        print("Exact Bessel fit: \nC: %.4G \nr: %.4G" % (C1, r1))
        
        if show == True:  # Do a plot of the fit:
            x_plot = np.linspace(min(x), max(x), 10000)
            plt.figure()
            plt.title("Interval: " + str(interval) + " cM")
            plt.yscale('log')
            plt.errorbar(x, y, yerr=error, fmt='go', label="Simulated IBD-Data", linewidth=2)
            plt.semilogy(x_plot, bessel_decay2(x_plot, C1, r1), 'r-.', label="Fitted Bessel decay", linewidth=2)  # Plot of exact fit
            plt.xlabel('Initial Distance', fontsize=25)
            plt.ylabel('Shared blocks >%0.f cM per pair' % self.IBD_treshold, fontsize=28)
            plt.tick_params(axis='x', labelsize=15)
            plt.tick_params(axis='y', labelsize=15)
            plt.legend(prop={'size':25})
            # Show sigma:
            plt.annotate(r'$\bar{\sigma}=%.4G \pm %.2G$' % (sigma_estimate1, sigma_std1) , xy=(0.1, 0.12), xycoords='axes fraction', fontsize=28)
            plt.annotate("Total blocks: %.0f" % len(block_list), xy=(0.8, 0.7), xycoords='axes fraction', fontsize=28)
            plt.show()
            
        # Do a very specific fit.
        f, axarr = plt.subplots(2, 3, sharex=True)  # Create sub-plots
        intervals = ([4, 4.15], [5, 5.2], [6, 6.4], [8, 8.5], [10, 10.5], [15, 15.8])  # Set the interval-list
        
        for i in range(0, 6):  # Loop through interval list
            curr_plot = axarr[i / 3, i % 3]  # Set current plot
            interval = intervals[i]    
            block_list = [block for block in self.IBD_blocks if interval[0] <= block[1] <= interval[1]]  # Update the block-List   

            self.IBD_analysis(show=False, blocks=block_list)
            x, y, error = self.IBD_results  # Load the data for fitting
            x_plot = np.linspace(min(x), max(x), 10000)
            parameters, cov_matrix = curve_fit(bessel_decay2, x, y, absolute_sigma=True, sigma=error)  # @UnusedVariable
            std_param = np.sqrt(np.diag(cov_matrix))  # Get the standard deviation of the results
            C1, r1 = parameters  # Fit with curve_fit:
            sigma_estimate1 = np.sqrt(2 * interval[0] / self.rec_rate) / (r1)  # Fit sigma
            sigma_std1 = sigma_estimate1 * std_param[1] / r1 
            
            print("C-estimate %.6f: " % (C1 * interval[0] / (interval[1] - interval[0])))
            curr_plot.set_yscale('log')
            curr_plot.errorbar(x, y, yerr=error, fmt='go', label="Simulated IBD-Data", linewidth=2)
            curr_plot.semilogy(x_plot, bessel_decay2(x_plot, C1, r1), 'r-.', label="Fitted Bessel decay", linewidth=2)  # Plot of exact fit
            curr_plot.set_ylim([min(y) / 5, max(y) * 5])
            # curr_plot.set_xlabel('Initial Distance', fontsize=25)
            # curr_plot.set_ylabel('Shared blocks >%0.f cM per pair' % self.IBD_treshold, fontsize=25)
            curr_plot.legend(prop={'size':12})
            curr_plot.set_title("Interval: " + str(interval) + " cM")
            # curr_plot.tick_params(axis='y', labelsize=15)
            # curr_plot.tick_params(axis='x', labelsize=15)
            # Show sigma:
            curr_plot.annotate(r'$\bar{\sigma}=%.4G \pm %.2G$' % (sigma_estimate1, sigma_std1) , xy=(0.1, 0.12), xycoords='axes fraction', fontsize=24)
            curr_plot.annotate("Blocks: %.0f" % len(block_list), xy=(0.5, 0.6), xycoords='axes fraction', fontsize=18)
        
        f.text(0.5, 0.04, 'Initial Distance', ha='center', va='center', fontsize=25)
        f.text(0.06, 0.5, 'Shared blocks', ha='center', va='center', rotation='vertical', fontsize=25)
        # plt.xlabel('Initial Distance', fontsize=25)
        # plt.ylabel('Shared blocks', fontsize=25)
        plt.show()
                
    def which_blocks(self):
        '''Analyze Distribution of origin of blocks contributing to IBD'''
        origin_list = []
        for IBD_block in self.IBD_blocks:
            origin_list.append(IBD_block[2])
            origin_list.append(IBD_block[3])
        
        # Count how often blocks are hit
        c = Counter(origin_list)
        # print(c.most_common())
        v = c.values()
        v.sort(reverse=True)
        
        # Print Mean and Variance:
        v = v + [0] * (len(self.start_list) - len(v))  # Fill up the missing chromosomes as 0
        print("\nMean= %.2f" % np.mean(v))
        print("Variance= %.2f" % np.var(v))
        
        # Plot histogram:
        ind = np.arange(len(v))
        plt.bar(ind, v, width=0.95, alpha=0.9)
        plt.xlabel('Block')
        plt.ylabel('Sharing Events of Block')
        plt.title('Histogram of IBD per block')
        plt.grid(True)
        plt.show()
        
        # Meta: Counter of counts:
        counts = Counter(v)
        hits = np.array(counts.keys())  
        values = np.array(counts.values())
        
        values = values / float(sum(values))
        
        def poisson(k, lamb):
            # Poisson function, parameter lamb is the fit parameter
            return (lamb ** k / factorial(k)) * np.exp(-lamb)
        
        # Fit with curve_fit
        parameters, cov_matrix = curve_fit(poisson, hits, values, p0=np.mean(v))  # @UnusedVariable
        print("\nParameters %.2f\n" % parameters[0])
        # plot poisson-deviation with fitted parameter
        x_plot = np.linspace(0, hits.max(), 1000)
          
        plt.figure()
        plt.plot(x_plot, poisson(x_plot, *parameters), 'r-', lw=2, label="Poisson Fit")
        plt.plot(hits, values, 'go', label="Simulated Data")
        plt.xlabel('Events per initial Chromosome')
        plt.ylabel('Probability')
        plt.legend()
        plt.show()
        
    def which_times(self, n_bins=30):
        '''Shows distribution of times when blocks coalesced.'''
        c_times = [block[4] for block in self.IBD_blocks]
        c_distances = [torus_distance(block[2][0], block[2][1], block[3][0], block[3][1], self.gridsize) 
                       for block in self.IBD_blocks]
        # Kernel Density Estimation
        kde = gaussian_kde(c_times)
        x_plot = np.linspace(0, self.t, 1000)
        
        # Fit the coalescence time distribution to IBD_share_pdf
        counts, bins = np.histogram(c_times, bins=self.t, range=[0, self.t])
        t_vals = [0.5 + bins[i] for i in range(0, len(counts))]  # Step is always one!
        parameters, _ = curve_fit(self.IBD_share_pdf, t_vals, counts)  # @UnusedVariable
        print(parameters)
        C = parameters
        fit_vals = self.IBD_share_pdf(x_plot, C)
        
        # Plot coalescence time graphs:
        f, axarr = plt.subplots(2, 1, sharex=True)  # @UnusedVariable 
        axarr[0].hist(c_times, bins=self.t, range=[0, self.t], facecolor='g', alpha=0.6, label="IBD sharing events")
        axarr[0].plot(x_plot, len(c_times) * kde(x_plot), 'r-', linewidth=2, label="Kernel Density Estimation")
        axarr[0].plot(x_plot, fit_vals, 'b-', linewidth=2, label="Theory fit")
        axarr[0].legend()
        # axarr[0].set_title('Distribution of Coalescence Times')
        axarr[0].grid(True)
        axarr[0].set_ylabel("Nr of long IBD sharing events", fontsize=14)
        # Coaltimes vrs block distance
        axarr[1].plot(c_times, c_distances, 'go', alpha=0.4)
        axarr[1].set_ylim(0, max(c_distances))
        axarr[1].set_xlim(0, self.t)
        axarr[1].set_ylabel("Initial Sampling Distance", fontsize=14)
        axarr[1].grid(True)
        plt.xlabel('Coalescence Time', fontsize=18)
        plt.tight_layout()
        plt.show()   
    
    
    def plot_fitted_data(self): 
        '''Plots the fit to the binned data set    
        This very specific plot method visualizes IBD-sharing '''
        f, axarr = plt.subplots(2, 3, sharex=True)  # Create sub-plots
        # intervals = ([3.0, 3.5], [3.5, 4.0], [4.0, 4.5], [4.5, 5], [5.0, 6], [6.0, 7], [7.0, 8.5], [8.5, 10])  # Set the interval-list
        intervals = ([5, 5.2], [6, 6.4], [7, 7.4], [8, 8.5], [10, 10.5], [15, 15.8])  # Set the interval-list
        
        for i in range(0, 6):  # Loop through interval list
            curr_plot = axarr[i / 3, i % 3]  # Set current plot
            interval = intervals[i] 
            block_list = [block for block in self.IBD_blocks if interval[0] <= block[1] <= interval[1]]  # Update the block-List   
            print("Length of block list: %.0f" % len(block_list))
            
            self.IBD_analysis(show=False, blocks=block_list)
            x, y, error = self.IBD_results  # Load the data for fitting
            x_plot = np.linspace(min(x), max(x), 10000)
            y_plot_est = bessel_decay_interval(x_plot, self.c_estimate, self.sigma_estimate, interval)  # Generates the Fit
            curr_plot.set_yscale('log')
            curr_plot.errorbar(x, y, yerr=error, fmt='go', label="Binned IBD-sharing", linewidth=2)
            curr_plot.semilogy(x_plot, y_plot_est, 'r-.', label="Fitted Bessel decay", linewidth=2)  # Plot of exact fit
            curr_plot.set_ylim([min(y) / 5, max(y) * 5])
            curr_plot.legend(prop={'size':12})
            curr_plot.set_title("Interval: " + str(interval) + " cM")
            # curr_plot.tick_params(axis='y', labelsize=15)
            # curr_plot.tick_params(axis='x', labelsize=15)
            
            # Show sigma:
            # #curr_plot.annotate(r'$\bar{\sigma}=%.4G \pm %.2G$' % (sigma_estimate1, sigma_std1) , xy=(0.1, 0.12), xycoords='axes fraction', fontsize=20)
            curr_plot.annotate("Blocks: %.0f" % len(block_list), xy=(0.5, 0.6), xycoords='axes fraction', fontsize=18)
        
        f.text(0.5, 0.04, 'Initial Distance', ha='center', va='center', fontsize=25)
        f.text(0.06, 0.5, 'Shared blocks per pair', ha='center', va='center', rotation='vertical', fontsize=25)
        plt.show()
    
    def mle_estimate(self, endog, exog):
        '''MLE estimate'''
        ml_estimator = MLE_estimation(endog, exog)
        print("Doing fit")
        results = ml_estimator.fit()  # method="nelder-mead"
        # results0 = ml_estimator.fit(method="BFGS")  # Do the actual fit. method="BFGS" possible
        self.sigma_estimate = results.params[1]  # Save Parameters
        self.c_estimate = results.params[0]
        
        fisher_info = np.matrix(ml_estimator.hessian(results.params))  # Get the Fisher Info matrix
        stds = np.sqrt(np.diag(-fisher_info.I))
        print("Sigma estimate: %.4f" % results.params[1])
        print("Estimated STD: %.4f" % stds[1])
        print("C estimate: %.4f" % results.params[0])
        print("Estimated STD: %.4f" % stds[0])
        # boot_mean, boot_std, _ = results.bootstrap(nrep=50, store=True) # Do some boots_trap
        # print(boot_mean)
        # print(boot_std)
        return fisher_info.I[1, 1]  # Return the estimated Variance in Sigma
    
    def mle_estimate_error(self):
        '''MLE-estimation from the POPRES analysis. Bins the data. And can deal with errors.
        Param[0] always C; Param[1] always sigma'''
        # First create mle_object
        pw_dist, pw_IBD, pair_nr = self.give_pairwise_statistics()  # Create full pw. statistics
        pw_dist, pw_IBD, pair_nr = self.bin_pairwise_statistics(pw_dist, pw_IBD, pair_nr)
        bl_shr_density = uniform_density
        start_params = [1.0, 2.0]  # 1: D 2: Sigma
        
        # Create MLE_estimation object:
        ml_estimator = MLE_estim_error(bl_shr_density, start_params, pw_dist, pw_IBD, pair_nr, error_model=False) 
        self.estimates = start_params  # Best guess without doing anything. Used as start for Bootstrap
        
        print("Doing fit...")
        results = ml_estimator.fit()  # method="nelder-mead"
        # results0 = ml_estimator.fit(method="BFGS")  # Do the actual fit. method="BFGS" possible
        self.estimates = results.params  # Save the paramter estimates
            
        fisher_info = np.matrix(ml_estimator.hessian(results.params))  # Get the Fisher Info matrix
        corr_mat = cov2corr(-fisher_info.I)
        print(corr_mat)
        stds = np.sqrt(np.diag(-fisher_info.I))
        self.stds = stds  # Save estimated STDS
            
        for i in range(len(results.params)):
            print("Parameter %i: %.6f" % (i, results.params[i]))
            print("CI: " + str(results.conf_int()[i]))
            # print("Estimated STD: %.6f" % stds[i])
        # print("D=%.5f" % self.from_C_to_D_e(results.params[0], results.params[1]))    
        print(results.summary())  # Give out the results.
        
        # self.mle_object = ml_estimator  # Remember the mle-estimation object.
    
    
    def plot_blocks(self):
        '''Plots all pairwise shared blocks'''
        ibd_blocks, start_list = self.IBD_blocks, self.start_list  # Load rel. data
        print(start_list)
        print(ibd_blocks[0])
        
        # Calculate fraction of coalesced loci: (information for SIDE PROJECT)
        tot_ibd_length = np.sum([i[1] for i in ibd_blocks])  # Length of all IBD-blocks in total
        n = len(start_list)
        tot_possible = n * (n - 1) / 2.0 * self.chromosome_l  # Length of totally possible pairwise IBD-blocks
        
        
        print("Fraction of loci coalesced: %.6f" % (tot_ibd_length / tot_possible))
        
        # c = np.array([(1, 0, 0, 1), (0, 1, 0, 1), (0, 0, 1, 1)])    # Colors
        
        # Generate the lines
        chrom_lines = [[(sl[0], sl[1]), (sl[0], sl[1] + 1.5)] for sl in start_list]
        
        # Generate the shared chromosomes
        

        for i in ibd_blocks:  # Do the beginning of a block 
            x_coord = [i[2][0] for i in ibd_blocks]
            y_begin = [i[2][1] + 1.5 / 100 * i[0] for i in ibd_blocks]
            y_end = [i[2][1] + 1.5 / 100 * (i[0] + i[1]) for i in ibd_blocks]
            
        for i in ibd_blocks:  # Do the ends of blocks
            x_coord1 = [i[3][0] for i in ibd_blocks]
            y_begin1 = [i[3][1] + 1.5 / 100 * i[0] for i in ibd_blocks]
            y_end1 = [i[3][1] + 1.5 / 100 * (i[0] + i[1]) for i in ibd_blocks]
            
        N = len(ibd_blocks)
        print("IBD-blocks found: %i" % N)
        
        # Do the connecting lines:
        # conn_lines = [[(x_coord[i], (y_begin[i]+y_end[i])/2.0), 
        #            (x_coord[i], (y_begin[i]+y_end[i])/2.0)] for i in range(len(ibd_blocks))]
        
        
        
        
        cmap = get_cmap(N)  # Get the color map
        cmap = [cmap(i) for i in range(N)]  # Get the color array
            
        shrd_chroms1 = [[(x_coord[i], y_begin[i]), (x_coord[i], y_end[i])] for i in range(len(ibd_blocks))]
        shrd_chroms2 = [[(x_coord1[i], y_begin1[i]), (x_coord1[i], y_end1[i])] for i in range(len(ibd_blocks))]
        
        lc = mc.LineCollection(chrom_lines, colors='0.75', linewidths=50)  # Generate the lines
        lc1 = mc.LineCollection(shrd_chroms1, colors=cmap, alpha=0.8, linewidths=50)  # Generate the first set
        lc2 = mc.LineCollection(shrd_chroms2, colors=cmap, alpha=0.8, linewidths=50)  # Generate the corresponding ends
        
        # lcc= mc.LineCollection(conn_lines, colors=cmap, alpha=0.8, linewidths=5) # Generate all connecting lines
        
        fig, ax = plt.subplots()  # @UnusedVariable
        ax.add_collection(lc)
        ax.add_collection(lc1)
        ax.add_collection(lc2)
        # ax.add_collection(lcc)
        
        plt.axis("equal")  # Do set the geographic distance equal
        ax.autoscale()
        ax.margins(0.05)
        # plt.xlim([46,76])
        plt.show()
    
    
    def from_C_to_D_e(self, C, sigma):
        '''Calculates the actual density from C and sigma'''
        G = self.chromosome_l / self.rec_rate  # Sex average of human genome. In Morgan!!
        return 2 * G / (4 * np.pi * sigma ** 2 * 2 * C)    
        
    def give_pairwise_statistics(self):
        '''Method which returns pairwise distance, IBD-sharing and pw. Number.
        Used for full MLE-Method. Return arrays'''
        l = len(self.start_list) 
        pair_IBD = np.zeros((l * (l - 1) / 2))  # List of IBD-blocks per pair
        pair_IBD = [[] for _ in pair_IBD]  # Initialize with empty lists
        
        # Iterate over all IBD-blocks
        for bpair in self.IBD_blocks:
            ibd_length = bpair[1]  # Get length in centiMorgan
            ind1 = self.start_list.index(bpair[2])
            ind2 = self.start_list.index(bpair[3])    
            j, i = min(ind1, ind2), max(ind1, ind2) 
            pair_IBD[i * (i - 1) / 2 + j].append(ibd_length)  # Append an IBD-block  
        
        # Get distance Array of all blocks
        pair_dist = [torus_distance(self.start_list[i][0], self.start_list[i][1],
                                    self.start_list[j][0], self.start_list[j][1], self.gridsize) for i in range(0, l) for j in range(0, i)]
        pair_nr = np.ones(len(pair_dist))
        return (np.array(pair_dist), np.array(pair_IBD), pair_nr) 
    
    def bin_pairwise_statistics(self, pw_dist, pair_IBD, pair_nr):
        '''Bins pairs of same length into one distance pair.
        This does not change the likelihood function but speeds up calculation'''
        distances = sorted(set(pw_dist))  # Produce the keys in a sorted fashion
        
        new_pair_IBD = [[] for _ in distances]  # Initialize the new shortened arrays
        new_pair_nr = [0 for _ in distances]
        
        for j in range(len(distances)):
            r = distances[j]
            for i in range(len(pw_dist)):  # Iterate over all pairs
                if pw_dist[i] == r:  # If Match
                    new_pair_IBD[j] += list(pair_IBD[i])  # Append the shared blocks
                    new_pair_nr[j] += pair_nr[i]  # Add the number of individuals
                    
        print("Nr. of effective distance bins: %i" % np.sum(new_pair_nr))
        print("Nr of total blocks for analysis: %i" % np.sum([len(i) for i in new_pair_IBD]))
        return(distances, new_pair_IBD, new_pair_nr)
            
    def IBD_share_pdf(self, t, C):
        return(C * t * np.exp(-2 * self.IBD_treshold * t / float(self.rec_rate))) 
        
#######################################################

     
def torus_distance(x0, y0, x1, y1, torus_size):
    # Calculates the Euclidean distance on a Torus
    dist_x = abs(x0 - x1)
    dist_y = abs(y0 - y1)
    distance = sqrt(min(dist_x, torus_size - dist_x) ** 2 + min(dist_y, torus_size - dist_y) ** 2)
    return(distance)

def bessel_decay(x, C, r):
    '''Fit to expected decay curve in 2d (C absolute value, r rate of decay)'''
    return(C * x * kv(1, r * x))   

def bessel_decay2(x, C, r):
    '''Fit to expected decay of certain block length'''
    return(C * x * x * kv(2, r * x)) 
        
def fit_exp_linear(t, y):
    # Fitting exponential decay and returns parameters: y=A*Exp(-kt)
    y = np.log(y)
    K, A_log = np.polyfit(t, y, 1)
    A = np.exp(A_log)
    return A, K       

def bessel_decay_interval(r, C, sigma, interval):
    '''Gives Bessel-Decay in a given interval. If r vector returns vector
    Intervals are in cM!!!'''
    l = np.sqrt(interval[0] * interval[1])  # Calculate Harmonic Mean
    b_l = C * r ** 2 / (2 * l / 100.0 * sigma ** 2) * kv(2, np.sqrt(2 * l / 100.0) * r / sigma)
    return b_l * (interval[1] - interval[0]) / 100.0

def uniform_density(l, r, params):
    '''Gives uniform density per cM(!) If l vector return vector'''
    G = 1.5
    D = params[0]  # Density
    sigma = params[1]
    C = G / (4 * np.pi * sigma ** 2 * D)  # The constant in front
    b_l = C * r ** 2 / (2.0 * (l / 100.0 * sigma ** 2)) * kv(2, np.sqrt(2.0 * l / 100.0) * r / sigma)
    return b_l / 100.0  # Factor for density in centi Morgan


def get_cmap(N):
    '''Returns a function that maps each index in 0, 1, ... N-1 to a distinct 
    RGB color'''
    color_norm = colors.Normalize(vmin=0, vmax=N - 1)
    scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='hsv') 
    def map_index_to_rgb_color(index):
        return scalar_map.to_rgba(index)
    return map_index_to_rgb_color


     
def save(path, ext='png', close=True, verbose=True):
    """Save a figure from pyplot.
    Parameters
    ----------
    path : string
    The path (and filename, without the extension) to save the
    figure to.
    ext : string (default='png')
    The file extension. This must be supported by the active
    matplotlib backend (see matplotlib.backends module). Most
    backends support 'png', 'pdf', 'ps', 'eps', and 'svg'.
    close : boolean (default=True)
    Whether to close the figure after saving. If you want to save
    the figure multiple times (e.g., to multiple formats), you
    should NOT close it in between saves or you will have to
    re-plot it.
    verbose : boolean (default=True)
    Whether to print information about when and where the image
    has been saved.
    """
    # Extract the directory and filename from the given path
    directory = os.path.split(path)[0]
    filename = "%s.%s" % (os.path.split(path)[1], ext)
    if directory == '':
        directory = '.'
     
    # If the directory does not exist, create it
    if not os.path.exists(directory):
        os.makedirs(directory)
     
    # The final path to save to
    savepath = os.path.join(directory, filename)
     
    if verbose:
        print("Saving figure to '%s'..." % savepath),
     
    # Actually save the figure
    plt.savefig(savepath, bbox_inches='tight')
    # Close it
    if close:
        plt.close()
     
    if verbose:
        print("Done") 
