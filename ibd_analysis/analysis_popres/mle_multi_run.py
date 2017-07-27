'''
Created on Feb 8, 2016
Data set for analyzing block-sharing data.
Handles the MLE-object. This is the object which contains IBD-summary data
and is saved by pickle
@author: Harald
'''

import numpy as np
import matplotlib.pyplot as plt
import warnings

# from mle_estimation import MLE_estimation, MLE_estimation_growth, MLE_estimation_dd Not needed anymore; look in old versions
from mle_estim_error import MLE_estim_error, MLE_Estim_Barrier
from scipy.stats import binned_statistic  # For calculating binned values for better visualization.
from statsmodels.stats.moment_helpers import cov2corr
from scipy.special import kv as kv  # Import Bessel functions of second kind
from scipy.optimize import curve_fit
from itertools import izip
from functools import partial
from copy import deepcopy

class MLE_analyse(object):
    '''
    This is a class which analyses the pre-processed Data object.
    Contains methods to analyse and visualize the data.
    Throughout genetic distances are measured in cM.
    '''
    countries = []  # List containing the countries of interest.
    nr_individuals = []  # Nr of individuals per country
    pw_distances = []  # Matrix with pairwise geographic distances.
    pw_block_sharing = []  # Matrix containing the pairwise block sharing.
    latlon_list = []  # List of Lat/lon of Positions
    
    lin_block_sharing = []  # List of list of the pairwise block-sharing    (Np-array)
    lin_dists = []  # List containing the pw distances
    lin_pair_nr = []  # List containing the number of individual pairs
    labels = []
    total_bl_nr = 0  # Contains the total number of last analyzed blocks.
    sigma_estimate = 0  # Contains the estimated sigma parameter.
    c_estimate = 0  # Contains the estimated C parameter.
    mu_estimate = 0  # Contains the estimated mu parameter. In absolute rate units not percent!
    estimates = []  # Vector for all the estimates.
    ci_s = []  # Array of start and end of CI interval
    stds = []  # Vector for the standard deviations.
    btst_estimates = []  # Array for the bootstrap estimates.
    mle_object = 0  # Place for the object used to do MLE.
    error_model = 0  # Default whether to use an error model or not
    all_chrom = 0  # Default whether to use specific human chromosome lenghts
    chrom_l = 1.5  # The Length of a Chromosome
    
    def __init__(self, data=0, pw_dist=[], pw_IBD=[], pw_nr=[], all_chrom=False, error_model=True, position_list=[]):
        '''
        Constructor. If data do the POPRES data processing (including ctry matrices, 
        Else just take the three vectors
        Chrom Length is in Morgan!
        '''
        self.error_model = error_model
        self.all_chrom = all_chrom
        if data:
            self.init_POPRES_data(data)  # Initialize POPRES data properly (which has additional info)
        else:  # Initialize the three important arrays
            self.lin_block_sharing = pw_IBD
            self.lin_dists = pw_dist
            self.lin_pair_nr = pw_nr
        if len(position_list) > 0:  # Overwrite Position List; in case it is given.
            self.position_list = position_list

        block_nr = np.sum([len(block_list) for block_list in pw_IBD])
        print("Total Block Nr. for Analysis: %i" % block_nr)
        
    def init_POPRES_data(self, data):
        '''Brings the POPRES data in needed shape. 
        I.e. generate the three important linear vectors'''
        self.countries = data.countries_oi
        self.pw_distances = data.pw_distances
        self.pw_block_sharing = data.pw_blocksharing
        self.nr_individuals = data.nr_individuals 
        # self.position_list = data.position_list
        self.position_list = data.position_list
        self.lin_dists, _ , self.lin_pair_nr, self.labels = self.return_linearized_data(3.0, 150)
        self.plot_cartesian_position()
    
    def plot_cartesian_position(self, barrier=[0, 0], angle=0):
        '''Plots the Cartesian Position of samples;
        after applying the right transformation.
        Also plots putative Position of the barrier.
        Angle is in Radiant!!'''
        position_list = self.centering_positions(self.position_list, [barrier, angle])
        x = position_list[:, 0]
        y = position_list[:, 1]
        
        # Do the actual Plot
        _, ax = plt.subplots()
        ax.scatter(x, y)
        for i, txt in enumerate(self.countries):
            ax.annotate(txt, (x[i], y[i]))
        plt.xlabel("x-Cooridnate [km]")
        plt.ylabel("y-Cooridnate [km]")
        plt.axvline(x=0, color='r', linestyle='-', linewidth=2)
        plt.show()
     
    def return_linearized_data(self, min_len, max_len):
        '''Returns data in linearized form; calculates average pair_sharing and linearized pair_sharing list.
        self.lin_block_sharing is created via calc_nr_shr_bl'''
        linear_dist = []
        pair_sharing = []
        pair_nr = []
        label = []
        
        pw_blocksharing = self.calc_nr_shr_bl(min_len, max_len)  # Calculate block sharing with given limits
        
        k = len(self.pw_distances[:, 0])
        # First linearize the data and give back useful comments
        for i in range(k):  
            for j in range(0, i):
                linear_dist.append(self.pw_distances[i, j])
                pw_nr = float(self.nr_individuals[i] * self.nr_individuals[j])
                pair_nr.append(pw_nr)
                label.append(self.countries[i] + "-" + self.countries[j])
                pair_sharing.append(pw_blocksharing[i, j] / float(pw_nr))        
        return (np.array(linear_dist), np.array(pair_sharing), np.array(pair_nr), label)   
            
    def calc_nr_shr_bl(self, threshold, threshold_top=150):
        '''Give back number MATRIX of shared blocks longer than threshold
        and below top threshold (if given). Create self.lin_block_sharing array.'''
        lin_block_sharing = []  # Delete if existing
        k = len(self.pw_distances[:, 0])
        pw_block_nr = np.zeros((k, k))    
        for i in range(k):
            for j in range(i):
                # print(self.pw_block_sharing[i, j])
                b_s = np.array(self.pw_block_sharing[i, j]).astype('float')
                pw_block_nr[i, j] = np.sum((b_s >= threshold) * (b_s <= threshold_top))  # Calculate number of interesting blocks
                lin_block_sharing.append(b_s[(b_s >= threshold) * (b_s <= threshold_top)])  # Keep interesting blocks in linearized array.
        self.total_bl_nr = np.sum(pw_block_nr)
        print("Interesting block sharing: %.0f " % self.total_bl_nr)  # Print interesting block sharing
        self.lin_block_sharing = np.asarray(lin_block_sharing)
        
        return pw_block_nr
    
    def extract_blocks_spec_len(self, min_len, max_len):
        '''Extracts blocks of specific length from linearized array.'''
        new_block_list = [[b for b in blocks if (min_len < b < max_len)] for blocks in self.lin_block_sharing]
        self.total_bl_nr = np.sum([len(b) for b in new_block_list])  # Update the total number of blocks of this length
        return np.array(new_block_list)
            
    def analyze_bin_ibd(self, min_len, max_len, n_bins=6, show=True):
        '''Extract binned IBD-data of spec. length and return summary statistics
        If bins given use them as distance bins; otherwise only number of bins'''
        pair_sharing = self.extract_blocks_spec_len(min_len, max_len)
        pair_sharing = np.array([len(blocks) for blocks in pair_sharing])
        
        lin_dists, labels, pair_nr = self.lin_dists, self.labels, self.lin_pair_nr
        
        # Do an interactive Plot
        if show == True:  
            self.interactive_plot(lin_dists, np.log10(np.array(pair_sharing) + 0.0001), labels, np.sqrt(pair_nr))
        
        # Generate bins and calculate the mean distance for pw country comparisons
        bin_mdist, bins, _ = binned_statistic(lin_dists, lin_dists, bins=n_bins, statistic='mean')  # Calculate mean bin distance
        bin_mdist = np.array(bin_mdist)
        
        # Go into bins to calculate weighted averages
        bin_nr = np.array([0 for _ in bins[:-1]]).astype('float')
        mean_sharing = np.array([0 for i in bins[:-1]]).astype('float')
        
        bins[0] -= 0.00001  # Do get first entry right
        bins[-1] += 0.00001  # Do get last entry right
        bin_inds = np.searchsorted(bins, lin_dists) - 1  # Calculate bin-nr for every country comparison
        
        for i in range(len(lin_dists)):
            bi = bin_inds[i]
            if 0 <= bi < len(bin_nr):  # If bin index makes sense(i.e. if it falls not outside the bins)
                bin_nr[bi] += pair_nr[i]  # Add to Nr of total pw. inds per distance bin.
                mean_sharing[bi] += pair_sharing[i]  # The weighted contribution
        
        # Do the normalizing. Deal with empty bins. (and bins without anything)
        errors = np.sqrt(mean_sharing) / bin_nr  # Assuming Poisson counts in bins.
        errors[mean_sharing == 0] = 1 / bin_nr[mean_sharing == 0]  # Deal with 0 errors (set it to error of 1)
            
        mean_sharing = mean_sharing / bin_nr  # Normalize for Nr of inds per bin
        
        good_inds = bin_nr != 0  # Inds for non-empty bins
        print(mean_sharing[good_inds])
        return(bin_mdist[good_inds], mean_sharing[good_inds], errors[good_inds])
    
    def plot_ibd_spec(self, min_len, max_len, show=True):    
        '''Visualize binned IBD against distance for blocks of specific length'''  
        (bin_mdist, mean_sharing, errors) = self.analyze_bin_ibd(min_len, max_len)
        
        # Do the Bessel Fit
        self.fit_specific_length(x=bin_mdist[:], y=mean_sharing[:], error=errors[:], interval=[min_len, max_len])
    
    def plot_ibd_min(self, min_len, max_len, show=True):
        '''Visualize binned IBD against distance for blocks of given minimum length'''
        (bin_mdist, mean_sharing, errors) = self.analyze_bin_ibd(min_len, max_len)
        # Do the Besself Fit and plot
        self.fit_bessel_decay(x=bin_mdist[:], y=mean_sharing[:], error=errors[:], ibd_threshold=min_len)
             
    def interactive_plot(self, x, y, labels, size):
        '''Generate interactive scatter plot with click-able labels'''
        if 1:  # picking on a scatter plot (matplotlib.collections.RegularPolyCollection)
            def give_country(ind):
                '''Gives back the country ids from linearized array:'''
                print() 
                
            def onpick3(event):
                ind = event.ind
                print 'onpick3 scatter:', np.take(x, ind), np.take(y, ind), np.take(labels, ind)

            fig = plt.figure()
            ax1 = fig.add_subplot(111)
            col = ax1.scatter(x, y, size, picker=True)  # @UnusedVariable
            # fig.savefig('pscoll.eps')
            fig.canvas.mpl_connect('pick_event', onpick3)
            
        plt.xlabel("Pair Distance (km)")
        plt.ylabel("Avg Blocksharing")
        plt.show()       
    
    def show_longest_blocks(self, nr=100):
        '''Prints the nr longest blocks '''
        (_, _, _, label) = self.return_linearized_data(0, 10000)  # Get the right labels
        
        shr_list = []
        label_list = []
        
        for i in range(len(self.lin_block_sharing)):
            shr_list += list(self.lin_block_sharing[i])  # Append block length
            label_list += [label[i]] * len(self.lin_block_sharing[i])  # Append labels      
        shr_list, label_list = np.array(shr_list), np.array(label_list)
        
        indices = shr_list.argsort()[-nr:][::-1]  # Sort to get number nr biggest entries
        
        j = 0
        for i in indices:
            j += 1   
            print("Nr %.0f Between %s Length %.3f" % (j, label_list[i], shr_list[i]))
            
    def fit_bessel_decay(self, x, y, error, ibd_threshold, show=True):
        '''Method to fit Bessel-Decay of Block sharing'''
        parameters, cov_matrix = curve_fit(bessel_decay, x, y, absolute_sigma=True, sigma=error, p0=(y[0], 1 / 500.0))
        std_param = np.sqrt(np.diag(cov_matrix))  # Get the standard deviation of the results
        C1, r1 = parameters  # Fit with curve_fit:
        sigma_estimate1 = np.sqrt(2 * ibd_threshold / 100.0) / (r1)  # Fit sigma
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
            plt.ylabel('Shared blocks >%1.f cM per pair' % ibd_threshold, fontsize=25)
            plt.tick_params(axis='x', labelsize=15)
            plt.tick_params(axis='y', labelsize=15)
            plt.legend(prop={'size':25})
            # Show sigma:
            plt.annotate(r'$\bar{\sigma}=%.4G \pm %.2G$' % (sigma_estimate1, sigma_std1) ,
            xy=(0.1, 0.12), xycoords='axes fraction', fontsize=30)
            plt.show()      
    
    def fit_specific_length(self, x, y, error, interval, show=True):
        '''Fit Bessel decay for blocks in specific length interval'''
        parameters, cov_matrix = curve_fit(bessel_decay2, x, y, absolute_sigma=True, sigma=error, p0=(y[0], 1 / 500.0))  
        std_param = np.sqrt(np.diag(cov_matrix))  # Get the standard deviation of the results
        C1, r1 = parameters  # Fit with curve_fit:
        sigma_estimate1 = np.sqrt(2.0 * interval[0] / 100) / (r1)  # Fit sigma (with interval midpoint value
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
            plt.ylabel('Shared blocks %0s cM per pair' % str(interval), fontsize=28)
            plt.tick_params(axis='x', labelsize=15)
            plt.tick_params(axis='y', labelsize=15)
            plt.legend(prop={'size':25})
            # Show sigma:
            plt.annotate(r'$\bar{\sigma}=%.4G \pm %.2G$' % (sigma_estimate1, sigma_std1) ,
            xy=(0.1, 0.12), xycoords='axes fraction', fontsize=28)
            plt.annotate("Total blocks: %.0f" % self.total_bl_nr, xy=(0.6, 0.7), xycoords='axes fraction', fontsize=28)
            plt.show()
                             
    
    def visualize_ibd_diff_lengths(self):
        '''This very specific plot method visualizes IBD-sharing '''
        f, axarr = plt.subplots(2, 4, sharex=True)  # Create sub-plots
        # intervals = ([3.0, 3.5], [3.5, 4.0], [4.0, 4.5], [4.5, 5], [5.0, 6], [6.0, 7], [7.0, 8.5], [8.5, 10])  # Set the interval-list
        intervals = ([4.0, 4.4], [4.4, 4.9], [4.9, 5.5], [5.5, 6.2], [6.2, 7], [7, 8.4], [8.4, 10], [10, 12])
        
        c_est = []
        
        for i in range(0, 8):  # Loop through interval list
            curr_plot = axarr[i / 4, i % 4]  # Set current plot
            interval = intervals[i] 
            # int_len=interval[1]-interval[0]
            (bin_mdist, mean_sharing, errors) = self.analyze_bin_ibd(interval[0], interval[1], show=False)  # Extract relevant binned data   
            x, y, error = bin_mdist, mean_sharing, errors  # Load the data for fitting
                        
            curr_plot.set_yscale('log')
            curr_plot.errorbar(x, y, yerr=error, fmt='go', label="IBD-Sharing", linewidth=2)
            # x_plot = np.linspace(min(x), max(x), 10000)
            # curr_plot.semilogy(x_plot, bessel_decay2(x_plot, C1, r1), 'r-.', label="Fitted Bessel decay", linewidth=2)  # Plot of exact fit
            curr_plot.set_ylim([min(y) / 3.0, max(y) * 3])
            # curr_plot.set_xlabel('Initial Distance', fontsize=25)
            # curr_plot.set_ylabel('Shared blocks >%0.f cM per pair' % self.IBD_treshold, fontsize=25)
            # curr_plot.legend(prop={'size':12})
            curr_plot.set_title("Interval: " + str(interval) + " cM")
            # curr_plot.tick_params(axis='y', labelsize=15)
            # curr_plot.tick_params(axis='x', labelsize=15)
            # Show sigma:
            # parameters, cov_matrix = curve_fit(bessel_decay2, x, y, absolute_sigma=True, sigma=error, p0=(y[0], 1 / 500.0))
            # C1, r1 = parameters  # Fit with curve_fit:
            # c_est.append(C1 * interval[0] / (interval[1] - interval[0]))
            # std_param = np.sqrt(np.diag(cov_matrix))  # Get the standard deviation of the results
            # sigma_estimate1 = np.sqrt(2.0 * interval[0] / 100.0) / (r1)  # Fit sigma
            # sigma_std1 = sigma_estimate1 * std_param[1] / r1 
            # curr_plot.annotate(r'$\bar{\sigma}=%.4G \pm %.2G$' % (sigma_estimate1, sigma_std1) ,
            # xy=(0.1, 0.12), xycoords='axes fraction', fontsize=20)
            curr_plot.annotate("Blocks: %.0f" % self.total_bl_nr, xy=(0.3, 0.7), xycoords='axes fraction', fontsize=16)
            curr_plot.locator_params(axis='x', nbins=5)
        
        f.text(0.5, 0.04, 'Distance [km]', ha='center', va='center', fontsize=25)
        f.text(0.06, 0.5, 'Shared blocks per pair', ha='center', va='center', rotation='vertical', fontsize=25)
        # plt.xlabel('Initial Distance', fontsize=25)
        # plt.ylabel('Shared blocks', fontsize=25)
        
        print("C-estimate: " + str(c_est))
        plt.show()
    
    def create_mle_model(self, model="constant", g=3537.4, start_param=0, diploid=0,
                         barrier_pos=[0,0], barrier_angle=0):
        '''Set MLE object. Set the model used for MLE model: what model to use
        g: genome length in cM -standard is human genome length four diploids
        start_param: What Start Parameters to Use
        all_chrom: Whether to use the formula for all chromosomes'''
            
        if model == "hetero":
            start_params = [1.0, 50, 1.0]  # 0.01, 70
            if start_param:
                start_params = start_param
            
            print("Initializing MLE-Object with Start Parameters: ")
            print(start_params)
            self.mle_object = MLE_Estim_Barrier(self.position_list, start_params,
                                    self.lin_block_sharing, self.lin_pair_nr, error_model=self.error_model, g=g / 100.0,
                                    diploid=diploid, barrier_pos=barrier_pos, barrier_angle=barrier_angle)
            self.estimates = start_params  # Best guess without doing anything. Used as start for Bootstrap
            return 0

        elif model == "constant":
            bl_shr_density = uniform_density
            start_params = [0.01, 70]
        elif model == "doomsday":
            bl_shr_density = dd_density
            start_params = [0.5, 70]
        elif model == "power_growth":
            bl_shr_density = powergrowth_density
            start_params = [1.0, 60, 1]
        elif model == "exp_const":
            bl_shr_density = exp_con_density
            start_params = [0.02, 60, 0.2, 0.01]
        elif model == "ddd":
            bl_shr_density = powergrowth_density_dd
            start_params = [0.001530, 60, 1, 0]
        else: 
            print("No suitable function found")
            
        if not self.all_chrom:
            bl_shr_density = partial(bl_shr_density, g=g / 100.0)  # Set the genome length (in Morgan!)
        
        # For chromosomal edge Effects:
        # From http://www.nature.com/ng/journal/v31/n3/pdf/ng917.pdf
        gss = np.array([270.27, 257.48, 218.17, 202.8, 205.69, 189.6, 179.34, 158.94, 157.73, 176.01,
             152.45, 171.09, 128.6, 118.49, 128.76, 128.86, 135.04, 120.59, 109.73, 98.35, 61.9, 65.86])  # All human chromosome lengths
        # gss = np.array([3537.4, ])  # For testing
        
        if self.all_chrom:  # Do the sum for multiple chromosomes. 
            temp_dens = partial(all_chromosomes, gs=gss / 100.0)  # Diploid Factor 4 is in all_chromosomes!
            bl_shr_density = partial(temp_dens, bl_density=bl_shr_density)  # Fix function
        
        
        
        if start_param:  # In case start params are given override
            start_params = start_param
        # Create MLE_estimation object. First endogenous Second exogenous Variables:
        self.mle_object = MLE_estim_error(bl_shr_density, start_params, self.lin_dists,
                                          self.lin_block_sharing, self.lin_pair_nr, error_model=self.error_model) 
        self.estimates = start_params  # Best guess without doing anything. Used as start for Bootstrap
    
    
    def mle_analysis_error(self):
        '''Does a maximum likelihood analysis with the full error model. Parameters can be found there
        Param[0] always C; Param[1] always sigma'''
        ml_estimator = self.mle_object 
        print("Doing fit...")
        results = ml_estimator.fit()  # method="nelder-mead"
        # results0 = ml_estimator.fit(method="BFGS")  # Do the actual fit. method="BFGS" possible
        try:
            self.estimates = results.params  # Save the paramter estimates (0: c 1:sigma ...)
            self.ci_s = results.conf_int()
            fisher_info = np.matrix(ml_estimator.hessian(results.params))  # Get the Fisher Info matrix
            corr_mat = cov2corr(-fisher_info.I)
            print(corr_mat)
            stds = np.sqrt(np.diag(-fisher_info.I))
            self.stds = stds  # Save estimated STDS
        except:  # In case one fails to get Confidence Intervalls
            raise warnings.warn("Failed to get Confidence Interval", RuntimeWarning)
            pass 
            
        for i in range(len(results.params)):
            print("Parameter %i: %.6f" % (i, results.params[i]))
            # print("Estimated STD: %.6f" % stds[i]) 
        print(results.summary())  # Give out the results.
        self.mle_object = ml_estimator  # Remember the mle-estimation object.
        
    def plot_fitted_data_error(self):
        '''Plot fit of full model to binned data set.'''
        f, axarr = plt.subplots(2, 2, sharex=True)  # Create sub-plots
        # intervals = ([3.0, 3.3], [3.3, 3.7], [3.7, 4.2], [4.2, 4.8], [4.8, 5.5], [5.5, 6.5], [6.5, 8], [8, 10])  # Set the interval-list
        # intervals = ([4, 4.5], [4.5, 5.2], [5.2, 6.5], [6.5, 8], [8, 10], [10, 12], [12, 14], [14, 18])  # Set the interval-list
        # intervals = ([4.0, 4.4], [4.4, 4.9], [4.9, 5.5], [5.5, 6.2], [6.2, 7], [7, 8.4], [8.4, 10], [10, 12])  # Set the interval-list
        intervals = ([4, 5], [5, 6.5], [6.5, 8], [8, 12])  # Set the interval-list
        mle_estim = self.mle_object  # Reload the object which did the MLE-Estimate
        # bins = mle_estim.mid_bins - 0.5 * mle_estim.bin_width
        # distances = [2, 10, 20, 30, 40, 50, 60]  # Distances to use for binning
        for i in range(0, 4):  # Loop through interval list
            curr_plot = axarr[i / 2, i % 2]  # Set current plot
            interval = intervals[i] 
            int_len = interval[1] - interval[0]  # Calculate the length of an interval
            (bin_mdist, mean_sharing, errors) = self.analyze_bin_ibd(interval[0], interval[1], show=False)  # Extract relevant binned data   
            x, y, error = bin_mdist, mean_sharing / int_len, errors / int_len
            x_plot = np.linspace(min(x), max(x), 100)
            y_plot_est = mle_estim.get_bl_shr_interval(interval, x_plot, self.estimates) / int_len  # Block sharing per pair and cM
                
            curr_plot.set_yscale('log')
            l2, = curr_plot.semilogy(x_plot, y_plot_est, 'r-.', linewidth=2, alpha=0.85)  # Plot of exact fit
            l1 = curr_plot.errorbar(x, y, yerr=error, fmt='go', linewidth=2, alpha=0.85)
            # curr_plot.set_ylim([min(y) / 3, max(y) * 3])
            # curr_plot.set_ylim(10 ** (-6), 0.1) # set consistent limits for nicer plots
            curr_plot.set_ylim([0.00001, 0.1])  # For the human analysis
            curr_plot.set_ylim([0.0005, 0.5])  # For the human analysis
            # curr_plot.set_xlim(0, 60)
            curr_plot.set_title("Interval: " + str(interval) + " cM")
            
            curr_plot.annotate("Blocks: %.0f" % self.total_bl_nr, xy=(0.6, 0.8), xycoords='axes fraction', fontsize=18)
        f.legend((l1, l2), ('Binned IBD-sharing', 'Fitted Bessel decay'), loc='upper center')
        f.text(0.5, 0.04, 'Distance', ha='center', va='center', fontsize=25)
        f.text(0.06, 0.5, 'Shared blocks per p. and cM', ha='center', va='center', rotation='vertical', fontsize=25)
        # plt.tight_layout()
        plt.show() 
        
    def plot_allin_one(self):
        '''Plot function to empirically plot best estimates for binned data in one window'''
        # intervals = ([3.0, 3.3], [3.3, 3.7], [3.7, 4.2], [4.2, 4.8], [4.8, 5.5], [5.5, 6.5], [6.5, 8], [8, 10])  # Set the interval-list
        # intervals = ([4, 4.5], [4.5, 5.2], [5.2, 6.5], [6.5, 8], [8, 10], [10, 12], [12, 14], [14, 18])  # Set the interval-list
        # intervals = ([4.0, 4.4], [4.4, 4.9], [4.9, 5.5], [5.5, 6.2], [6.2, 7], [7, 8.4], [8.4, 10], [10, 12])  # Set the interval-list
        intervals = ([4, 5], [5, 6.5], [6.5, 8], [8, 12])  # Set the interval-list
        # bins = mle_estim.mid_bins - 0.5 * mle_estim.bin_width
        # distances = [2, 10, 20, 30, 40, 50, 60]  # Distances to use for binning
        
        cons, dd, pg = self.get_all_estimates()  # Run and load the mle  objects
        l0 = [0 for _ in intervals]  # Placeholder for labels
        labels = [0 for _ in intervals]
        
        plt.figure()  
        plt.yscale('log')
        plt.ylim([0.0003, 1])  # For the human analysis
        # plt.title("Interval: " + str(interval) + " cM")
        x_plot = np.linspace(200, 1100, 200)  # Generate the x-axis
        plt.xlabel('Distance [km]', fontsize=25)
        plt.ylabel('Shared blocks per p. and cM', fontsize=25)
        c = ['r', 'y', 'c', 'b']
        for i in range(len(intervals)):
            interval = intervals[i] 
            int_len = interval[1] - interval[0]  # Calculate the length of an interval
            (bin_mdist, mean_sharing, errors) = self.analyze_bin_ibd(interval[0], interval[1], show=False)  # Extract binned data within interval   
            x, y, error = bin_mdist, mean_sharing / int_len, errors / int_len  # Normalize properly
            
            # Plot the theory predictions 
            l1, = plt.semilogy(x_plot, cons.get_bl_shr_interval(interval, x_plot) / int_len,
                              c[i] + ':', linewidth=2, alpha=0.85) 
            l2, = plt.semilogy(x_plot, dd.get_bl_shr_interval(interval, x_plot) / int_len
                               , c[i] + '--', linewidth=2, alpha=0.85)  # Plot of exact fit
            l3, = plt.semilogy(x_plot, pg.get_bl_shr_interval(interval, x_plot) / int_len
                              , c[i] + '-.', linewidth=3, alpha=0.85)
            
            # Plot the Block-Sharing:
            l0[i] = plt.errorbar(x, y, yerr=error, fmt=c[i] + 'o', linewidth=2, alpha=0.85)
            labels[i] = "%s cM: Blocks: %.0f" % (str(interval), self.total_bl_nr)
            
        f1 = plt.legend((l1, l2, l3), (r'$D=C$', r'$D=C/t$', r'$D=Ct^{-\beta}$'), loc=(0.25, 0.8))
        plt.gca().add_artist(f1)  # Add the legend manually to the current Axes.
        plt.legend(l0, labels, loc="upper right")  # Block Lengths
        # plt.tight_layout()
        plt.show() 
    
    def get_all_estimates(self):
        '''Function that get estimates for three models. Return the three fitted MLE-objects'''
        self.create_mle_model("constant")  # Create the object
        self.mle_analysis_error()  # Do the fit
        cons = deepcopy(self.mle_object)  # Save it
        
        self.create_mle_model("doomsday")
        self.mle_analysis_error()
        dd = deepcopy(self.mle_object)
           
        self.create_mle_model("power_growth")
        self.mle_analysis_error()
        pg = deepcopy(self.mle_object)

        return([cons, dd, pg])
    
    def boots_trap_ctry(self, nr=100):
        '''Does a bootstrap with nr run over countries pairs'''
        bl_shr_density = self.mle_object.density_fun  # Extract the last used block-sharing function
        start_params = self.estimates  # Extract the last inferred parameters                    
        res = []  # Vector for bootstrap estimates
        lin_shr, lin_d, lin_nr = self.lin_block_sharing, self.lin_dists, self.lin_pair_nr
        
        for _ in range(nr):
            r_ind = np.random.randint(len(lin_d), size=len(lin_d))  # Get random resampling
            ml_est_temp = MLE_estim_error(bl_shr_density, start_params, lin_d[r_ind],
                                          lin_shr[r_ind], lin_nr[r_ind], error_model=self.error_model) 
            temp_results = ml_est_temp.fit()  
            res.append(temp_results.params)
        res = np.array(res)
        self.btst_estimates = res  # Save the results
        self.analyse_bts_results(res)  # Analyse the results-vector
    
    def boot_trap_blocks(self, nr=10):
        '''Do a boots trap over all block pairs. Resample all blocks; first multinomial per pop-per and
        then within population. Nr: Number of boots-trap runs. Save results to analysis-object'''
        bl_shr_density = self.mle_object.density_fun  # Extract the last used block-sharing function
        
        # block_l, block_u = self.mle_object.min_len, self.mle_object.max_len
        start_params = self.estimates  # Extract the last inferred parameters (to have a good start)
        res = []  # Vector for bootstrap estimates
        
        # (linear_dist, _, pair_nr, _) = self.return_linearized_data(block_l, block_u)  # First linearize the data as usual
        # exog = np.column_stack((linear_dist, pair_nr))  # Stack the exogenous variables together
        
        len_list = np.array([len(i) for i in self.lin_block_sharing]).astype(np.float)  # Nr of shared blocks per Ctry pair
        # nr_pairs = float(np.sum(len_list))  # How many pairs in total
        
        
        # Temporally add blocks to empty lists (only here) so that in np.choice no bugs occurs
        lin_bls = [b_list if (len(b_list) > 0) else np.array([1, ]) for b_list in self.lin_block_sharing]
        
        for i in range(nr):  # Do the bootstrap runs
            # Create the random resampling:
            # First: Number of new blocks per Ctr-pair:
            bt_list = np.array([np.random.poisson(nr) for nr in len_list])
            # bt_list = np.random.multinomial(nr_pairs, len_list / nr_pairs)  # Resample nr shrd per ctry pair
            lin_bs = [np.random.choice(lin_bls[j], bt_list[j]) for j in range(len(len_list))]  # Within country resample
            lin_bs = np.array(lin_bs)
            ml_estimator_temp = MLE_estim_error(bl_shr_density, start_params, self.lin_dists,
                                                lin_bs, self.lin_pair_nr, error_model=self.error_model) 
            temp_results = ml_estimator_temp.fit()  
            res.append(temp_results.params)
        res = np.array(res)
        self.btst_estimates = res  # Save the results
        self.analyse_bts_results(res)  # Analyse the results-vector    
        
    def plot_loglike_surface(self, nr_intervals=15, params=0):
        '''Creates a likelihood surface for list of param. 
        rows: parameters 2 columns: upper and lower limit
        nr_intervals: Number of intervals
        If not paramst given take 7 estimated stds'''
        if params == 0:
            params0 = self.estimates[0] - 7 * self.stds[0], self.estimates[0] + 7 * self.stds[0]
            params1 = self.estimates[1] - 7 * self.stds[1], self.estimates[1] + 7 * self.stds[1]
            params = [params0, params1]
        params = np.array(params)
        x_vec = np.linspace(params[0, 0], params[0, 1], nr_intervals)
        y_vec = np.linspace(params[1, 0], params[1, 1], nr_intervals)
        xv, yv = np.meshgrid(x_vec, y_vec)
        xt, yt = xv.flatten(), yv.flatten()  # Linearize for list generation
        
        z = [self.mle_object.loglike([xt[i], yt[i]]) for i in range(len(xt))]  # Calc Log Likelihoods
        z = np.ceil(z)  # Round up for better plotting   
        levels = np.arange(max(z) - 30, max(z) + 1, 2)  # Every two likelihood units
        
        plt.figure()
        ax = plt.contourf(xv, yv, z.reshape((nr_intervals, nr_intervals)), levels=levels, alpha=0.8)
        plt.plot(self.estimates[0], self.estimates[1], 'ko')
        
        # plt.clabel(ax, inline=1, fontsize=10)
        plt.colorbar(ax, format="%i")
        plt.title("Log Likelihood Surface", fontsize=20)
        plt.xlabel(r"$D_e$", fontsize=20)
        plt.ylabel(r"$\sigma$", fontsize=20)
        
        if len(self.btst_estimates) > 0:  # In case there are bootstrap results plot them.
            plt.scatter(self.btst_estimates[:, 0], self.btst_estimates[:, 1], marker='x')    
        plt.show()
        
    def jack_knife_ctries(self):
        '''Fit model without certain countries; and predict their residuals
        Basically do a jack-knive'''
        # Copy the relevant original data - we dont want to overwrite it
        pw_distances, pw_block_sharing = np.copy(self.pw_distances), np.copy(self.pw_block_sharing)
        nr_individuals = np.copy(self.nr_individuals)
        
        # Extract relevant info from the MLE-object
        bl_shr_density = self.mle_object.density_fun  # Extract the last used block-sharing function
        start_params = self.estimates  # Extract the last inferred parameters  
        res = []  # Vector for the Jack-Knive estimates
                          
        for i in range(len(self.countries)):  # Iterate over all countries
            # Delete rows and columns associated to that raw. Use adv. indexing:
            indices = np.array([1 if j != i else 0 for j in range(len(nr_individuals))]).astype("bool")   
            self.nr_individuals = nr_individuals[indices]
            self.pw_distances = pw_distances[:, indices][indices, :]
            self.pw_block_sharing = pw_block_sharing[indices, :][:, indices]
            
            # First linearize the data as usual. This updates self.lin_block_sharing
            (linear_dist, _, pair_nr, _) = self.return_linearized_data(3.0, 150.0)  
            ml_estimator_temp = MLE_estim_error(bl_shr_density, start_params, linear_dist,
                                                self.lin_block_sharing, pair_nr, error_model=self.error_model) 
            temp_results = ml_estimator_temp.fit()  
            res.append(temp_results.params)
        res = np.array(res)
        # Restore original values:
        self.pw_distances, self.pw_block_sharing = pw_distances, pw_block_sharing
        self.nr_individuals = nr_individuals
        
        self.btst_estimates = res  # Save the results
        self.analyse_bts_results(res)  # Analyse the results-vector
        
        print(np.column_stack((self.countries, res)))  # Give out results per country
        
        self.get_country_residuals()
       
    def get_country_residuals(self):
        '''Get the residuals for a county. Needs run of Jack-Knife (typically called from there)'''
        intervals = [[4.0, 6.0], [6.0, 8.0], [8.0, 12.0]]
                
        pw_distances = self.pw_distances  # Load full pairwise distance Matrix
        # Get empirical block sharing matrix for every interval
        pw_bl_shr = [self.calc_nr_shr_bl(i[0], i[1]) for i in intervals] 
        
        results_pred = []  # Vector for the results
        results_emp = []
        for i in range(len(self.countries)):  # Iterate over every country
            indices = np.array([1 if j != i else 0 for j in range(len(self.countries))]).astype("bool") 
            params = self.btst_estimates[i, :]  # Load the estimated parameters from jack-knive
            nr_pairs = np.array(self.nr_individuals[indices] * self.nr_individuals[i])  # Extract number of pairs
            pw_dists = pw_distances[i, indices] + pw_distances[indices, i]  # Distances to other countries
            pw_bl_shr_prune = [np.sum(k[i, indices] + k[indices, i]) for k in pw_bl_shr]  # Get empirical block sharing
            
            # Calculate the estimated block-sharing with every other country:
            blocks_thr = []
            for interval in intervals:
                bs_pp = self.mle_object.get_bl_shr_interval(interval, pw_dists, params) 
                blocks_thr.append(np.sum(nr_pairs * bs_pp))  # Multiply pp-sharing by number of pairs
            results_pred.append(blocks_thr)
            results_emp.append(pw_bl_shr_prune)
            
            print("\nEstimated block-sharing %s:" % self.countries[i])
            for k in range(len(intervals)):
                print("True: %.1f Estimated %f " % (pw_bl_shr_prune[k], blocks_thr[k]))
                
        results_pred = np.array(results_pred)
        results_emp = np.array(results_emp)
        print(results_pred)
        self.visualize_residuals(results_pred, results_emp, self.countries)  # Visualize Residuals
    
    def visualize_residuals(self, res_th, res_emp, countries):
        '''Method for visualizing the residuals'''
        k = len(countries)
        x_base = np.array([7 * i for i in range(k)])  # Basis X-value
        w = 1  # Bar width
        opacity = 0.3
        opacity1 = 0.6
        
        err_th = np.sqrt(res_th)
        
        plt.figure()
        plt.bar(x_base, res_th[:, 0], w, yerr=err_th[:, 0], alpha=opacity, color='r', label="4-6 cM")
        plt.bar(x_base + 2 * w, res_th[:, 1], w, yerr=err_th[:, 1], alpha=opacity, color='m', label="6-8 cM")
        plt.bar(x_base + 4 * w, res_th[:, 2], w, yerr=err_th[:, 2], alpha=opacity, color='b', label="8-12 cM")
        
        plt.bar(x_base + w, res_emp[:, 0], w, alpha=opacity1, color='r', label="4-6 cM")
        plt.bar(x_base + 3 * w, res_emp[:, 1], w, alpha=opacity1, color='m', label="6-8 cM")
        plt.bar(x_base + 5 * w, res_emp[:, 2], w, alpha=opacity1, color='b', label="8-12 cM")
        plt.xticks(x_base + 3.5, countries, rotation='vertical')  # Put out the Countrie Labels
        plt.legend(loc="upper left")
        plt.grid()
        plt.show()
    
    def calculate_pw_residuals(self, intervals=0, params=0):
        '''Analyze the residual-matrix for pairs of countries'''
        if intervals == 0: intervals = [[4.0, 6.0], [6.0, 8.0], [8.0, 12.0]]  # Default Values
        if params == 0: params = self.estimates
        ctrs = self.countries  # Load the country list
        k = len(ctrs)  # Get the total number of countries
        
        pw_dist = self.pw_distances  # PW Distances
        pw_bl_shr = [self.calc_nr_shr_bl(i[0], i[1]) for i in intervals]  # Calc. emp. block-shr matrices
        # Get Matrix of pairwise individuals
        ind_mat = np.array([[i * j for j in self.nr_individuals] for i in self.nr_individuals])
        emp_shr, th_shr = np.zeros((k, k, 3)), np.zeros((k, k, 3))  # Create-block sharing matrices  
        
        for i in range(k):  # Loop over all pairwise countries
            for j in range(i):                  
                for l in range(len(intervals)):  # Loop over the intervals
                    interval = intervals[l]
                    emp_shr[i, j, l] = pw_bl_shr[l][i, j]  # Record empirical block-sharing
                    
                    # Get theoretical pairwise block sharing
                    bs_pp = self.mle_object.get_bl_shr_interval(interval, [pw_dist[i, j], ], params) 
                    th_shr[i, j, l] = (ind_mat[i, j] * bs_pp)  # Multiply pp-sharing by number of pairs
                    print("Block Shr %s - Block Shr %s" % (ctrs[i], ctrs[j]))
                    print("Expected: %.4f Actual: %i" % (th_shr[i, j, l], emp_shr[i, j, l]))
                    
        for l in range(len(intervals)):
            self.plot_pw_residuals(th_shr[:, :, l], emp_shr[:, :, l], intervals[l], ctrs)
            
    def plot_pw_residuals(self, thr_mat, emp_mat, interval, ctrs):
        '''Method for plotting the pairwise residuals'''
        k = len(ctrs)
        il1 = np.tril_indices(k, k=-1)  # Indices of lower triangular matrix
        ctrs = ["AT", "HU", "CZ", "SK", "SL", "PL", "RO", "BG", "MK", "BA", "HR", "RS", "ME", "AL"]
        
        
        z_vals = 2 * (np.sqrt(emp_mat) - np.sqrt(thr_mat))  # Do an Variance Stabilizing Transform
        mask = np.logical_not(np.tri(z_vals.shape[0], k=-1))
        z_vals = np.ma.array(z_vals, mask=mask)  # mask out the lower triangle

        plt.figure()
        plt.axis("equal")
        c = plt.pcolor(z_vals, vmin=-4.5, vmax=4.5, alpha=0.85)  # cmap RdBu
        ax = plt.gca()  # Get the axis object
        ax.xaxis.tick_top()  # Move x-axis ticks to top
        plt.xticks(np.arange(k) + 0.5, ctrs, rotation='vertical')
        plt.yticks(np.arange(k) + 0.5, ctrs, rotation='horizontal')
        # plt.tick_params(labelsize=6)
        ax.text(.6, .2, str(interval) + " cM",
        horizontalalignment='center',
        transform=ax.transAxes, fontsize=20)
        
        ths = thr_mat[il1]
        emps = emp_mat[il1]
        # sts = np.sqrt(ths)
        
#         def show_values(pc, fmt=r'$ %.2f \pm %.2f$' + "\n" + r'$%i$'):    # Comment out for talk rep
#             pc.update_scalarmappable()
#             ax = pc.get_axes()
#             for p, color, thr, std, emp in izip(pc.get_paths(), pc.get_facecolors(), ths, sts, emps):  # pc.get_array()
#                 x, y = p.vertices[:-2, :].mean(0)
#                 if np.all(color[1] > 0.15):
#                     color = (0.0, 0.0, 0.0)
#                 else:
#                     color = (1.0, 1.0, 1.0)
#                 ax.text(x, y, fmt % (thr, std, emp), ha="center", va="center", color=color)
                
                
        def show_values(pc, fmt=r'$ %.1f $' + "\n" + r'$%i$'):  # For talk w/o stds
            pc.update_scalarmappable()
            ax = pc.get_axes()
            for p, color, thr, emp in izip(pc.get_paths(), pc.get_facecolors(), ths, emps):  # pc.get_array()
                x, y = p.vertices[:-2, :].mean(0)
                if np.all(color[1] > 0.15):
                    color = (0.0, 0.0, 0.0)
                else:
                    color = (1.0, 1.0, 1.0)
                ax.text(x, y, fmt % (thr, emp), ha="center", va="center", color=color)
               
        show_values(c)
        plt.colorbar()
        plt.show()        
                       
    def analyse_bts_results(self, res):
        '''Method for analyzing bootstrap/jack-knife results. Gives out various summary statistics'''
        res = np.array(res)
        self.btst_estimates = res  # Save the results
        mean_bt, stds_bt = res.mean(0), res.std(0) 
        print(res)
        print("\nCorrelation Matrix:")
        print(np.corrcoef(res, rowvar=0))  # Print correlation matrix
         
        print("Mean Jack-Knive:")
        print(mean_bt)
        print("STD Jack-Knive:")
        print(stds_bt)
        
        for i in range(len(self.estimates)):
            print("\n Parameter %i. \nMean: %.6f" % (i, mean_bt[i]))
            print("Std: %.8f" % stds_bt[i])
            print("Cfd Interval: %.6f - %.6f" % (np.percentile(res[:, i], 2.5), np.percentile(res[:, i], 97.5)))   
            
    def show_IBD_hist(self):
        '''Method to generate histogram of IBD-sharing of various lengths'''
        print(self.lin_block_sharing)
        # First generate vector of all shared blocks:
        
        a = [item for sublist in self.lin_block_sharing for item in sublist]  # First generate list of all items
        bins = np.arange(4, 20, 1)
        plt.figure()
        plt.hist(a, bins)
        plt.xlabel("Block length [cM]", fontsize=20)
        plt.ylabel("Number of blocks", fontsize=20)
        plt.title("Histogram of block lengths", fontsize=20)  # Distribution of block lengths in analysis
        plt.show()
        
    
    
    def centering_positions(self, positions, barrier_location):
        '''
        Change of coordinates so that the barrier is at x=0
        '''
        center = barrier_location[0]
        angle = barrier_location[1]
        c = np.cos(angle)
        s = np.sin(angle)
        rotation_matrix = np.array([[c, -s], [s, c]])
        return np.matmul(positions - center, rotation_matrix)    
    
    
#     def which_times(self):
#         '''Calculate a summary of coalesence times of the coalescence times of blocks 
#         generated under the model. Bin blocks in different categorie'''
#         (linear_dist, _, pair_nr, _) = self.return_linearized_data(3.0, 150.0)  # First linearize the data (and calc lin_block_sharing)
#         t = np.linspace(0, 200, 200).astype('float')  # The times
#         y_plot = np.array([0 for _ in t]).astype('float')
#         
#         for i in range(len(pair_nr)):  # Iterate over all pairwise countries.
#             params = np.append(self.estimates, 1)  # Append one for Doomsday Model
#             y_plot += powergrowth_density_t_l0(4.0, linear_dist[i], t, params) * pair_nr[i]
#         
#         print(y_plot)    
#         plt.figure()
#         plt.plot(t, y_plot)
#         plt.xlabel("Time (Generations)")
#         plt.ylabel("Block sharing probability")
#         plt.show()           
################################################################################
        
def bessel_decay(x, C, r):
    '''Fit to expected decay curve in 2d (C absolute value, r rate of decay)'''
    return(C * x * kv(1, r * x))       
    
def bessel_decay2(x, C, r):
    '''Fit to expected decay of certain block length C absolute Value, r rate of decay'''
    return(C * x * x * kv(2, r * x)) 

def bd_basis(l, r, D, sigma, b):
    '''Bessel decay for power growth model and G=1
    Central Ingredient for further calculations. Return density per cM'''
    C = 2 ** (-3 - 3 * b / 2.0) / (np.pi * sigma ** 2 * D)  # The constant in front

    b_l = C * (r / (np.sqrt(l) * sigma)) ** (2 + b) * kv(2 + b,
    np.sqrt(2.0 * l) * r / sigma)
    return b_l / 100.0  # Factor for density in centi Morgan

def bessel_decay_interval(r, C, sigma, interval, mu=0):
    '''Gives Bessel-Decay in a given interval If r vector returns vector'''
    l = 2.0 / (1.0 / interval[0] + 1.0 / interval[1])  # Calculate Harmonic Mean
    # l=interval[0]
    l_e = l - mu / 2.0  # Update for population growth!
    b_l = C * r ** 2 / (2 * l_e / 100.0 * sigma ** 2) * kv(2, np.sqrt(2 * l_e / 100.0) * r / sigma)
    return b_l * (interval[1] - interval[0]) / 100.0

def dd_decay_interval(r, C, sigma, interval):
    '''Gives the Doomsday decay of a given interval. If r vector return vector'''
    l = 2.0 / (1.0 / interval[0] + 1.0 / interval[1])  # Calculate Harmonic Mean
    b_l = C * r ** 3 / (4.0 * np.sqrt(2) * (l / 100 * sigma ** 2) ** (3.0 / 2.0)) * kv(3, np.sqrt(2.0 * l / 100.0) * r / sigma)
    return b_l * (interval[1] - interval[0]) / 100.0

    
def uniform_density(l, r, params, g):
    '''Gives uniform density per cM(!) If l vector return vector.
    Includes chromosomal edge effects'''
    l = l / 100.0  # Switch to Morgan
    G = g  # 35.374 for human data
    D = params[0]
    sigma = params[1]
     
    b_l = (G - l) * bd_basis(l, r, D, sigma, 0) + bd_basis(l, r, D, sigma, -1)
    return b_l

def dd_density(l, r, params, g):
    '''Gives the Doomsday density per cM(!) If l vector return vector
    Includes chromosomal edge effects'''
    l = l / 100.0  # Switch to Morgan
    G = g  # 35.374 for human data
    D = params[0]
    sigma = params[1]
     
    b_l = (G - l) * bd_basis(l, r, D, sigma, 1) + bd_basis(l, r, D, sigma, 0)
    return b_l

def powergrowth_density(l, r, params, g):
    '''Gives the Powergrowth density of block sharing per cM(!) If l vector return vector
    Includes chromosomal edge effects'''
    l = l / 100.0  # Switch to Morgan
    G = g  # 35.374 for human data
    D = params[0]
    sigma = params[1]
    beta = params[2]
     
    b_l = (G - l) * bd_basis(l, r, D, sigma, beta) + bd_basis(l, r, D, sigma, beta - 1)
    return b_l

def all_chromosomes(l, r, params, bl_density, gs):
    '''Gives density per cM(!) over all chromosomes in gs. 
    Assumes diploids (that's the factor four)
    If r vector - returns vector
    '''
    res = 4.0 * np.sum([bl_density(l, r, params, gi) for gi in gs], axis=0)  # Sum over all chromosomes
    # Problem np.sum initially summed over r; but only need to sum over gi! SOLVED
    return(res)

# Original powergrowth density
# def powergrowth_density(l, r, params, g):
#     '''Gives uniform density per cM(!) If l vector return vector'''
#     G = g  #  35.374 for human data
#     D = params[0]
#     sigma = params[1]
#     b = params[2]
#     C = G / (np.pi * sigma ** 2 * D)  # The constant in front
# 
#     b_l = C * 2 ** (-3 - 3 * b / 2.0) * (r / (np.sqrt(l / 100.0) * sigma)) ** (2 + b) * kv(2 + b,
#     np.sqrt(2.0 * l / 100.0) * r / sigma)
#     return b_l / 100.0  # Factor for density in centi Morgan


# Original doomsday density
# def dd_density(l, r, params, g):
#     '''Gives the Doomsday density per cM(!) If l vector return vector'''
#     G = g  # Manually entered. REMOOOOOVVE 35.374 for human data
#     D = params[0]
#     sigma = params[1]
#     C = G / (4 * np.pi * sigma ** 2 * D)  # The constant in front
#     b_l = C * r ** 3 / (4.0 * np.sqrt(2) * (l / 100.0 * sigma ** 2) ** (3 / 2.0)) * kv(3, np.sqrt(2.0 * l / 100.0) * r / sigma)
#     return b_l / 100.0  # Factor for density in centi Morgan
    


# Original uniform density
# def uniform_density(l, r, params, g):
#     '''Gives uniform density per cM(!) If l vector return vector'''
#     G = g  # 35.374 for human data
#     D = params[0]
#     sigma = params[1]
#     
#     C = G / (4 * np.pi * sigma ** 2 * D)  # The constant in front
#     b_l = C * r ** 2 / (2.0 * (l / 100.0 * sigma ** 2)) * kv(2, np.sqrt(2.0 * l / 100.0) * r / sigma)
#     return b_l / 100.0
    

def powergrowth_density_t_l0(l0, r, t, params, g):
    '''Gives the expected number of blocks longer l0 at time t.'''
    D = params[0]
    sigma = params[1]
    b = params[2]
    C = g / (4 * np.pi * sigma ** 2 * D)  # The constant in front
    b_l = C * t ** b * np.exp(-r ** 2.0 / (4.0 * sigma ** 2 * t) - t * 2.0 * l0 / 100.0)
    return b_l
    
def exp_con_density(l, r, params, g):
    '''Gives density for hyperexponential-constant growth mix. (per cM) If l vector return vector''' 
    D = params[0]
    sigma = params[1]
    mu = params[2]
    E = params[3]  # Second constant
    C = g / (4 * np.pi * sigma ** 2 * D)  # The constant in front
    return uniform_density(l, r, [C, sigma]) - uniform_density(l + mu / 2.0, r, [E, sigma])  # Reduce it to uniform case

def powergrowth_density_dd(l, r, params, g):
    '''Gives a powergrowth with a dooms day'''
    D = params[0]
    sigma = params[1]
    b = params[2]
    T = params[3]
    C = g / (4 * np.pi * sigma ** 2 * D)  # The constant in front
    return powergrowth_density(l, r, [C , sigma, b]) + uniform_density(l, r, [T * C, sigma])


# def from_C_to_D_e(C, sigma):
#   '''Calculates the actual density from C and sigma'''
#  G = 35.374  # Sex average of human genome. In Morgan!!
# return 2 * G / (4 * np.pi * sigma ** 2 * 2 * C)


    

        
