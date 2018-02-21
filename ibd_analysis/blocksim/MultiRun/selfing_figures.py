'''
Created on Feb 7, 2018
Creates nice figures for the selfing-paper.
(I.e. IBD sharing in case there is selfing)
Produce Figures in standardized Size: Reason
@author: hringbauer
'''

import matplotlib.pyplot as plt
import numpy as np
import os as os

data_folder = "."  # The Folder where all the data is in.


###########################################
# Helper Functions to do the Loading of Results

def produce_path(run, folder, subfolder=None):
    '''Produces the Path of results'''
    if subfolder == None:
        path = data_folder + folder + "/estimate" + str(run).zfill(2) + ".csv"
    else:
        path = data_folder + folder + subfolder + "/estimate" + str(run).zfill(2) + ".csv"
    return path

def load_estimates(array, folder, subfolder=None, param=0):
        '''Function To load estimates.
        param: Which Parameter (Index) to load.
        array: Which Results (Nr!) one wants to load.
        Return Estimates. CI low and CI up as vectors
        as well as array which of the Results are actually there.
        Also visualizes it.'''
        array_there = []
        estimates = []
        ci_low = []
        ci_up = []
        
        for i in array:
            path = produce_path(i, folder, subfolder)
            exist = os.path.exists(path)  # Check if Data-Set is there. Return 1 if yes/ 0 if no.
            if exist:
                array_there.append(i)  # Indicate that Result was loaded
                data = np.loadtxt(path, delimiter='$')  # Load the Data
                estimates.append(data[param, 0])
                ci_low.append(data[param, 1])
                ci_up.append(data[param, 2])
            else:
                raise Warning("Cannot find %s" % path)
                
        estimates, ci_low, ci_up, array_there = np.array(estimates), np.array(ci_low), np.array(ci_up), np.array(array_there)
        return estimates, ci_low, ci_up, array_there

def argsort_bts(x, nr_bts):
    '''Arg Sorts a Vector within Bootstraps.
    Return: 
    Sorted indices
    Indices of true value'''
    assert(len(x) % nr_bts == 0)  # Check whether Bootstraps diveds nr of given array
    
    inds_sorted = np.zeros(len(x)).astype("int")
    true_inds = []
    
    # Iterate over batches
    for i in range(0, len(x), nr_bts):
        inds = range(i, i + nr_bts)
        
        inds_batch = np.argsort(x[inds])
        inds_sorted[inds] = inds_batch + i  # Return the sorted indices shifted.
        true_ind = np.where(inds_batch == 0)[0][0]  # Get the index of the true value.
        
        true_inds.append(true_ind + i)
        
    true_inds = np.array(true_inds)
    return (inds_sorted, true_inds)


###########################################
# Functions to do the Plots

def fig_fusing_time():
    '''Plot the time of Fusing of Blocks'''
    t_max = 9
    s = [0.5, 0.8, 0.95]  # The Values for Selfing
    q = 0.95  # Fraction of fusions.
    
    def selfing_t(t, s):
        '''Probability of Selfing at generation t'''
        p = (s / 2.0) ** t
        return p
    
    def median_time(s, q):
        '''Time until when fraction q of all fusings have occured'''
        t_med = np.log(1 - q) / (np.log(s) - np.log(2.0))
        return t_med
        
    gens = range(1, t_max)
    ps = [selfing_t(t, s[0]) for t in gens]
    ps1 = [selfing_t(t, s[1]) for t in gens]
    ps2 = [selfing_t(t, s[2]) for t in gens]
    
    print(ps)   
    t_med = median_time(s[0], q)
    t_med1 = median_time(s[1], q)
    t_med2 = median_time(s[2], q)
    print("%.2f Quantile: %.4f gens" % (q, t_med))
    
    lw = 3  # Width of the plot lines
    hw = 3  # Width of the vertical line
    
    colors = ["Gold", "Coral", "Crimson"]
    
    plt.figure(figsize=(7, 7))
    plt.plot(gens, ps, "ro-", color=colors[0], zorder=1, linewidth=lw, label=r"s$=%.2f$" % s[0])
    plt.plot(gens, ps1, "ro-", color=colors[1], zorder=1, linewidth=lw, label=r"s$=%.2f$" % s[1])
    plt.plot(gens, ps2, "ro-", color=colors[2], zorder=1, linewidth=lw, label=r"s$=%.2f$" % s[2])
    plt.xlabel("Generations back", size=18)
    plt.ylabel("Fusing Probability", size=18)
    plt.axvline(t_med, label="95% Quantile", color=colors[0], zorder=0, linewidth=hw, alpha=0.5)
    plt.axvline(t_med1, color=colors[1], zorder=0, linewidth=hw, alpha=0.5)
    plt.axvline(t_med2, color=colors[2], zorder=0, linewidth=hw, alpha=0.5)
    plt.legend(fontsize=18)
    # plt.title("Selfing Rate: %.2f" % s, fontsize=18)
    plt.show()

###########################################
# Do the actual Plots:
def calc_correction_factor(s):
    '''Calculates the correction Factor: 
    I.e. the fraction of Recombination Events that are effective.'''
    cf = np.sqrt((2.0 - 2.0 * s) / (2.0 - s))  # Fraction of Effectie Rec. Events
    return cf
    
def fig_selfing_estimates():
    '''Load and plot the figures of estimates for different values of selfing.
    6 Values 50 replicates each'''
    # Load the Estimates
    selfing_rates = [0, 0.5, 0.7, 0.8, 0.9, 0.95]
    cfs = [calc_correction_factor(s) for s in selfing_rates]  # Calculates the Correction Factor
    replicates = 50
    
    array = range(300)  # To load the estimates
    folder = "/selfing"  # Selfing225 Selfing
    
    # Load the Dispersal Estimates:
    estimates, ci_low, ci_up, _ = load_estimates(array, folder, subfolder=None, param=1) 
    # Make absolute Errors:
    error_up = estimates - ci_low  # @UnusedVariable
    error_down = ci_up - estimates  # @UnusedVariable
    
    cs = ["Crimson", "Coral"]
    cs_corr = ["Blue", "LightBLue"]
    ms = 4
    ticks = [replicates / 2 + replicates * i for i in xrange(len(selfing_rates))]
    s_ticks = ["s=%.2f" % selfing_rates[i] for i in xrange(len(selfing_rates))]
    
    inds_sorted, _ = argsort_bts(estimates, replicates)  # Get the Indices for sorted
    plt.figure(figsize=(6, 4))
    
    # Plot the replicate batches:
    for i in xrange(len(selfing_rates)):
        c_i = i % 2  # Color Index
        c = cs[c_i]  # Load the color
        cc = cs_corr[c_i]  # Load the corrected Color
        x_inds = np.array(range(i * replicates, (i + 1) * replicates))
        inds = inds_sorted[x_inds]  # Extract Indices
        
        print(x_inds)
        print(estimates[inds])
        # plt.errorbar(x_inds, estimates[inds], yerr=[error_down[inds], error_up[inds]], color=c, zorder=1, marker='o')
        plt.plot(x_inds, estimates[inds], color=c, zorder=1, marker='o', linestyle="", markersize=ms)
        
        # Plot corrected Estimates:
        cf = cfs[i]  # The right correction Factor
        # plt.errorbar(x_inds, estimates[inds] * cf, yerr=[error_down[inds] * cf, error_up[inds] * cf], color=cc, zorder=1, marker='o')
        plt.plot(x_inds, estimates[inds] * cf, color=cc, zorder=1, marker='o', linestyle="", markersize=ms)
    # plt.errorbar(x_inds, estimates[inds], yerr=[error_down[inds], error_up[inds]], color=c, zorder=1, marker='o', label="Raw Estimate")
    plt.plot(x_inds, estimates[inds], color=c, zorder=1, marker='o', label="Raw Estimate", linestyle="", markersize=ms)
    # plt.errorbar(x_inds, estimates[inds] * cf, yerr=[error_down[inds] * cf, error_up[inds] * cf], color=cc, zorder=1, marker='o', label="Corrected")
    plt.plot(x_inds, estimates[inds] * cf, color=cc, zorder=1, marker='o', label="Corrected", linestyle="", markersize=ms)
        
    # Calculate the Correction Factor:
    plt.axhline(2.0, linewidth=2, color="green", zorder=0, label="True Value")  # Plot the True Value

    # plt.legend(loc="upper right")
    plt.xlabel("Dataset", fontsize=18)
    plt.ylabel(r"Estimated $\sigma$", fontsize=18)
    plt.ylim([0, 15])
    plt.xticks(ticks, s_ticks, fontsize=14)
    plt.legend(fontsize=18, loc="upper left")
    plt.yticks(fontsize=14)
    plt.show()



if __name__ == '__main__':
    #fig_fusing_time()  # Pic of Fusing time.
    fig_selfing_estimates()
    # estimates, ci_low, ci_up, _ = load_estimates([299, ], "/selfing", subfolder=None, param=1)
    # print(estimates) 
    
    
    
