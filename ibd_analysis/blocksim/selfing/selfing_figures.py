'''
Created on Feb 7, 2018
Creates figures for the selfing-paper.
(I.e. IBD sharing in case there is selfing)
Produce Figures in standardized Size and Color Scheme and saves them as PDF.
@author: hringbauer
'''

import sys
sys.path.append('..')
sys.path.append('../../analysis_popres')
from multi_runs import get_theory_sharing, into_bins, get_normalization_lindata  # @UnresolvedImport

import matplotlib.pyplot as plt
import cPickle as pickle
import numpy as np
import os as os

data_folder = "../MultiRun/"  # The Folder where all the data is saved into.
fig_folder = "Figs/"  # Where to save the Figures to

###########################################
# Helper Functions to do the Loading of Results


def produce_path(run, folder, subfolder=None, name=None):
    '''Produces the Path of results'''
    if name == None:
        name = "/estimate"
    if subfolder == None:
        path = data_folder + folder + name + str(run).zfill(2) + ".csv"
    else:
        path = data_folder + folder + subfolder + name + str(run).zfill(2) + ".csv"
    return path


def load_estimates(array, folder, subfolder=None, name=None, param=0):
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
            path = produce_path(i, folder, subfolder, name=name)
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
    
    save_name = fig_folder + "fusing_time.pdf"
    plt.savefig(save_name, bbox_inches='tight', pad_inches=0)
    plt.show()

###########################################
###########################################


def calc_correction_factor(s):
    '''Calculates the correction Factor: 
    I.e. the fraction of Recombination Events that are effective.'''
    cf = np.sqrt((2.0 - 2.0 * s) / (2.0 - s))  # Fraction of Effectie Rec. Events
    return cf

    
def fig_selfing_estimates(show=2, folder="/selfing", sigma=2):
    '''Load and plot the figures of estimates for different values of selfing.
    6 Values 50 replicates each.
    show=0 Only Estimates
    show=1 Also corrections
    show=3 Also Density corrections'''
    # Load the Estimates
    selfing_rates = [0, 0.5, 0.7, 0.8, 0.9, 0.95]
    # cfs = [calc_correction_factor(s) for s in selfing_rates]  # Calculates the Correction Factor
    replicates = 50
    lfs = 11  # Label Font Size
    array = range(300)  # To load the estimates

    cs = ["Crimson", "Coral"]
    cs_corr = ["Blue", "LightBLue"]
    ms = 4
    ticks = [replicates / 2 + replicates * i for i in xrange(len(selfing_rates))]
    s_ticks = ["s=%.2f" % selfing_rates[i] for i in xrange(len(selfing_rates))]

    ######################################################################
    # ## Make the actual Figure:
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))  # Make two subplots
    
    # Load Estimates for the first figure (Dispersal): 
    
    estimates, ci_low, ci_up, _ = load_estimates(array, folder, subfolder=None, param=1)  # @UnusedVariable
    estimates_c, _, _, _ = load_estimates(array, folder, name="/corrected", param=1)  # Corr. estimates
    inds_sorted, _ = argsort_bts(estimates, replicates)  # Get the Indices for sorted
    inds_sorted_c, _ = argsort_bts(estimates_c, replicates)  # Get the Indices for sorted
    
    # Plot the replicate batches:
    for i in xrange(len(selfing_rates)):
        c_i = i % 2  # Color Index
        c = cs[c_i]  # Load the color
        cc = cs_corr[c_i]  # Load the corrected Color
        x_inds = np.array(range(i * replicates, (i + 1) * replicates))
        inds = inds_sorted[x_inds]  # Extract Indices
        inds_c = inds_sorted_c[x_inds]
        
        # plt.errorbar(x_inds, estimates[inds], yerr=[error_down[inds], error_up[inds]], color=c, zorder=1, marker='o')
        ax1.plot(x_inds, estimates[inds], color=c, zorder=1, marker='o', linestyle="", markersize=ms)
        
        # Plot corrected Estimates:
        # cf = cfs[i]  # The right correction Factor
        # plt.errorbar(x_inds, estimates[inds] * cf, yerr=[error_down[inds] * cf, error_up[inds] * cf], color=cc, zorder=1, marker='o')
        if show >= 1:
            # plt.plot(x_inds, estimates[inds] * cf, color=cc, zorder=1, marker='o', linestyle="", markersize=ms)
            ax1.plot(x_inds, estimates_c[inds_c], color=cc, zorder=1, marker='o', linestyle="", markersize=ms)
    
    # Plot the last one again, for the labels:        
    ax1.plot(x_inds, estimates[inds], color=c, zorder=1, marker='o', label="Raw Estimate", linestyle="", markersize=ms)
    
    if show >= 1:
        ax1.plot(x_inds, estimates_c[inds_c], color=cc, zorder=1, marker='o', label="Corrected", linestyle="", markersize=ms)
        
    # Calculate the Correction Factor:
    ax1.axhline(sigma, linewidth=2, color="green", zorder=0, label="True Value")  # Plot the True Value

    # plt.legend(loc="upper right")
    ax1.set_xlabel("Dataset", fontsize=18)
    ax1.set_ylabel(r"Estimated $\sigma$", fontsize=18)
    ax1.set_ylim([0, 12])
    ax1.set_xticks(ticks)
    ax1.set_xticklabels(s_ticks)
    ax1.tick_params(labelsize=lfs)  # Could be axis=both
    
    ax1.legend(fontsize=16, loc="upper left")
    # plt.savefig("sigma2.pdf", bbox_inches='tight', pad_inches=0)
    
    ################################
    # Plot the estimates for Density:
    estimates, ci_low, ci_up, _ = load_estimates(array, folder, subfolder=None, param=0)  # @UnusedVariable
    estimates_c, _, _, _ = load_estimates(array, folder, subfolder=None, name="/corrected", param=0) 
    inds_sorted, _ = argsort_bts(estimates, replicates)  # Get the Indices for sorted
    inds_sorted_c, _ = argsort_bts(estimates_c, replicates)  # Get the Indices for the corrections
    
    # Plot the replicate batches:
    for i in xrange(len(selfing_rates)):
        c_i = i % 2  # Color Index
        c = cs[c_i]  # Load the color
        cc = cs_corr[c_i]  # Load the corrected Color
        x_inds = np.array(range(i * replicates, (i + 1) * replicates))
        inds = inds_sorted[x_inds]  # Extract Indices
        inds_c = inds_sorted_c[x_inds]
        
        ax2.plot(x_inds, estimates[inds], color=c, zorder=1, marker='o', linestyle="", markersize=ms)
        if show >= 1:
            ax2.plot(x_inds, estimates_c[inds_c], color=cc, zorder=1, marker='o', linestyle="", markersize=ms)
        true_val = 0.5 * (2.0 - selfing_rates[i])
        if show >= 2:
            ax2.plot([x_inds[0], x_inds[-1]], [true_val, true_val], color="green", zorder=0, linestyle="-", linewidth=2)
        
        # Plot corrected Estimates:
        # cf = cfs[i]  # The right correction Factor
        # plt.plot(x_inds, estimates[inds] * cf, color=cc, zorder=1, marker='o', linestyle="", markersize=ms)
    ax2.plot(x_inds, estimates[inds], color=c, zorder=1, marker='o', label="Raw Estimate", linestyle="", markersize=ms)
    
    if show >= 1:
        ax2.plot(x_inds, estimates_c[inds_c], color=cc, zorder=1, marker='o', label="Corrected", linestyle="", markersize=ms)
        ax2.plot([x_inds[0], x_inds[-1]], [true_val, true_val], color="green", zorder=0,
                 linestyle="-", linewidth=2, label="True Value")
    # plt.plot(x_inds, estimates[inds] * cf, color=cc, zorder=1, marker='o', label="Corrected", linestyle="", markersize=ms)
        
    # Calculate the Correction Factor:
    if show <= 1:
        ax2.axhline(1.0, linewidth=2, color="green", zorder=0, label="True Value")  # Plot the True Value

    # plt.legend(loc="upper right")
    ax2.set_xlabel("Dataset", fontsize=18)
    ax2.set_ylabel(r"Estimated $D_e$", fontsize=18)
    ax2.set_ylim([0, 1.3])
    ax2.set_xticks(ticks, s_ticks)
    ax2.set_xticks(ticks)
    ax2.set_xticklabels(s_ticks)
    ax2.tick_params(labelsize=lfs)  # Could be axis=both
    
    ax2.legend(fontsize=16, loc="upper right")
    
    # plt.savefig("D2.pdf", bbox_inches='tight', pad_inches=0)
    save_name = fig_folder + folder + ".pdf"  # The save name
    plt.savefig(save_name, bbox_inches='tight', pad_inches=0)  # Save without Boundaries
    plt.show()
    
###########################################
###########################################


def fig_selfing_emp_thr(distances=[], itvs=[], save_names="blocks", 
                        mr_folder="../MultiRun/selfing_block_save/",
                        svec=[], reps=25, sigma=2, D=1, b=0):
    """Plot theoretical against empirical block sharing for various rates of selfing.
    save_names: List of names where to find the Data.
    distances: List of [min, max] of distance bins.
    itvs: List of [min, max] of Intervalls.
    mr_folder: String of the Folder Name where block lengths are save.
    svec: Array of Selfing rates.
    reps: How many Replicates per selfing rate.
    sigma, D, b: Paramters of the run (b Growth rate parameter)
    """
    
    if len(distances) == 0:
        distances = [[2, 10], [10, 20], [20, 30], [30, 40], [40, 50], [50, 60]]  # Distances used for binning
    
    if len(itvs) == 0:
        itvs = [[5, 7], [7, 10], [10, 14], [14, 20]]  # Bins for the block length binning (in cM)
    
    if len(svec) == 0:
        svec = [0, 0.5, 0.8, 0.95]
    fs = 20
    lfs = 12
    
    c = ["Gold", "Coral", "Crimson", "Brown"]  # The colors of the lines
    
    ##########################
    ##########################
    # Load the block sharing:
    # for i in range(len(svec)):
        # reps*i, reps*(i+1)
    # Get effective Run Nr:
    run = 1
    #eff_run_nr = run % reps 
    eff_scenario = run // reps
    
    
    full_path = mr_folder + save_names + str(run).zfill(2) + ".p"
    #full_path = "../MultiRun/selfing_block_save/blocks01.p"
    pair_dist, pair_IBD, pair_nr, _ = pickle.load(open(full_path, "rb"))
    print("Total Nr IBD Blocks: %i" % np.sum([len(l) for l in pair_IBD]))
    print("Minimum Length of IBD block: %.2f" % np.nanmin(np.concatenate(pair_IBD)))
    
    distances = [[1, 6], [6, 12], [12, 18], [18, 24], [24, 30], [30, 36]]  # Distances used for binning
    itvs = [[5, 7], [7, 10], [10, 14], [14, 20]]  # Bins for the block length binning (in cM)
    
    # Import the theoretical sharing:
    s = svec[eff_scenario]
    cf = (2 - 2 * s) / (2 - s)  # Calculate the Correction Factor
    itvs_t = [[i[0] / cf, i[1] / cf] for i in itvs]  # Stretch Intervalls for the correction Factor
    
    # Now process that Sample to IBD in bins
    bl_nr = into_bins(pair_dist, pair_IBD, itvs_t, distances)  # Get the empirical data matrix
    nr_pairs_bins = get_normalization_lindata(distances, pair_dist, pair_nr)  # Get Nr. factor: Pairs of inds
    int_len = np.array([i[1] - i[0] for i in itvs])  # Length of the intervals - to normalize for that
    
    frac_sharing = bl_nr / (nr_pairs_bins * int_len[:, None])  # Calculate the fractions per pair        
        
    #############
    dist_means = np.array([np.mean(i) for i in distances])  # Mean distances
    thr_shr = get_theory_sharing(itvs, distances, sigma, b, D)  # Get predicted sharing
    
    plt.figure()
    for i in range(len(itvs)):
        lab = str(itvs[i]) + " cm"
        plt.plot(dist_means, thr_shr[i], color=c[i], label=lab, linewidth=2)
        plt.plot(dist_means, frac_sharing[i,:], "o")
        
    plt.yscale("log")
    plt.xlabel(r'Pw. Distance [$\sigma$]', fontsize=fs)
    plt.ylabel('IBD-blocks per pair and cM', fontsize=fs)
    plt.legend(fontsize=lfs)
    plt.show()
    
    save_name = fig_folder + "" + ".pdf"  # The save name
    plt.savefig(save_name, bbox_inches='tight', pad_inches=0)  # Save without Boundaries

    
def test_IBD_binning():
    full_path = "../MultiRun/selfing_block_save/blocks01.p"
    pair_dist, pair_IBD, pair_nr, _ = pickle.load(open(full_path, "rb"))
    print("Total Nr IBD Blocks")
    print(np.sum([len(l) for l in pair_IBD]))
    print("Minimum Length of IBD block:")
    print(np.nanmin(np.concatenate(pair_IBD)))
    
    distances = [[1, 6], [6, 12], [12, 18], [18, 24], [24, 30], [30, 36]]  # Distances used for binning
    itvs = [[5, 7], [7, 10], [10, 14], [14, 20]]  # Bins for the block length binning (in cM)
    
    # Now process that Sample to IBD in bins
    bl_nr = into_bins(pair_dist, pair_IBD, itvs, distances)  # Get the empirical data matrix
    nr_pairs_bins = get_normalization_lindata(distances, pair_dist, pair_nr)  # Get Nr. factor: Pairs of inds
    int_len = np.array([i[1] - i[0] for i in itvs])  # Length of the intervals - to normalize for that
    
    print(bl_nr)
    print(nr_pairs_bins)
    
    print(bl_nr / nr_pairs_bins)
    
    # mean_shr = np.mean(bl_nr, axis=0)
    # sts = np.std(bl_nr, axis=0)
    # emp_shr = mean_shr / nr_pairs_bins
    # emp_shr = emp_shr / int_len[:, None]
    # emp_sts = sts / nr_pairs_bins
    # emp_sts = emp_sts / int_len[:, None]
    

if __name__ == '__main__':
    # fig_fusing_time()  # Pic of Fusing time.
    # fig_selfing_estimates(show=2, folder="selfing_3-12cm", sigma=2)  # selfing_noshrink selfing_500cm selfing_3-12cm_sigma3 selfing_3-12cm_sigma3
    # fig_selfing_estimates(show=2, folder="selfing_3-12cm_sigma3", sigma=3)  # selfing_noshrink selfing_500cm selfing_3-12cm_sigma3 selfing_3-12cm_sigma3
    fig_selfing_emp_thr()  # Make a Figure of IBD block sharing with various rates of Selfing!
    # test_IBD_binning()
    
