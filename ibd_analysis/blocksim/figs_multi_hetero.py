'''
Created on July 5th, 2017
Contains Methods to load and visualize the puptputs of multi_run_hetero
@author: Harald Ringbauer
'''


# from scipy.special import kv as kv  # Import Bessel functions of second kind

import numpy as np
import matplotlib.pyplot as plt
import cPickle as pickle  # @UnusedImport
import os

# Contains various Methods to analyze the Runs of Multirun Hetero!

def load_estimates(data_folder="./testfolder", data_set_nr=1, scenario=0):
    '''Load and return estimates.
    Return Empty Sets if data not found'''
    full_path = data_folder + "/scenario" + str(scenario) + "/data_set_nr_" + str(data_set_nr).zfill(2) + ".csv"
    # If data non-existent return nothing:
    if not os.path.exists(full_path):
        print("Not found: %s" % full_path)
        return ([], [])
    data = np.loadtxt(full_path, delimiter='$').astype('float64')
    params, ci_s = data[:, 0], data[:, 1:]
    return (params, ci_s)
    
def load_estimates_range(data_folder="./testfolder", data_set_vec=[1], scenario=0, fil=True):
    '''Load Estimates for a Range of Datasets.
    fil: Whether to delete values that are missing'''
    params = np.array([load_estimates(data_folder=data_folder, scenario=scenario, data_set_nr=i)[0] for i in data_set_vec])
    cis = np.array([load_estimates(data_folder=data_folder, scenario=scenario, data_set_nr=i)[1] for i in data_set_vec])
    
    if fil == True:
    # Maybe check for missig data HERE
        lengths = np.array(map(len, params))
        inds = ~(lengths == 0)
        
        params = params[inds]
        cis = cis[inds]
    
    return params, cis
        
        
def plot_diff_start(data_folder="./testfolder", scenarios=range(4), data_set_nr=1):
    '''Load and plot the Different Starting Value Estimates'''
    
    # Load the Data:
    params = np.array([load_estimates(data_folder=data_folder, scenario=i, data_set_nr=data_set_nr)[0] 
                     for i in scenarios])  # Dimension len(scenario), len(params)
    cis = np.array([load_estimates(data_folder=data_folder, scenario=i, data_set_nr=data_set_nr)[1] 
                  for i in scenarios])  # Dimension len(scenario), len(params), 2
    
    # First print the results:
    for i in range(np.shape(params)[1]):
        print("\nParameter %i: " % i)
        for s in scenarios:
            print("Scenario  %.4f (%.4f , %.4f): " % (params[s, i], cis[s, i, 0], cis[s, i, 1]))
            
    x_vec_l = np.array(scenarios) - 0.1
    x_vec_r = x_vec_l + 0.2
    x_min, x_max = min(x_vec_l), max(x_vec_r)
    #
    # plt.figure()
    f, ((ax1, ax2, ax3)) = plt.subplots(3, 1, sharex=True, figsize=(6, 6))
    c_left, c_right = "salmon", "aqua"
    cd_left, cd_right = "crimson", "navy"
    
    ax1.hlines(600, x_min, x_max, linewidth=3, color=c_left)
    ax1.hlines(1000, x_min, x_max, linewidth=3, color=c_right)
    ax1.errorbar(x_vec_l, params[:, 0], yerr=params[:, 0] - cis[:, 0, 0], fmt="o", label="Left", color=cd_left)
    ax1.errorbar(x_vec_r, params[:, 1], yerr=params[:, 1] - cis[:, 1, 0], fmt="o", label="Right", color=cd_right)
    ax1.set_ylabel(r"$D_e$", fontsize=18, rotation=0, labelpad=15)
    ax1.set_ylim([0, 1500])
    ax1.legend()
    # ax1.legend()
    
    ax2.hlines(0.8, x_min, x_max, linewidth=3, color=c_left)
    ax2.hlines(0.4, x_min, x_max, linewidth=3, color=c_right)
    ax2.errorbar(x_vec_l, params[:, 2], yerr=params[:, 2] - cis[:, 2, 0], fmt="o", label="Left", color=cd_left)
    ax2.errorbar(x_vec_r, params[:, 3], yerr=params[:, 3] - cis[:, 3, 0], fmt="o", label="Right", color=cd_right)
    ax2.set_ylabel(r"$\sigma$", fontsize=18, rotation=0, labelpad=15)
    ax2.set_ylim([0, 1.0])
    # ax2.legend()
    
    ax3.hlines(1.0, x_min, x_max, linewidth=3, color=c_left)
    # ax3.hlines(1000, x_min, x_max, linewidth=3, color=c_right)
    ax3.errorbar(x_vec_l, params[:, 4], yerr=params[:, 4] - cis[:, 4, 0], fmt="o", label="Left", color=cd_left)
    # ax3.errorbar(x_vec_r, params[:, 1], yerr=params[:, 1] - cis[:, 1, 0], fmt="o", label="Right", color=cd_right)
    ax3.set_ylabel(r"$\beta$", fontsize=18, rotation=0, labelpad=15)
    ax3.set_ylim([0, 1.1])
    # ax3.legend()
    

    plt.xlabel("Scenario", fontsize=18)
    plt.xticks(scenarios)
    plt.show()
    
def plot_eight_scenarios(folder="./hetero_runs", scenario_nr=9, replicate_nr=20):
    '''Function to plot the outcome of the eight scenarios.'''
    scenarios = range(scenario_nr)
    replicates = range(replicate_nr)
    data_set_nrs = range(scenario_nr * replicate_nr)
    
    # True Parameters:
    sigmas = [[0.8, 0.4], [0.4, 0.6], [0.5, 0.5], [0.5, 0.5], [0.4, 0.6], [0.4, 0.6], [0.4, 0.6], [0.4, 0.6], [0.8, 0.8]]
    nr_inds = [[600, 1000], [1000, 500], [40, 20], [1500, 1000], [40, 20], [1500, 1000], [20, 40], [1000, 1500], [100, 100]]
    betas = [1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.5]
    
    # Load the Data:
    params = [load_estimates(data_folder=folder, scenario=i, data_set_nr=j)[0] 
                     for i in scenarios for j in replicates]  # Dimension len(scenario), len(params)
    cis = [load_estimates(data_folder=folder, scenario=i, data_set_nr=j)[1] 
                  for i in scenarios for j in replicates]  # Dimension len(scenario), len(params), 2
    
    # Extract Indices where Dataset is actually there:
    lengths = np.array(map(len, params))
    good_inds = np.where(lengths > 0)[0]  # Check
    
    # Extract Params and Cis as Numpy Array:
    params = np.array([params[i] for i in good_inds]) 
    cis = np.array([cis[i] for i in good_inds])
    data_set_nrs_found = np.array(data_set_nrs)[good_inds]
    print(np.shape(cis))
    
    
    # First print the results:
#     for i in xrange(len(params[0])):
#         print("\nParameter %i: " % i)
#         for j in xrange(len(good_inds)):
#             print("Dataset %i:   %.4f (%.4f , %.4f): " % (data_set_nrs_found[j], params[j, i], cis[j, i, 0], cis[j, i, 1]))
    
    
    
    # Do the plot
    x_min, x_max = min(data_set_nrs), max(data_set_nrs)
    x_vec_l = data_set_nrs_found - 0.1
    x_vec_r = x_vec_l + 0.2
    
    # Prepare the color vector (alternating colors)
    cd_lefts = ["crimson", "DarkRed"]  # DarkRed
    cd_rights = ["aqua", "Aquamarine"]
    cd_lefts = np.tile(np.repeat(cd_lefts, replicate_nr), scenario_nr)
    cd_rights = np.tile(np.repeat(cd_rights, replicate_nr), scenario_nr)
    # Extract the right Colors:
    cd_lefts = cd_lefts[good_inds]
    cd_rights = cd_rights[good_inds]
    
    cd_left = "red"
    cd_right = "blue"   
    # Make the x-Vector:
    x_vec = [[i * replicate_nr, (i + 1) * replicate_nr] for i in xrange(scenario_nr)]
    #print(x_vec)
    
    
    f, ((ax1, ax2, ax3)) = plt.subplots(3, 1, sharex=True, figsize=(8, 8))
    # Print the Data Points
    for i in xrange(len(data_set_nrs_found)):
        ax1.errorbar(x_vec_l[i], params[i, 0], yerr=params[i, 0] - cis[i, 0, 0], fmt="o",  color=cd_lefts[i], alpha=0.7)
        ax1.errorbar(x_vec_r[i], params[i, 1], yerr=params[i, 1] - cis[i, 1, 0], fmt="o", color=cd_rights[i], alpha=0.7)
        ax2.errorbar(x_vec_l[i], params[i, 2], yerr=params[i, 2] - cis[i, 2, 0], fmt="o", color=cd_lefts[i], alpha=0.7)
        ax2.errorbar(x_vec_r[i], params[i, 3], yerr=params[i, 3] - cis[i, 3, 0], fmt="o", color=cd_rights[i], alpha=0.7)
        ax3.errorbar(x_vec_l[i], params[i, 4], yerr=params[i, 4] - cis[i, 4, 0], fmt="o", color=cd_lefts[i], alpha=0.7)
    # ax3.errorbar(x_vec_r, params[:, 1], yerr=params[:, 1] - cis[:, 1, 0], fmt="o", label="Right", color=cd_right)
     
    # Print the Lines:
    for i, x in enumerate(x_vec):
        ax1.hlines(nr_inds[i][0], x[0], x[1], linewidth=3, color=cd_left)
        ax1.hlines(nr_inds[i][1], x[0], x[1], linewidth=3, color=cd_right)
        ax2.hlines(sigmas[i][0], x[0], x[1], linewidth=3, color=cd_left)
        ax2.hlines(sigmas[i][1], x[0], x[1], linewidth=3, color=cd_right)
        ax3.hlines(betas[i], x[0], x[1], linewidth=3, color=cd_left)  
    
    ax1.set_ylabel(r"$D_e$", fontsize=18, rotation=0, labelpad=15)
    ax2.set_ylabel(r"$\sigma$", fontsize=18, rotation=0, labelpad=15)
    ax3.set_ylabel(r"$\beta$", fontsize=18, rotation=0, labelpad=15)
    ax1.set_ylim([5, 5000])
    ax1.set_yscale("log")
    ax2.set_ylim([0, 1.1])
    ax3.set_ylim([-0.3, 1.2])
    
    # For Legend
    ax1.hlines(nr_inds[0][0], x_vec[0][0], x_vec[0][1], linewidth=3, color=cd_left, label="Left")
    ax1.hlines(nr_inds[0][1], x_vec[0][0], x_vec[0][1], linewidth=3, color=cd_right, label="Right")
    ax1.legend(loc="lower left")
     
    plt.xlabel("Scenario", fontsize=18)
    plt.xticks(data_set_nrs[::10])
    plt.show()
    
    
# Some testing:

if __name__ == "__main__":
    # params, ci_s = load_estimates(data_set_nr=5, scenario=0)   
    # print(params)
    # print(ci_s)   
    # ps,cis=load_estimates_range(data_folder="./testfolder", scenario=0, data_set_vec=range(6))
    # print(ps)
    # print("Testin Successful!")
    
    
    # plot_diff_start()     # Plots different starting values
    plot_eight_scenarios()  # Plots eight scenarios
    # a=load_estimates(data_folder="./hetero_runs", scenario=1, data_set_nr=1)[1]
    # print(a)
    
