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
    
    print(type(cis[0, 2, 1]))
    # First print the results:
    
    for i in range(np.shape(params)[1]):
        print("\nParameter %i: " % i)
        for s in scenarios:
            print("Scenario  %.4f (%.4f , %.4f): " % (params[s, i], cis[s, i, 0], cis[s, i, 1]))
            
    x_vec_l = np.array(scenarios) - 0.1
    x_vec_r = x_vec_l + 0.2
    x_min, x_max = min(x_vec_l), max(x_vec_r)
    #
    #plt.figure()
    f, ((ax1, ax2, ax3)) = plt.subplots(3, 1, sharex=True, figsize=(6,6))
    c_left, c_right = "salmon", "aqua"
    cd_left, cd_right = "crimson", "navy"
    
    ax1.hlines(600, x_min, x_max, linewidth=3, color=c_left)
    ax1.hlines(1000, x_min, x_max, linewidth=3, color=c_right)
    ax1.errorbar(x_vec_l, params[:, 0], yerr=params[:, 0] - cis[:, 0, 0], fmt="o", label="Left", color=cd_left)
    ax1.errorbar(x_vec_r, params[:, 1], yerr=params[:, 1] - cis[:, 1, 0], fmt="o", label="Right", color=cd_right)
    ax1.set_ylabel(r"$D_e$", fontsize=18, rotation=0, labelpad=15)
    ax1.set_ylim([0,1500])
    ax1.legend()
    # ax1.legend()
    
    ax2.hlines(0.8, x_min, x_max, linewidth=3, color=c_left)
    ax2.hlines(0.4, x_min, x_max, linewidth=3, color=c_right)
    ax2.errorbar(x_vec_l, params[:, 2], yerr=params[:, 2] - cis[:, 2, 0], fmt="o", label="Left", color=cd_left)
    ax2.errorbar(x_vec_r, params[:, 3], yerr=params[:, 3] - cis[:, 3, 0], fmt="o", label="Right", color=cd_right)
    ax2.set_ylabel(r"$\sigma$", fontsize=18, rotation=0, labelpad=15)
    ax2.set_ylim([0,1.0])
    #ax2.legend()
    
    ax3.hlines(1.0, x_min, x_max, linewidth=3, color=c_left)
    #ax3.hlines(1000, x_min, x_max, linewidth=3, color=c_right)
    ax3.errorbar(x_vec_l, params[:, 4], yerr=params[:, 4] - cis[:, 4, 0], fmt="o", label="Left", color=cd_left)
    #ax3.errorbar(x_vec_r, params[:, 1], yerr=params[:, 1] - cis[:, 1, 0], fmt="o", label="Right", color=cd_right)
    ax3.set_ylabel(r"$\beta$", fontsize=18, rotation=0, labelpad=15)
    ax3.set_ylim([0,1.1])
    #ax3.legend()
    

    plt.xlabel("Scenario", fontsize=18)
    plt.xticks(scenarios)
    plt.show()
    
# Some testing:

if __name__ == "__main__":
    # params, ci_s = load_estimates(data_set_nr=5, scenario=0)   
    # print(params)
    # print(ci_s)   
   # ps,cis=load_estimates_range(data_folder="./testfolder", scenario=0, data_set_vec=range(6))
    # print(ps)
    # print("Testin Successful!")
    
    
    plot_diff_start()
