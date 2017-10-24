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
    _, ((ax1, ax2, ax3)) = plt.subplots(3, 1, sharex=True, figsize=(6, 6))
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
    
def plot_eight_scenarios(folder="./hetero_runs", scenario_nr=9, replicate_nr=20, sigmas=[], nr_inds=[], betas=[], print_res=False, title="Insert Title"):
    '''Function to plot the outcome of the eight scenarios.'''
    scenarios = range(scenario_nr)
    replicates = range(replicate_nr)
    data_set_nrs = range(scenario_nr * replicate_nr)
    
    # True Parameters:
    # sigmas = [[0.8, 0.4], [0.4, 0.6], [0.5, 0.5], [0.5, 0.5], [0.4, 0.6], [0.4, 0.6], [0.4, 0.6], [0.4, 0.6], [0.8, 0.8]]
    # nr_inds = [[600, 1000], [1000, 500], [40, 20], [1500, 1000], [40, 20], [1500, 1000], [20, 40], [1000, 1500], [100, 100]]
    # betas = [1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.5]
    
    # If Only one Parameter is given, expand it.
    if len(sigmas) == 1:
        sigmas = [sigmas[0] for _ in xrange(scenario_nr)]
        
    if len(nr_inds) == 1:
        nr_inds = [nr_inds[0] for _ in xrange(scenario_nr)]
    
    if len(betas) == 1:
        betas = [betas[0] for _ in xrange(scenario_nr)]
    
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
    if print_res == True:
        for i in xrange(len(params[0])):
            print("\nParameter %i: " % i)
            for j in xrange(len(good_inds)):
                print("Dataset %i:   %.4f (%.4f , %.4f): " % (data_set_nrs_found[j], params[j, i], cis[j, i, 0], cis[j, i, 1]))
    
    
    
    # Do the plot
    scale = 1
    base_font = 18
    x_min, x_max = scale*min(data_set_nrs), scale*max(data_set_nrs)
    x_vec_l = (scale*data_set_nrs_found - 0.1)
    x_vec_r = (x_vec_l + 0.2)
    
    # Prepare the color vector (alternating colors)
    cd_lefts = ["#ff421d", "#b8001e"]  # DarkRed
    cd_rights = ["#1700f5", "#8523ff"]
    cd_lefts = np.tile(np.repeat(cd_lefts, replicate_nr), scenario_nr)
    cd_rights = np.tile(np.repeat(cd_rights, replicate_nr), scenario_nr)
    # Extract the right Colors:
    cd_lefts = cd_lefts[good_inds]
    cd_rights = cd_rights[good_inds]
    
    cd_left = "red"
    cd_right = "blue"   
    # Make the x-Vector:
    x_vec = [[scale*i * replicate_nr, scale*(i + 1) * replicate_nr] for i in xrange(scenario_nr)]
    #print(x_vec)

    _, ((ax1, ax2, ax3)) = plt.subplots(3, 1, sharex=True, figsize=(8, 8))
    # Print the Data Points
    for i in xrange(len(data_set_nrs_found)):
        ax1.errorbar(x_vec_l[i], params[i, 0], yerr=params[i, 0] - cis[i, 0, 0], fmt="o",  color=cd_lefts[i], alpha=0.7, capsize=6)
        ax1.errorbar(x_vec_r[i], params[i, 1], yerr=params[i, 1] - cis[i, 1, 0], fmt="o", color=cd_rights[i], alpha=0.7, capsize=6)
        ax2.errorbar(x_vec_l[i], params[i, 2], yerr=params[i, 2] - cis[i, 2, 0], fmt="o", color=cd_lefts[i], alpha=0.7, capsize=6)
        ax2.errorbar(x_vec_r[i], params[i, 3], yerr=params[i, 3] - cis[i, 3, 0], fmt="o", color=cd_rights[i], alpha=0.7, capsize=6)
        ax3.errorbar(x_vec_l[i], params[i, 4], yerr=params[i, 4] - cis[i, 4, 0], fmt="o", color=cd_lefts[i], alpha=0.7, capsize=6)

    # ax3.errorbar(x_vec_r, params[:, 1], yerr=params[:, 1] - cis[:, 1, 0], fmt="o", label="Right", color=cd_right)
     
    # Print the Lines:
    for i, x in enumerate(x_vec):
        ax1.hlines(nr_inds[i][0], x[0], x[1], linewidth=3, color=cd_left)
        ax1.hlines(nr_inds[i][1], x[0], x[1], linewidth=3, color=cd_right)
        ax2.hlines(sigmas[i][0], x[0], x[1], linewidth=3, color=cd_left)
        ax2.hlines(sigmas[i][1], x[0], x[1], linewidth=3, color=cd_right)
        ax3.hlines(betas[i], x[0], x[1], linewidth=3, color=cd_left)  
    
    ax1.set_ylabel(r"$D_e$", fontsize=base_font+4, rotation=0, labelpad=15)
    ax2.set_ylabel(r"$\sigma$", fontsize=base_font+4, rotation=0, labelpad=15)
    ax3.set_ylabel(r"$\gamma$", fontsize=base_font+4, rotation=0, labelpad=15)
    ax1.set_ylim([5, 5000])
    ax1.set_yscale("log")
    ax2.set_ylim([0, 1.1])
    ax3.set_ylim([-0.3, 1.2])
    ax1.set_title(title, fontsize=base_font)
    
    ax1.set_xlim([x_min-9, x_max+9])
    ax2.set_xlim([x_min-9, x_max+9])
    ax3.set_xlim([x_min-9, x_max+9])
    
    # For Legend
    ax1.hlines(nr_inds[0][0], x_vec[0][0], x_vec[0][1], linewidth=3, color=cd_left, label="Parameter (left)")
    ax1.hlines(nr_inds[0][1], x_vec[0][0], x_vec[0][1], linewidth=3, color=cd_right, label="Parameter (right)")
    ax1.errorbar(x_vec_l[0], params[0, 0], yerr=params[0, 0] - cis[0, 0, 0], fmt="o",  color=cd_lefts[0], alpha=0.7, capsize=6, label="Estimator (left)")
    ax1.errorbar(x_vec_r[0], params[0, 1], yerr=params[0, 1] - cis[0, 1, 0], fmt="o", color=cd_rights[0], alpha=0.7, capsize=6, label="Estimator (right)")
    ax1.legend(loc="lower left", fontsize=base_font-10)
     
    plt.xlabel("Scenario", fontsize=base_font)
    plt.xticks(data_set_nrs[::10])
    
    plt.show()
    
    
    
# Do the plotting. Uncomment what you need!!

if __name__ == "__main__":
    # params, ci_s = load_estimates(data_set_nr=5, scenario=0)   
    # print(params)
    # print(ci_s)   
    # ps,cis=load_estimates_range(data_folder="./testfolder", scenario=0, data_set_vec=range(6))
    # print(ps)
    # print("Testin Successful!")
    
    # Plots different starting values:
    # plot_diff_start()     
    
    # Plots the 9 Scenarios for Different starting Values
    sigmas = [[0.8, 0.4], [0.4, 0.8], [0.5, 0.5], [0.5, 0.5], [0.4, 0.8], [0.4, 0.8], [0.4, 0.8], [0.4, 0.8], [0.8, 0.8]]
    nr_inds = [[500, 1000], [1000, 500], [40, 20], [2000, 1000], [40, 20], [1500, 1000], [20, 40], [100, 200], [100, 100]]
    betas = [1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.5, 0.5]
    #plot_eight_scenarios(folder="./hetero_runs1", sigmas=sigmas, nr_inds=nr_inds, betas=betas, title="Original Run")             # Plots eight Scenarios.
    plot_eight_scenarios(folder="./hetero_runs_isotropic", sigmas=sigmas, nr_inds=nr_inds, betas=betas, title="9 simulated scenarios")          # Plot the rerun eight Scenarios.
    #plot_eight_scenarios(folder="./hetero_runs_isotropicL100", sigmas=sigmas, nr_inds=nr_inds, betas=betas, title="Run V2. L=100")   # Plot the rerun eight Scenarios.
    
    # Plots the Scenarios for different discretizations:
    #nr_inds = [[100, 200], ]
    #sigmas = [[0.4, 0.8], ]
    #betas = [0.5, ]
    #plot_eight_scenarios(folder="./var_discrete", scenario_nr=5, sigmas=sigmas, nr_inds=nr_inds, betas=betas, title="Various Discretizations")  # Plots eight scenarios
    
    # Plots the reflected Scenarios
    #sigmas = [[0.8, 0.4], [0.4, 0.8], [0.8, 0.4], [0.4, 0.8], [0.4, 0.8], [0.8, 0.4]]
    #nr_inds = [[500, 1000], [1000, 500], [1000, 500], [500, 1000], [100, 200],[200, 100]]
    #betas = [1.0, 1.0, 1.0, 1.0, 0.5, 0.5]
    #plot_eight_scenarios(folder="./hetero_runs_symmetric", scenario_nr=6, sigmas=sigmas, nr_inds=nr_inds, betas=betas, title="Three mirrored scenarios")
    
    nr_inds = [[100, int(i * 100)] for i in [0.25, 0.5, 0.75, 1, 1.5, 2]]
    sigmas = [[0.4, 0.8] for _ in xrange(6)]
    betas = [0.5 for _ in xrange(6)]
    plot_eight_scenarios(folder="./hetero_runs_var_beta", scenario_nr=6, sigmas=sigmas, nr_inds=nr_inds, betas=betas, title="Increasing Beta")
    
    
