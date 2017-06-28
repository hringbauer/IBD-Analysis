'''
Created on Aug 18, 2015
Class for analyzing multiple selfing runs.
@author: Harald
'''

from selfing.grid_selfing import Grid
from mle_multi_run import Analysis
import cPickle as pickle
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from random import shuffle

t = 200  # Run Time of a single generation #t=400
nr_runs = 50  # 20
file_name = "mle_vs1.p"  # File name to read write to.
selfing_correct = 0.5
sample_sizes = (100, 270, 440, 625)
# sample_sizes=(100,)  # For test reasons

def single_run():
    ''' Do a single run, parameters are saved in grid'''
    # Do the run.
    start = timer()
    grid = Grid()
    grid.reset_grid()  # Reset grid just in case
    grid.set_samples()
    grid.update_t(t) 
    end = timer()        
    print("\nRun time: %.2f s" % (end - start))

    data = Analysis(grid)
    data.fit_expdecay(show=False)
    sigma_classic = data.sigma_estimate
    block_classic = len(data.IBD_blocks)
    block_all = len(grid.IBD_blocks1)
    
    data.IBD_blocks = grid.IBD_blocks1
    data.fit_expdecay(show=False)
    sigma_all = data.sigma_estimate
              
    return(sigma_classic, block_classic, sigma_all, block_all)        

def analysis_run():
    # Do Number of runs for single parameter
    grid = Grid()
    parameters = (grid.sigma, grid.gridsize, grid.sample_steps, grid.dispmode)
    results = np.zeros((nr_runs, 4))  # Container for the data
    
    '''Runs the statistical mle_multi_run'''
    for i in range(0, nr_runs):
        print("Doing run: %.1f" % i)
        results[i, :] = single_run()  # Do the run and save the results 
    
    print("RUN COMPLETE!!")
    pickle.dump((results, parameters), open(file_name, "wb"))  # Pickle the data
    print("SAVED")
    
def analyze_stats():
    (results, parameters) = pickle.load(open(file_name, "rb"))
    print(" Sigma %.2f \n Grid Size: %.2f \n Sample Steps: %.2f \n Disp Mode: %s \n " % (parameters[0], parameters[1], parameters[2], parameters[3]))
    
    # Extract Dispersal Estimates and errors
    sigma_estimates1 = results[:, 2]
    sigma_estimates = results[:, 0]
    sigma1 = np.mean(sigma_estimates1) * np.sqrt(1 - selfing_correct / (2.0 - selfing_correct))
    sigma = np.mean(sigma_estimates) * np.sqrt(1 - selfing_correct / (2.0 - selfing_correct))
    errors1 = np.std(sigma_estimates1)
    errors = np.std(sigma_estimates)
    
    # Extract block results and errors
    blocks1 = np.mean(results[:, 3])
    blocks = np.mean(results[:, 1])
    berrors1 = np.std(results[:, 3])
    berrors = np.std(results[:, 1])
    
    # Plot the estimates
    f, axarr = plt.subplots(2, 1, sharex=True)  # @UnusedVariable
    plt.xlabel("Neighborhood Size", fontsize=18)
    
    axarr[0].errorbar([0.9, 1.1], [sigma, sigma1], yerr=[errors, errors1], fmt='yo', label="0: Old 1: All", linewidth=2)
    axarr[0].hlines(parameters[0], 0.5, 1.5, colors='red', linestyle='dashed', linewidth=2, label="True Dispersal Rate")
    axarr[0].legend()
    axarr[0].tick_params(axis='x', labelsize=14)
    axarr[0].tick_params(axis='y', labelsize=14)
    axarr[0].set_ylabel("Estimated Dispersal rate", fontsize=18)
    
    axarr[1].errorbar([0.9, 1.1], [blocks, blocks1], yerr=[berrors, berrors1], fmt='yo', linewidth=2)
    # axarr[1].set_yscale("log")
    axarr[1].set_ylabel("Nr of IBD-blocks >l", fontsize=18)
    axarr[1].tick_params(axis='x', labelsize=14)
    axarr[1].tick_params(axis='y', labelsize=14)
    plt.tight_layout()
    plt.show()

def run_var_methods():
    '''Tests the difference between effective and true blocks for various methods'''
        # Do Number of runs for single parameter
    grid = Grid()
    dispersal = ["normal", "laplace", "uniform", "demes"]  # Vector for the various dispersal modes
    parameters = (grid.sigma, grid.gridsize, grid.sample_steps, dispersal)
    results = np.zeros((nr_runs, 4, 4))  # Container for the data
    
    j = 0
    for disp_mode in dispersal:
        grid.dispmode = disp_mode  # Set the dispersal mode
        '''Runs the statistical mle_multi_run'''
        for i in range(0, nr_runs):
            print("Doing run: %.1f \nDispersal mode: %s" % (i, disp_mode))
            results[i, :, j] = single_run()  # Do the run and save the results 
        j += 1
        
    print("RUN COMPLETE!!")
    pickle.dump((results, parameters), open(file_name, "wb"))  # Pickle the data
    print("SAVED")

def analyze_var_methods():
    '''Analyze the results fo run_var_methods'''
    (results, parameters) = pickle.load(open(file_name, "rb"))
    print(" Sigma %.2f \n Grid Size: %.2f \n Sample Steps: %.2f \n Disp Mode: %s \n " % (parameters[0], parameters[1], parameters[2], parameters[3]))
    
    # Extract Dispersal Estimates and errors
    sigma_estimates_true = np.mean(results[:, 0, :], axis=0)
    sigma_estimates_eff = np.mean(results[:, 2, :], axis=0)
    errors_true = np.std(results[:, 0, :], axis=0)
    errors_eff = np.std(results[:, 2, :], axis=0)
    
    # Extract block results and errors
    blocks_true = np.mean(results[:, 1, :], axis=0)
    blocks_eff = np.mean(results[:, 3, :], axis=0)
    berrors_true = np.std(results[:, 1, :], axis=0)
    berrors_eff = np.std(results[:, 3, :], axis=0)
    
    # Plot the estimates
    f, axarr = plt.subplots(2, 1, sharex=True)  # @UnusedVariable
    
    axarr[0].errorbar([0, 1, 2, 3], sigma_estimates_true, yerr=errors_true, fmt='go', label="True blocks", linewidth=2)
    axarr[0].errorbar([0.1, 1.1, 2.1, 3.1], sigma_estimates_eff, yerr=errors_eff, fmt='ko', label="Effective blocks", linewidth=2)
    axarr[0].hlines(parameters[0], -0.5, 5, colors='red', linestyle='dashed', linewidth=2, label="True Dispersal Rate")
    axarr[0].set_xlim(-0.5, 5)
    axarr[0].legend()
    axarr[0].tick_params(axis='x', labelsize=14)
    axarr[0].tick_params(axis='y', labelsize=14)
    axarr[0].set_ylabel("Estimated Dispersal rate", fontsize=18)
    
    axarr[1].errorbar([0, 1, 2, 3], blocks_true, yerr=berrors_true, fmt='go', linewidth=2)
    axarr[1].errorbar([0.1, 1.1, 2.1, 3.1], blocks_eff, yerr=berrors_eff, fmt='ko', linewidth=2)
    # axarr[1].set_yscale("log")
    axarr[1].set_ylabel("Nr of IBD-blocks >l", fontsize=18)
    axarr[1].tick_params(axis='x', labelsize=4)
    axarr[1].tick_params(axis='y', labelsize=14)
    plt.xlabel("Various models", fontsize=18)
    # plt.tight_layout()
    plt.show()
 
def mle_multiple_runs():
    '''Run multiple times and analyze both MLE and Bessel-Fit sigma'''
    grid = Grid()
    parameters = (grid.sigma, grid.gridsize, grid.sample_steps, grid.dispmode)
    results = np.zeros((nr_runs, 2))  # Container for the data
    
    '''Runs the statistical mle_multi_run'''
    for i in range(0, nr_runs):
        print("Doing run: %.1f" % i)
        grid = Grid()
        grid.reset_grid()  # Reset grid just in case
        grid.set_samples()
        grid.update_t(t)  # Do the actual run
        # grid.update_IBD_blocks_demes(5)
        # grid.analyze_IBD_mat()      # Analyze the full matrix and extract blocks into IBD1
        # grid.IBD_blocks=grid.IBD_blocks1    # Modify IBD-List
        grid.conv_IBD_list_to_pair_IBD()  # Put IBD-List into pair-IBD-list
        
        data = Analysis(grid)
        data.fit_expdecay(show=False)  # Do the classic fit
        sigma_classic = data.sigma_estimate
        
        data.mle_estimate(grid.pair_IBD / grid.rec_rate, grid.pair_dist)
        sigma_mle = data.sigma_estimate 
        
        results[i, :] = [sigma_classic, sigma_mle]  # Save the results to vector 
        pickle.dump((results, parameters), open(file_name, "wb"))  # Pickle the data
    print("Finished") 

def run_var_samp():
    '''Runs simulations for various sample sizes and saves estimates and parameters.'''
    grid = Grid()
    results = np.zeros((len(sample_sizes), nr_runs, 3))  # Container for the data
    parameters = (grid.sigma, grid.gridsize, sample_sizes, grid.dispmode)
    sample_steps = grid.sample_steps 
    
    '''Actual runs:'''
    row = 0
    for k in sample_sizes:
        position_list = [(i + sample_steps / 2, j + sample_steps / 2, 0) for i in range(0, grid.gridsize, sample_steps) for j in range(0, grid.gridsize, sample_steps)]
        # position_list = [(i + sample_steps / 2, j  + sample_steps / 2, 0) for i in range(0, sample_steps * k, sample_steps) for j in range(0, sample_steps * k, sample_steps)]


        for i in range(0, nr_runs):
            print("Doing run: %.1f for %.0f samples" % (i, k))
            grid = Grid()
            grid.reset_grid()  # Delete everything
            shuffle(position_list)  # Randomize position List
            grid.set_samples(position_list[:k])  # Set the samples
            grid.update_t(t)  # Do the actual run
            if grid.dispmode == "demes":
                grid.update_IBD_blocks_demes(5)  # Update position for deme mle_multi_run!!
            grid.conv_IBD_list_to_pair_IBD()  # Put IBD-List into pair-IBD-list
            data = Analysis(grid)  # Do Data-Analysis
            data.fit_expdecay(show=False)  # Do the classic fit
            sigma_classic = data.sigma_estimate
        
            var_sigma = data.mle_estimate(grid.pair_IBD, grid.pair_dist)  # Variance of sigma as estimated from the Fisher-Information
            sigma_mle = data.sigma_estimate 
            results[row, i, :] = (sigma_classic, sigma_mle, var_sigma)
            pickle.dump((results, parameters), open(file_name, "wb"))  # Pickle the data
        row += 1  # Go one down in the results_row
            
        print("RUN COMPLETE!!")
    pickle.dump((results, parameters), open(file_name, "wb"))  # Pickle the data
    print("SAVED") 

            
def analyze_multiple_selfing():
    '''Analyze the results of the'''
    (results, parameters) = pickle.load(open(file_name, "rb"))
    # sigmas_classic = results[:, 0]
    # sigmas_mles = results[:, 1]
    
    result=3    # Position of result to analyze
    sigmas_classic = results[result, :, 0]
    sigmas_mles = results[result, :, 1]
    var_sigmas = -results[result, :, 2]
    print(np.sqrt(var_sigmas))
    print(parameters)
    print("Mean Regression mle_multi_run: %.4f" % np.mean(sigmas_classic))
    print("Mean MLE estimates: %.4f" % np.mean(sigmas_mles))
    print("Standard Deviations Regression: %.4f" % np.std(sigmas_classic))
    print("Standard Deviations MLE: %.4f" % np.std(sigmas_mles))
    print("Estimated STD of MLE estimates: %.4f" % np.sqrt(np.mean(var_sigmas)))
    
    plt.figure()
    x_dist = np.linspace(0, 1, num=len(sigmas_classic))
    plt.scatter(0.5 + x_dist, results[0, :, 0], c='b', label="Regression")
    plt.scatter(2 + x_dist, results[0, :, 1], c='r', label="MLE")
    plt.scatter(4 + x_dist, results[1, :, 0], c='b')
    plt.scatter(5.5 + x_dist, results[1, :, 1], c='r')
    plt.scatter(7.5 + x_dist, results[2, :, 0], c='b')
    plt.scatter(9 + x_dist, results[2, :, 1], c='r')
    plt.scatter(11 + x_dist, results[3, :, 0], c='b')
    plt.scatter(12.5 + x_dist, results[3, :, 1], c='r')
    plt.xlabel("Sample Size", fontsize=20)
    plt.ylabel("Estimated " + r"$\mathbf{\sigma}$", fontsize=20)
    plt.xticks([1.5, 5, 8.5, 12], ["100", "270", "440", "625"], fontsize=20)
    plt.hlines(2, -0.5, 14, label="True " + r"$\mathbf{\sigma}$", color='k', linewidth=2)
    plt.legend()
    plt.ylim([0, 3])
    plt.show()
    
    plt.plot(sigmas_classic, sigmas_mles, 'ro')
    plt.xlabel("Regression Estimate")
    plt.ylabel("MLE Estimate")
    plt.show()
    
    print("Correlation: %.4f" % np.corrcoef(sigmas_classic, sigmas_mles)[0, 1])
    
if __name__ == '__main__':
    inp = input("What do you want to do? \n (1) Run Analysis\n (2) Load Analysis\n (3) Run various sample sizes\n (4) Analyze various sample sizes\n (5) Run multiple MLEs \n (6) Analyze multiple MLEs\n")
    if inp == 1:
        analysis_run()
    elif inp == 2:
        analyze_stats()
    elif inp == 3:
        run_var_methods()
    elif inp == 4:
        analyze_var_methods()
    elif inp == 5:
        # mle_multiple_runs()
        run_var_samp()
    elif inp == 6:
        analyze_multiple_selfing()
