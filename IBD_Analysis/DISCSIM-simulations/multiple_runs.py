'''
Created on Mar 20, 2015
Container for multiple Runs.
@author: Harald Ringbauer
'''

from analysis import Analysis
from math import sqrt
import cPickle as pickle
import numpy as np
import matplotlib.pyplot as plt
import discsim
import ercs
from units import Unit_Transformer
from timeit import default_timer as timer
from IBD_detection import IBD_Detector
from random import shuffle
from scipy.special import kv as kv  # Import Bessel functions of second kind

nr_runs = 10  # How many runs
u_range = [1 / float(i) for i in range(2, 43, 5)]  # Range of u-values
sample_sizes = (100, 270, 440, 625)
   
# Some simulation constants:
grid_size = 90
sample_steps = 3  # Should be even! (4)
u = 1 / (8 * np.pi)
r = sqrt(2)  # sqrt(2)
sigma = sqrt(r ** 2 / 2.0)
recombination_rate = 0.001  # Recombination rate (NOT in CM!!)
num_loci = 1500  # Total number of loci
time = 1000  # Generation time for a single run
startlist = []
IBD_treshold = 40  # Nr of loci considered IBD


def single_run(run_i, u=u, nb=False):
    ''' Do a single run, parameters are saved in grid'''
    trans = Unit_Transformer(grid_size, u, r)
    sim = discsim.Simulator(grid_size)  # Create new Discsim-Simulator
    startlist = [(i, j) for i in range(0, grid_size, sample_steps) for j in range(0, grid_size, sample_steps)]
    # startlist = [j for i in zip(startlist, startlist) for j in i]  # Do for diploids
    sim.sample = [None] + startlist
    sim.event_classes = [ercs.DiscEventClass(r, u, rate=grid_size ** 2)]  # Fall with constant rate per unit area
    sim.recombination_probability = recombination_rate
    sim.num_loci = num_loci
    sim.max_population_size = 100000
    
    # Do the run.
    start = timer()
    for i in range(1, int(np.ceil(trans.to_model_time(time)))):  # Update to generation time!
        print("Simulation %.0f Doing step: %.0f u: %.3f" % (run_i, i, u))
        sim.run(until=(i))
    end = timer()
            
    print("\nRun time: %.2f s" % (end - start))
    print("Total Generations: %.2f" % time)
    print("Transformation factor: 1 Time unit is %.3f generations:" % trans.to_gen_time(1.0))
    
    # Extract pedigrees and do Block detection            
    pi, tau = sim.get_history()  # Extract the necessary data
    tau = trans.to_gen_time(np.array(tau))  # Vectorize and measure time in Gen time.
    chrom_l = num_loci * recombination_rate * 100
    det = IBD_Detector(tau, pi, recombination_rate, grid_size, startlist, IBD_treshold, time, chrom_l)  # Turn on a IBD_Detector

        
    ########################################################################################
    if nb == False:
        # Do classic IBD-Detection
        det.IBD_detection()
        block_nr = len(det.IBD_list)
        print("Number of IBD-blocks detected %.2f" % block_nr)
    
        data = Analysis(det)  # Do Data-Analysis and extract sigma!
        data.fit_expdecay(show=False)
        sigma0 = data.sigma_estimate
                      
        return(sigma0, block_nr)
    
    elif nb == True:
        # Mode where effective Detection Values are returned:
        # det.IBD_detection() 
        # block_nr0 = len(det.IBD_blocks)
        # print("Number of classic IBD-blocks detected %.2f" % block_nr0)
        # Do the MLE Inference
        # analysis = det.create_MLE_object()
        # analysis.create_mle_model("constant", chrom_l, [0.5, sigma])
        # analysis.mle_analysis_error()
        # D0, sigma0 = analysis.estimates[0], analysis.estimates[1]

        # Do effective Detecition
        det.IBD_detection_eff()
        block_nr1 = len(det.IBD_blocks)
        print("Number of effective IBD-blocks detected %.2f" % block_nr1)
        # Do the MLE inference
        analysis = det.create_MLE_object()
        analysis.create_mle_model("constant", chrom_l, [0.5, sigma])
        analysis.mle_analysis_error()
        D1, sigma1 = analysis.estimates[0], analysis.estimates[1]
              
        return(sigma1, D1, block_nr1)      

def analysis_run():
    # Do Number of runs for single parameter
    parameters = (sigma, grid_size, sample_steps, "DISCSIM")
    results = np.zeros((nr_runs, 2))  # Container for the data
    
    '''Runs the statistical mle_multi_run'''
    for i in range(0, nr_runs):
        print("Doing run: %.1f" % i)
        results[i, :] = single_run(i)  # Do the run and save the results 
    
    print("RUN COMPLETE!!")
    pickle.dump((results, parameters), open("disc_estats.p", "wb"))  # Pickle the data
    print("SAVED") 
    
 
def empirical_IBD_list(save_name):
    '''Generate empirical IBD-list. Nr. of run times'''
    parameters = (sigma, grid_size, sample_steps, "DISCSIM")
    results = []  # Container for the data
    startlist = [(i + sample_steps / 2.0, j + sample_steps / 2.0) for i in range(0, grid_size, sample_steps) for j in range(0, grid_size, sample_steps)]
    trans = Unit_Transformer(grid_size, u, r)
    
    '''Actual runs:'''
    for i in range(nr_runs):
        print("Doing run: %i" % i)
        sim = discsim.Simulator(grid_size)
    # startlist = [j for i in zip(startlist, startlist) for j in i]  # Do for diploids
        sim.sample = [None] + startlist
        sim.event_classes = [ercs.DiscEventClass(r, u, rate=grid_size ** 2)]  # Fall with constant rate per unit area
        sim.recombination_probability = recombination_rate
        sim.num_loci = num_loci
        sim.max_population_size = 100000

        # Do the actual run
        for j in range(1, int(np.ceil(trans.to_model_time(time)))):  # Update to generation time!
            print("Simulation %i Doing step: %i" % (i, j))
            sim.run(until=(j))
            
        pi, tau = sim.get_history()  # Extract the necessary data
        tau = trans.to_gen_time(np.array(tau))  # Vectorize and measure time in Gen time.
        
        chrom_l = num_loci * recombination_rate * 100
        det = IBD_Detector(tau, pi, recombination_rate, grid_size, startlist, IBD_treshold, time, chrom_l)  # Turn on a IBD_Detector
        det.IBD_detection() 
        pair_dist, pair_IBD, pair_nr = det.give_lin_IBD(bin_pairs=True)
         
        results.append([pair_dist, pair_IBD, pair_nr])
        
    pickle.dump((np.array(results), parameters), open(save_name, "wb"))  # Pickle the data
    print("SAVED")   


  
def run_var_sample(save_name):
    results = np.zeros((len(sample_sizes), nr_runs, 2))  # Container for the data
    position_list = [(i + sample_steps / 2, j + sample_steps / 2) for i in range(0, grid_size, sample_steps) for j in range(0, grid_size, sample_steps)]
    
    '''Actual runs:'''
    row = 0
    for k in sample_sizes: 
        for j in range(0, nr_runs):
            # Single run:
            print("Doing run: %.1f for %.0f samples" % (j + 1, k))
            trans = Unit_Transformer(grid_size, u, r)
            sim = discsim.Simulator(grid_size)  # Create new Discsim-Simulator
            
            shuffle(position_list)  # Randomize position List
            sim.sample = [None] + position_list[:k]  # Set k random chromosomes
            sim.event_classes = [ercs.DiscEventClass(r, u, rate=grid_size ** 2)]  # Fall with constant rate per unit area
            sim.recombination_probability = recombination_rate
            sim.num_loci = num_loci
            sim.max_population_size = 100000
            
            # Do the run.
            start = timer()
            for i in range(1, int(np.ceil(trans.to_model_time(time)))):  # Update to generation time!
                sim.run(until=(i))
            end = timer()
                    
            print("\nRun time: %.2f s" % (end - start))
            print("Total Generations: %.2f" % time)
            
            # Extract pedigrees and do Block detection        
            pi, tau = sim.get_history()
            tau = trans.to_gen_time(np.array(tau))  # Vectorize and measure time in Gen time.
            det = IBD_Detector(tau, pi, recombination_rate, grid_size, position_list[:k], IBD_treshold)  # Turn on a IBD_Detector
            det.IBD_detection()
            
            block_nr = len(det.IBD_list)
            print("Number of IBD-blocks detected %.2f" % block_nr)
            
            # Do Data mle_multi_run of Blocks and extract sigma
            data = Analysis(det)  # Do Data-Analysis and extract sigma!
            data.fit_expdecay(show=False)
            sigma = data.sigma_estimate
            print("Sigma Estimate: %.4f\n" % sigma)
            results[row, j, :] = (sigma, block_nr)
        row += 1  # Go one down in the results_row
            
        print("RUN COMPLETE!!")
    parameters = (sigma, grid_size, sample_sizes, "DISCSIM")
    pickle.dump((results, parameters), open(save_name, "wb"))  # Pickle the data
    pickle.dump((results, parameters), open("DataE/test1.p", "wb"))  # Pickle the data
    print("SAVED")     

def analysis_nb_run():
    # Do number of runs for varying neighborhood size
    results = np.zeros((len(u_range), nr_runs, 3))  # Container for results
    parameters = [sigma, grid_size, sample_steps, []]
    
    for idx, u in enumerate(u_range):
        parameters[3].append(u)  # Save u!
        # Do the runs
        for i in range(0, nr_runs):
            results[idx, i, :] = single_run(i, u, nb=True)
        pickle.dump((results, parameters), open("nb_stats_mle.p", "wb"))  # Pickle the data
        print("SAVED")
    print("RUN COMPLETE - YEEEEEAAAAH")


def analyze_stats():
    load_name = raw_input("What save to you want to load?\n") 
    (results, parameters) = pickle.load(open(load_name, "rb"))
    print(" Sigma %.2f \n Grid Size: %.2f \n Sample Steps: %.2f \n Dispersal mode: %s\n" % (parameters[0], parameters[1], parameters[2], parameters[3]))
    
    # results[:,0]*=0.001
    plt.figure()
    plt.hist(results[:, 0], bins=20, alpha=0.5)
    plt.axvline(2, color='r', linestyle='dashed', linewidth=3)
    plt.xlabel("Estimation")
    plt.ylabel("Number")
    plt.show()
    print(results[:, 0].mean())
    print(results[:, 0].std())

def analysis_nb_stats():
    (results, parameters) = pickle.load(open("nb_stats.p", "rb"))
    # print(results)
    print(" Sigma %.2f \n Grid Size: %.2f \n Sample Steps: %.2f \n" % (parameters[0], parameters[1], parameters[2]))
    u_range = [2 / i for i in parameters[3]]
    
    # Extract Dispersal Estimates and errors
    sigma_estimates1 = results[:, :, 2]
    sigma_estimates = results[:, :, 0]
    sigma1 = np.mean(sigma_estimates1, 1)
    sigma = np.mean(sigma_estimates, 1)
    errors1 = np.std(sigma_estimates1, 1)
    errors = np.std(sigma_estimates, 1)
    
    # Extract block results and errors
    blocks1 = np.mean(results[:, :, 3], 1)
    blocks = np.mean(results[:, :, 1], 1)
    berrors1 = np.std(results[:, :, 3], 1)
    berrors = np.std(results[:, :, 1], 1)
    
    # Plot the estimates
    f, axarr = plt.subplots(2, 1, sharex=True)  # @UnusedVariable
    plt.xlim([0, 115])
    plt.xlabel("Neighborhood Size", fontsize=18)
    
    axarr[0].errorbar(u_range, sigma1, yerr=errors1, fmt='yo', label="Effective recombination only", linewidth=2)
    axarr[0].errorbar(u_range, sigma, yerr=errors, fmt='bo', label="True blocks", linewidth=2)
    axarr[0].hlines(1, 4, 110, colors='red', linestyle='dashed', linewidth=2, label="True Dispersal Rate")
    axarr[0].legend()
    axarr[0].tick_params(axis='x', labelsize=14)
    axarr[0].tick_params(axis='y', labelsize=14)
    axarr[0].set_ylabel("Estimated Dispersal rate", fontsize=18)
    
    axarr[1].errorbar(u_range, blocks1, yerr=berrors1, fmt='yo', linewidth=2)
    axarr[1].errorbar(u_range, blocks, yerr=berrors, fmt='bo', linewidth=2)
    axarr[1].set_yscale("log")
    axarr[1].set_ylabel("Nr of IBD-blocks >l", fontsize=18)
    axarr[1].tick_params(axis='x', labelsize=14)
    axarr[1].tick_params(axis='y', labelsize=14)
    # plt.tight_layout()
    plt.show()
    
def bessel_l0(r, D, sigma, G=1.5, l0=0.04):
    '''Returns expected Block sharing per pair longer than l0'''
    exp_nr = 2 ** (-5 / 2.0) * G / (np.pi * D * sigma ** 2) * r / (sigma * np.sqrt(l0)) * kv(1, np.sqrt(2.0 * l0) * r / sigma)
    return(exp_nr)

def get_nr_shr_blocks(D, sigma, dist, pair_nr):
    '''Gives back the expected Number of shared blocks;
    given the start list. Assumes a constant model.'''
    exp_shr = 0
    for j in range(len(dist)):
        exp_shr += bessel_l0(dist[j], D, sigma) * pair_nr[j]  # G and l0 as predefined in bessel_l0
    return exp_shr        
        
def analysis_nb_stats1():
    '''Newer version of analysing various neighborhood sizes.
    Analyises only effective blocks. Prints Density and Dispersal
    estimates, and number of blocks found'''
    (results, parameters) = pickle.load(open("nb_stats_mle.p", "rb"))
    sigma_t, grid_size, sample_steps = parameters[0], parameters[1], parameters[2]
    # print(results)
    print(" Sigma %.2f \n Grid Size: %.2f \n Sample Steps: %.2f \n" % (parameters[0], parameters[1], parameters[2]))
    u_s = np.array(parameters[3])
    u_range = 2 / u_s
    # Extract Dispersal Estimates and errors
    sigma_estimates = results[:, :, 0]
    D_estimates = results[:, :, 1]
    
    sigma = np.mean(sigma_estimates, 1)
    errors = np.std(sigma_estimates, 1)
    
    D = np.mean(D_estimates, 1)
    D_errors = np.std(D_estimates, 1)
    
    # Extract block results and errors
    blocks = np.mean(results[:, :, 2], 1)
    berrors = np.std(results[:, :, 2], 1)
    
    # Get underlying block list
    start_list = [(i, j) for i in range(0, grid_size, sample_steps) for j in range(0, grid_size, sample_steps)]
    det = IBD_Detector([[], ], [[], ], 1, grid_size, start_list, 0, 0, 0)
    dist, _, pair_nr = det.give_lin_IBD(bin_pairs=True)  # Get the distance and pair-list
    
    exp_shr = []
    for u in u_s:  # Calculate the Densities; and exp. nr of shared blocks
        D_i = 1 / (2 * np.pi * u * sigma_t ** 2)
        exp_shr.append(get_nr_shr_blocks(D_i, sigma_t, dist, pair_nr))
           
    # Plot the estimates
    f, axarr = plt.subplots(3, 1, sharex=True)  # @UnusedVariable
    plt.xlim([0, 85])
    plt.xlabel("Neighborhood Size", fontsize=18)
    
    axarr[0].errorbar(u_range, sigma, yerr=errors, fmt='bo', label="Estimated", linewidth=3)
    axarr[0].hlines(sigma_t, 0, 85, colors='red', linestyle='dashed', linewidth=2, label="True")
    axarr[0].legend(loc="lower right")
    axarr[0].set_ylim([0, 1.5])
    axarr[0].tick_params(axis='x', labelsize=14)
    axarr[0].tick_params(axis='y', labelsize=14)
    axarr[0].set_ylabel(r"Dispersal $\sigma$", fontsize=18)
    
    axarr[1].errorbar(u_range, D, yerr=D_errors, fmt='bo', label="Estimated", linewidth=3)
    d_true = 1 / (2 * u_s * np.pi)
    axarr[1].plot(u_range, d_true, 'r-', linestyle='dashed', linewidth=2, label="Theory")
    # axarr[1].legend(loc='lower right')
    axarr[1].tick_params(axis='x', labelsize=14)
    axarr[1].tick_params(axis='y', labelsize=14)
    axarr[1].set_ylabel(r"Density $D$", fontsize=18)
    
    axarr[2].errorbar(u_range, blocks, yerr=berrors, fmt='bo', linewidth=3)
    axarr[2].plot(u_range, exp_shr, 'r-', linestyle='dashed', linewidth=2)
    axarr[2].set_yscale("log")
    axarr[2].set_ylabel(r"Nr IBD blocks", fontsize=18)
    axarr[2].tick_params(axis='x', labelsize=14)
    axarr[2].tick_params(axis='y', labelsize=14)
    plt.tight_layout()
    plt.show()
    
    
    
if __name__ == '__main__':
    inp = input("What do you want to do? \n (1) Run Analysis \n (2) Load Analysis\n (3) Run NB-Analysis\n"
                " (4) Analysis NB \n (5) Run Varying samples\n (6) Create Emp. IBD-List\n")
    if inp == 1:
        analysis_run()
    elif inp == 2:
        analyze_stats()
    elif inp == 3:
        analysis_nb_run()
    elif inp == 4:
        analysis_nb_stats1()
    elif inp == 5:
        save_name = raw_input("What do you want to save to?\n")
        run_var_sample(save_name)
    elif inp == 6:
        empirical_IBD_list("discsim20.p")
    elif inp == 7:
        print("Manually copy to multi-runs please")
        # analyze_emp_IBD_list("discsim_emplist.p")
    

