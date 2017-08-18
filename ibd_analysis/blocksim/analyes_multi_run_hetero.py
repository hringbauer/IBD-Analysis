'''
Created on July 5th, 2017
Contains Methods to visualize the Oupturts of multi_run_hetero
@author: Harald Ringbauer
'''

import sys  # @UnusedImport
from grid import factory_Grid
# from analysis import Analysis, torus_distance
# from random import shuffle
# from itertools import combinations
# from bisect import bisect_right
# from math import pi
# from scipy.special import kv as kv  # Import Bessel functions of second kind

# import cPickle as pickle
import numpy as np
import cPickle as pickle
# import matplotlib.pyplot as plt
import os

# Contains various Methods to analyze the Runs of Multirun Hetero!

def load_estimates(data_set_nr, scenario):


    def save_estimates(self, params, ci_s, data_set_nr, scenario=0):
        '''Saves Estimates. Params: Parameters of best estimates.
        ci_s: Confidence Intervalls.'''
        full_path = self.data_folder + "/scenario" + str(scenario) + "/data_set_nr_" + str(data_set_nr).zfill(2) + ".csv"
        # Check whether Directory exists and creates it if necessary
        
        print("Saving...")
        directory = os.path.dirname(full_path)  # Extract Directory
        print("Directory: " + directory)
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        data = np.column_stack((params, ci_s))  # Stacks Estimates and Parameters together
        print(data)
        np.savetxt(full_path, data, delimiter="$")  # Save the coordinates

        
    def single_run(self, data_set_nr, scenario=0):
        '''Does a single run. 
        Scenario gives the number of the scenario which is run.
        data_set_nr give the data_set number which has to be run'''
        
        print("Doing run %i for Scenario %i" % (data_set_nr, scenario))
        
        # Makes the Grid and sets all Parameters
        grid = factory_Grid(model="hetero")  # Creates Grid with Default Parameters
        grid = self.set_grid_parameters(grid, scenario=scenario)  # Resets this Default Parameters
        grid.reset_grid()  # Delete everything and re-initializes Grid (but stores Parameters)
        print("Sigmas: ")
        print(grid.sigmas)
        
        print("Population Densities: ")
        print(grid.start_inds)
        
        print("Beta:")
        print(grid.beta)
        
        # Set the Samples 
        # print(self.position_list[:20])
        grid.set_samples(self.position_list)  # Set the samples
        print("Nr. of Samples successfully set: %i" % len(self.position_list))
        # Runs the Simulations for time t
        grid.update_t(self.max_t)  # Do the actual run
        
        # Do the maximum Likelihood estimation
        # mle_ana = grid.create_MLE_object(bin_pairs=True, plot=True)
        # mle_ana.create_mle_model("constant", grid.chrom_l, [1.0, 1.0], diploid=False)  # Runs the analysis.
        # mle_ana.create_mle_model("", grid.chrom_l, [1.0,1.0,0.5], diploid=False) 
        # mle_ana.create_mle_model("hetero", grid.chrom_l, start_param=[50, 0.2], diploid=False)
        # For Debugging: Test different Likelihoods (to see whether degenerate!)
        mle_ana = grid.create_MLE_object(reduce_start_list=True, plot=self.plot_positions)  # Create the MLE-object
        
        # print("Try out different Parameters for Likelihood")
        # mle_ana.mle_object.loglikeobs(np.array([30, 31, 0.2, 0.6]))
        # mle_ana.mle_object.loglikeobs(np.array([30, 30, 0.21, 0.6]))
        # mle_ana.mle_object.loglikeobs(np.array([30, 30, 0.2, 0.61]))
        # print("Coordinates: ")
        # print(mle_ana.mle_object.coords_bary)
        # a = int(input("Enter 0"))   # Wait for Input
        mle_ana.create_mle_model("hetero", grid.chrom_l, start_param=np.array([800.0, 800.0, 0.6, 0.6, 1.0]), diploid=False,
                                 barrier_pos=self.barrier_pos, barrier_angle=self.barrier_angle)
        # mle_ana.mle_object.loglikeobs(np.array([30.0, 30.0, 0.4, 0.4, 0.]))
        
        # mle_ana.mle_object.loglikeobs(np.array([30.0, 30.0, 0.4, 0.4, 0.0]))
        
        # mle_ana.create_mle_model("constant", grid.chrom_l, [30.0, 0.4], diploid=False)  # Runs the analysis.
        # mle_ana.mle_object.loglikeobs(np.array([30.0, 0.4]))
        
        # mle_ana.mle_object.loglikeobs(np.array([50.0, 50.0, 0.4, 0.4]))
        
        # mle_ana.create_mle_model("constant", grid.chrom_l, [50.0, 0.4], diploid=False)  # Runs the analysis.
        # mle_ana.mle_object.loglikeobs(np.array([50.0, 0.4]))
        
        # mle_ana.create_mle_model("power_growth", grid.chrom_l, [500.0, 0.4, 0.5], diploid=False)
        
        
        # Save or Load IBD-blocks (if needed):
        ibd_path="./ibd_blocks.p"
        pickle.dump(grid.IBD_blocks, open(ibd_path, "wb"), protocol=2)  # Pickle the data
        print("Pickling of %i blocks complete" % len(grid.IBD_blocks))
        #grid.IBD_blocks = pickle.load(open(ibd_path, "rb"))  # Load data from pickle file!
        
        
        mle_ana.mle_analysis_error()  # Analyses the samples
        
        ci_s = mle_ana.ci_s
        estimates = mle_ana.estimates
        
        
        
        self.save_estimates(estimates, ci_s, data_set_nr, scenario)
        print("Results SAVED!") 
        
        # Save addtional Infos about the run
        if data_set_nr == 0:
            print("To Implement")
        print("Run Complete UUUUHHH YYEEEEAHH!")
        





