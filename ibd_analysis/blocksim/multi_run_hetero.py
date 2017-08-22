'''
Created on July 5th, 2017
Contains a class that can do multiple Runs; and save Results to according folders.
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


class MultiRunHetero(object):
    '''
    Class that can produce as well as calling analysis for datasets.
    Can be given parameters; and has according folder where data is saved.
    Can Simulate the 8 scenarios of Raphael - saves result in folders Hetero1 - Hetero8;
    and then analyzes them.
    '''
    data_folder = ""  # Folder where to save to
    nr_data_sets = 0  # Number of the datasets
    # multi_processing = 0  # Whether to actually use multi-processing
    scenario = 0  # 1-8 are Raphaels scenarions
    chrom_l = 5000  # Length of the chromosome (in cM!)

    plot_positions = False
    # All Parameters for the grid
    gridsize = 200  # 60  # 180/2  # 160 # 180 # 98
    rec_rate = 100.0  # Everything is measured in CentiMorgan; Float!
    dispmode = "mig_mat"  # raphael/mig_mat possible
    sigma = 1.0       
    IBD_treshold = 4.0  # Threshold over which IBD stored.
    delete = True  # If TRUE: blocks below threshold are deleted and not traced back any more..
    drawlist_length = 10000  # Variable for how many random Variables are drawn simultaneously.
    max_t = 500  # Runs the Simulations for time t.
    
    # Barrier Parameters:
    barrier_pos = [100, 0]  # The Position of the Barrier
    barrier_angle = 0  # Angle of Barrier (in Radiant)

    # The Parameters of the 8 Scenarios.
    sigmas = [[0.8, 0.4], [0.4, 0.6], [0.5, 0.5], [0.5, 0.5], [0.4, 0.6], [0.4, 0.6], [0.4, 0.6], [0.4, 0.6], [0.8, 0.8]]
    assert(len(sigmas) == 9)
    nr_inds = [[600, 1000], [1000, 500], [40, 20], [1500, 1000], [40, 20], [1500, 1000], [20, 40], [1000, 1500], [100, 100]]
    assert(len(nr_inds) == 9)
    betas = [1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.5]
    assert(len(betas) == 9)
    
    # start_params=[[800.0, 800.0, 0.6, 0.6, 1.0],[800.0, 800.0, 0.5, 1.2, 0.8],[800.0, 1600.0, 1.0, 1.0, 1.0], [20.0, 20.0, 0.5, 0.5, 0.3]]
    # start_params=map(np.array,start_params)
    
    # start_param = np.array([80, 80, 0.5, 0.5 , 0.5])  # Original Start-Params used for Inference in 8 Scenarios 
    start_param = np.array([110, 90, 0.7, 0.9, 0.5])
    
    # Position_List:
    # position_list = [(85 + i * 2, 85 + j * 2, 0) for i  # For test of small grid
    #             in range(15) for j in range(15)]
    
    # New Position List
    # position_list = [[85, 95], [95, 95], [105, 95], [115, 95], [85, 105], [95, 105], [105, 105], [115, 105],
    #                 [85, 115], [95, 115], [105, 115], [115, 115]]
    
    # Narrower Position List to check for power
    # position_list = [[91, 95], [97, 95], [103, 95], [109, 95], [91, 101], [97, 101], [103, 101], [109, 101],
    #                                  [91, 107], [97, 107], [103, 107], [109, 107]]
    position_list = [[100 + i, 100 + j] for i in xrange(-10, 11, 4) for j in xrange(-6, 7, 4)]
    pop_size = 20  # Nr of individuals per position.
    
    
    
    def __init__(self, data_folder, nr_data_sets, position_list=[], multi_processing=0):
        '''
        Constructor. Sets the Grid class to produce and the MLE class to analyze the Data
        '''
        self.data_folder = data_folder
        self.nr_data_sets = nr_data_sets
        self.multi_processing = multi_processing 
        self.set_position_list()  # Creates the Position List
        self.gridsize = self.gridsize + self.gridsize % 2  # Make Sure Grid Size is even.
        
        # Overwrites Position List; in case None is given.
        if len(position_list) > 0:
            self.position_list = position_list
    
    def set_position_list(self):
        '''Sets the position Llist'''
        self.position_list = [(x[0], x[1], 2 * i) for i in xrange(self.pop_size) for x in self.position_list]  # Create The Starting Sample
        return 0
        
        
    def create_data_set(self, data_set_nr, position_path=None, genotype_path=None):
        '''Method to create data_set nr data_set_nr.'''
        # If method is called without path:
        if (position_path == None) or (genotype_path == None):
            raise NotImplementedError("Implement creation of Data-Set!")
        
        else:    
            position_list = np.loadtxt(position_path, delimiter='$').astype('float64')
            genotype_matrix = np.loadtxt(genotype_path, delimiter='$').astype('float64')
            self.save_data_set(position_list, genotype_matrix, data_set_nr) 
        print("Dataset successfully created!")
    
    def set_grid_parameters(self, grid, scenario=0):
        '''Sets all the Parameters of a given Grid object.'''
        grid.chrom_l = self.chrom_l
        grid.gridsize = self.gridsize  # 60  # 180/2  # 160 # 180 # 98
        grid.rec_rate = self.rec_rate  # Everything is measured in CentiMorgan; Float!
        grid.dispmode = self.dispmode  # normal/uniform/laplace/laplace_refl/demes/raphael       
        grid.IBD_treshold = self.IBD_treshold  # Threshold over which IBD is detected.
        grid.delete = self.delete  # If TRUE: blocks below threshold are deleted.
        grid.drawlist_length = self.drawlist_length  # Variable for how many random Variables are drawn simultaneously.
        grid.barrier_pos = self.barrier_pos[0]  # Sets the position of the vertical Barrier.
        
        # The Parameters of the 8 Scenarios.
        grid.sigmas = np.array(self.sigmas[scenario])
        grid.start_inds = np.array(self.nr_inds[scenario])  # Set the Nr of Individuals.
        grid.beta = self.betas[scenario]
        return grid
        
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

    def test_drawer(self, grid, pos, reps=10000):
        '''Test the drawer of the grid. Draw rep. many replicates
        and print Mean and Std of parental position.'''
        parents = [grid.get_parents_pos(pos[0],pos[1]) for _ in range(10000)]
        print(parents[:5])
        x_off_sets = [(parent[0][0]) for parent in parents]
        y_off_sets = [(parent[0][1]) for parent in parents]
        # print(np.corrcoef([x_off_sets, y_off_sets]))
        print("Parental Position:")
        print(pos)
        print(len(x_off_sets))
        print("Mean x-Axis: %.4f" % np.mean(x_off_sets))
        print("Mean y-Axis: %.4f" % np.mean(y_off_sets))
        print("STD x-Axis: %.4f" % np.std(x_off_sets))
        print("STD y-Axis: %.4f" % np.std(y_off_sets))
        print("Test finished.")
        
    def single_run(self, data_set_nr, scenario=0, load_blocks=False, save_blocks=False):
        '''Does a single run. 
        Scenario gives the number of the scenario which is run.
        data_set_nr give the data_set number which has to be run'''
        
        print("Doing run %i for Scenario %i" % (data_set_nr, scenario))
        ibd_path = "./ibd_blocks.p"
        
        # Makes the Grid and sets all Parameters
        grid = factory_Grid(model="hetero")  # Creates Grid with Default Parameters
        grid = self.set_grid_parameters(grid, scenario=scenario)  # Resets this Default Parameters
        grid.reset_grid()  # Delete everything and re-initializes Grid (but stores Parameters)
        
        # Print some output:
        print("Sigmas: ")
        print(grid.sigmas)
        print("Population Densities: ")
        print(grid.start_inds)
        print("Beta:")
        print(grid.beta)
        print("Barrier Position: %i" % grid.barrier_pos)
        
        # For Debugging: Test the parent draw of the grid:
        # self.test_drawer(grid, [30,10], reps=10000)
        # self.test_drawer(grid, [120,10], reps=10000)
        
        # Set the Samples 
        # print(self.position_list[:20])
        grid.set_samples(self.position_list)  # Set the samples
        print("Nr. of Samples successfully set: %i" % len(self.position_list))
        
        # Runs the Simulations for time t or loads the blocks
        if load_blocks == False:
            grid.update_t(self.max_t)  # Do the actual run
        
        elif load_blocks == True:
            grid.IBD_blocks = pickle.load(open(ibd_path, "rb"))  # Load data from pickle file!
            print("Loading of %i blocks complete" % len(grid.IBD_blocks))
            
        # Save or Load IBD-blocks (if needed):
        if save_blocks == True:
            pickle.dump(grid.IBD_blocks, open(ibd_path, "wb"), protocol=2)  # Pickle the data
            print("Pickling of %i blocks complete" % len(grid.IBD_blocks))
            
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
        
            
        # Set the Start Parameters
        # start_param = self.start_params[scenario]
        mle_ana.create_mle_model("hetero", grid.chrom_l, start_param=self.start_param, diploid=False,
                                 barrier_pos=self.barrier_pos, barrier_angle=self.barrier_angle)
        
        # mle_ana.mle_object.loglikeobs(np.array([30.0, 30.0, 0.4, 0.4, 0.]))
        
        # mle_ana.mle_object.loglikeobs(np.array([30.0, 30.0, 0.4, 0.4, 0.0]))
        
        # mle_ana.create_mle_model("constant", grid.chrom_l, [30.0, 0.4], diploid=False)  # Runs the analysis.
        # mle_ana.mle_object.loglikeobs(np.array([30.0, 0.4]))
        
        # mle_ana.mle_object.loglikeobs(np.array([50.0, 50.0, 0.4, 0.4]))
        
        # mle_ana.create_mle_model("constant", grid.chrom_l, [50.0, 0.4], diploid=False)  # Runs the analysis.
        # mle_ana.mle_object.loglikeobs(np.array([50.0, 0.4]))
        
        # mle_ana.create_mle_model("power_growth", grid.chrom_l, [500.0, 0.4, 0.5], diploid=False)
        
        mle_ana.mle_analysis_error()  # Analyses the samples
        
        ci_s = mle_ana.ci_s
        estimates = mle_ana.estimates
        
        self.save_estimates(estimates, ci_s, data_set_nr, scenario)
        print("Results SAVED!") 
        
        # Save addtional Infos about the run
        if data_set_nr == 0:
            print("To Implement")
        print("Run Complete UUUUHHH YYEEEEAHH!")
        

#########################################################################################

# ## Here are Methods that can create and analyze Data Sets:
def cluster_run(data_set_nr, scenarios=8, replicates=20):
    '''Script to run stuff on the cluster.'''
    eff_run_nr = data_set_nr % replicates  
    eff_scenario = data_set_nr / replicates
    
    assert(eff_scenario * replicates + eff_run_nr == data_set_nr)  # Sanity Check.
    multirun = MultiRunHetero("./hetero_runs", 10)
    
    multirun.single_run(eff_run_nr, eff_scenario)  # Does the actual Run.


# Some testing:

if __name__ == "__main__":
    # scenario = int(sys.argv[1])  # Which data-set to use
    data_set_nr = 130    
    # data_set_nr = int(sys.argv[1]) - 1  # Substract 1 as on cluster on starts with 1
    # scenario = 3
    # scenario = scenario - 1 
    multirun = MultiRunHetero("./scenarios", 180)
    # multirun.single_run(data_set_nr, scenario, load_blocks=True, save_blocks=False)
    
    # data_set_nr = int(sys.argv[1])  # Which data-set to use
    # data_set_nr = 2
    cluster_run(data_set_nr)
    
    
    
'''

multirun = MultiRunHetero("./testfolder", 10)
multirun.single_run(1, 0)
'''




