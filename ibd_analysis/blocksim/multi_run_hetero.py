'''
Created on July 5th, 2017
Contains a class that can do multiple Runs; and save Results to according folders.
@author: Harald Ringbauer
'''

from grid import factory_Grid
# from analysis import Analysis, torus_distance
# from random import shuffle
# from itertools import combinations
# from bisect import bisect_right
# from math import pi
# from scipy.special import kv as kv  # Import Bessel functions of second kind

# import cPickle as pickle
import numpy as np
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
    #multi_processing = 0  # Whether to actually use multi-processing
    scenario = 0  # 1-8 are Raphaels scenarions
    chrom_l = 500  # Length of the chromosome (in cM!)

    # All Parameters for the grid
    gridsize = 200  # 60  # 180/2  # 160 # 180 # 98
    rec_rate = 100.0  # Everything is measured in CentiMorgan; Float!
    dispmode = "mig_mat"  # raphael/mig_mat possible
    sigma = 1.0       
    IBD_treshold = 4.0  # Threshold over which IBD stored.
    delete = True  # If TRUE: blocks below threshold are deleted and not traced back any more..
    drawlist_length = 1000  # Variable for how many random Variables are drawn simultaneously.
    max_t = 200  # Runs the Simulations for time t.
    barrier_pos = 100  # The Position of the Barrier

    # The Parameters of the 8 Scenarios.
#     sigmas = [[0.3, 0.5], [0.3, 0.5], [0.4, 0.4], [0.4, 0.4], [0.3, 0.5][0.3, 0.5], [0.3, 0.5], [0.3, 0.5]]
#     assert(len(sigmas) == 8)
#     nr_inds = [[30, 30], [1000, 1000], [40, 20], [1500, 1000], [40, 20], [1500, 1000], [20, 40], [1000, 1500]]
#     assert(len(nr_inds) == 8)
#     betas = [0, 1, 0, 1, 0, 1, 0, 1]
#     assert(len(betas) == 8)
    
    sigmas = [[0.5, 0.5], ]
    nr_inds = [[20, 20], ]
    betas = [0, ]
    
    # Position_List:
    # position_list = [(85 + i * 2, 85 + j * 2, 0) for i  # For test of small grid
    #             in range(15) for j in range(15)]
    
    # New Position List
    position_list = [[85, 95], [95, 95], [105, 95], [115, 95], [85, 105], [95, 105], [105, 105], [115, 105],
                     [85, 115], [95, 115], [105, 115], [115, 115]]

    pop_size = 20  # Nr of individuals per position.
    
    
    
    def __init__(self, data_folder, nr_data_sets, position_list=[], multi_processing=0):
        '''
        Constructor. Sets the Grid class to produce and the MLE class to analyze the Data
        '''
        self.data_folder = data_folder
        self.nr_data_sets = nr_data_sets
        self.multi_processing = multi_processing 
        self.set_position_list()  # Creates the Position List
        self.gridsize = self.gridsize + self.gridsize % 2 # Make Sure Grid Size is even.
        
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
        
    
        # The Parameters of the 8 Scenarios.
        grid.sigmas = np.array(self.sigmas[scenario])
        grid.nr_inds = np.array(self.nr_inds[scenario])
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

        
    def single_run(self, data_set_nr, scenario=0):
        '''Does a single run. 
        Scenario gives the number of the scenario which is run.
        data_set_nr give the data_set number which has to be run'''
        
        print("Doing run %i for Scenario %i" % (data_set_nr, scenario))
        
        # Makes the Grid and sets all Parameters
        grid = factory_Grid(model="hetero") # Creates Grid with Default Parameters
        grid = self.set_grid_parameters(grid, scenario=scenario)  # Resets this Default Parameters
        grid.reset_grid()  # Delete everything and re-initializes Grid
        
        
        # Set the Samples 
        print(self.position_list[:20])
        grid.set_samples(self.position_list)  # Set the samples
        # Runs the Simulations for time t
        grid.update_t(self.max_t)  # Do the actual run
        
        # Do the maximum Likelihood estimation
        # mle_ana = grid.create_MLE_object(bin_pairs=True, plot=True)
        mle_ana = grid.create_MLE_object(reduce_start_list=True, plot=True)  # Create the MLE-object
        #mle_ana.create_mle_model("constant", grid.chrom_l, [1.0, 1.0], diploid=False)  # Runs the analysis. 
        mle_ana.create_mle_model("hetero", grid.chrom_l, [50.0, 0.4], diploid=False)
        mle_ana.mle_analysis_error()  # Analyses the samples
        
        ci_s = mle_ana.ci_s
        estimates = mle_ana.estimates
        
        
        print("RUN COMPLETE!!")
        
        self.save_estimates(estimates, ci_s, data_set_nr, scenario)
        print("Results SAVED!") 
        
        # Save addtional Infos about the run
        if data_set_nr == 0:
            print("To Implement")
        print("Run Complete UUUUHHH YYEEEEAHH!")
        

#########################################################################################

# ## Here are Methods that can create and analyze Data Sets:
def cluster_run(data_set_nr, scenarios=8, replicates=10):
    '''Script to run stuff on the cluster.'''
    eff_run_nr = data_set_nr % replicates
    eff_scenario = data_set_nr / scenarios
    
    assert(eff_scenario * replicates + eff_run_nr == data_set_nr)  # Sanity Check.
    multirun = MultiRunHetero("./hetero_runs", 10)
    multirun.single_run(data_set_nr, scenario)


if __name__ == "__main__":
    data_set_nr = 1
    scenario = 0
    # data_set_nr = int(sys.argv[1])  # Which data-set to use
    #cluster_run()
    
    multirun = MultiRunHetero("./testfolder", 10)
    multirun.single_run(data_set_nr, scenario)



















