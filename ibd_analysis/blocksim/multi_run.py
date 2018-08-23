'''
Created on February 6th, 2018
Contains Class that does multiple replicate runs.
Has infrastructure for that in place. Various tests inherit from 
that class. Basically an extensive wrapper for the Grid class.
Parameters for every single run can found as class variables.
@author: Harald Ringbauer
'''

from grid import factory_Grid

import numpy as np
import cPickle as pickle
import os
import sys  # @UnusedImport


class MultiRun(object):
    '''
    Class that can run replicates of Datasets. Producing them as
    well as analyzing them. Results are saved to /results subfolder
    Simulations to /simulation subfolder.
    TO OVERWRITE IN SUBCLASSES:
    The Parameters Below:
    self.set_grid_params(), create_mle_object()
    '''
    # MultiRun Parameters
    nr_replicates = 50  # Nr of Replicates
    nr_scenarios = 5  # Nr of Scenarios
    data_folder = "./MultiRun"  # Folder where to save to
    folder = "./MultiRun/Subfolder"  # The Folder (data_folder + subfolder) into which is saved.
    subfolder = ""  # Potential Subfolder (if multiple Replicates are simulated)
    
    # General Parameters:
    multiprocessing = False  # If Multiprocessing available
    processor_nr = 1  # What the Nr of Processors is in that case.
    output = True  # Whether to print out Output
    
    # Parameters for the Grid (important ones), overwrite them in inherited Classes.
    chrom_l = 150  # Length of the chromosome (in cM!)
    gridsize = 496  # 60  # 180/2  # 160 # 180 # 98
    grid = []  # Will become the Grid-Matrix for Chromosomes
    grid1 = []  # Will become the Grid-Matrix for previous generation
    IBD_blocks = []  # Detected IBD-blocks 
    rec_rate = 100.0  # Everything is measured in CentiMorgan; Float!
    dispmode = "laplace"  # normal/uniform/laplace/laplace_refl/demes/raphael    #laplace_refl
    sigma = 2.99  # 1.98  #0.965 #sigma = 1.98      
    IBD_detect_threshold = 4.0  # Threshold over with IBD blocks are detected (in cM)
    IBD_treshold = 4.0  # Threshold for which IBD blocks are filtered (in cM)
    delete = True  # If TRUE: blocks below threshold are deleted
    healing = False  # Whether recombination breakpoints are healed (in Multiblock Generation).
    post_process = False  # Whether to postprocess IBD list (I.e. merge up).
    ps_spacing = 0.01  # The max. spacing of gaps between IBD blocks that are merged (in cM)
    start_list = []  # Remember where initial chromosomes sat
    update_list = []  # Positions which need updating
    max_t = 200  # Time in generations back
    drawlist_length = 100000  # Variable for how many random Variables are drawn simultaneously
    
    grid_type = "classic"  # Which Type of Grid: classic/growing/hetero/selfing
    drawer = 0  # Object that draws Parents. Important since this one is about Demography.
    
    position_list = [(235 + i * 2, 235 + j * 2, 0) for i  # For test of small grid
                 in range(15) for j in range(15)]
    
    # Parameters for Inference:
    mle_model = "constant"  # Which MLE Model to use for Inference: constant, doomsday, power_growth, ddd, hetero
    reduce_start_list = False
    bin_pairs = True
    start_param = [0.5, 2.0]  # At which Parameters to start Inference.
    min_len = 3  # Minimum Block length to analyze (cM).
    max_len = 12  # Maximum Block length to analyze (cM).
    
    def __init__(self, folder, subfolder="", replicates=0, multi_processing=0):
        '''
        Constructor. Sets the Grid class to produce and the MLE class to analyze the Data
        '''
        self.folder = self.data_folder + folder
        self.subfolder = subfolder
        # Create Directory if needed
        directory = os.path.dirname(self.folder)
        print("Directory: %s" % directory)
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        if replicates > 0:
            self.nr_replicates = replicates
        self.multi_processing = multi_processing  # Whether to use MultiProcessing.
        
    def get_effective_nr(self, run):
        '''Return effective Scenario Nr and Dataset Nr for run.'''
        eff_run_nr = run % self.nr_replicates  
        eff_scenario = run / self.nr_replicates
        assert(eff_scenario * self.nr_replicates + eff_run_nr == run)  # Sanity Check.
        return eff_scenario, eff_run_nr
    
    ############################
    # Methods for Saving and Loading
    
    def save_estimates(self, params, ci_s, run, filename=None):
        '''Saves Estimates. Params: Parameters of best estimates.
        ci_s: Confidence Intervalls. Standardized Format: .csv Files'''
        if not filename:
            filename = "estimate"
        full_path = self.folder + self.subfolder + "/" + filename + str(run).zfill(2) + ".csv"
        # Check whether Directory exists and creates it if necessary
        
        print("Saving Estimates...")
        self.check_directory(full_path) 
        data = np.column_stack((params, ci_s))  # Stacks Estimates and Parameters together
        np.savetxt(full_path, data, delimiter="$")  # Save the coordinates
    
    def load_ibd_blocks(self, run):
        '''Loads IBD Blocks'''
        full_path = self.folder + self.subfolder + "/blocks" + str(run).zfill(2) + ".p"
        ibd_blocks = pickle.load(open(full_path, "rb"))
        return ibd_blocks
    
    def save_ibd_blocks(self, run, grid):
        '''Saves IBD blocks. 
        run: Which run.''' 
        ibd_blocks = grid.IBD_blocks
        print("Saving IBD blocks...")
        full_path = self.folder + self.subfolder + "/blocks" + str(run).zfill(2) + ".p"
        self.check_directory(full_path)
        pickle.dump(ibd_blocks, open(full_path, "wb"))
    
    def save_parameters(self, filename=None):
        '''Save important Parameters that are used'''
        if not filename:
            filename = "parameters.csv"
            
        full_path = self.folder + self.subfolder + "/" + filename
        self.check_directory(full_path)
        param_names = ["sigma", "chrom_length"]
        values = [self.sigma, self.chrom_l]
        data = np.column_stack((param_names, values))  # Stacks Estimates and Parameters together
        np.savetxt(full_path, data, delimiter="$", fmt="%s")  # Save the coordinates)
        
    def check_directory(self, path):
        """Create Directory if not already there."""
        directory = os.path.dirname(path)  # Extract Directory
        print("Directory: %s" % directory)
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    ############################
    # Methods for Producing Data
    def create_grid(self, run):
        '''Create Grid object; with right Parameters.'''
        grid = factory_Grid(model=self.grid_type)  # Creates empty grid (with whatever defaults there are over there)
        grid = self.set_grid_params(grid, run=run)  # Set the Parameters.
        grid.reset_grid()  # Delete everything and re-initializes Grid (but store Parameters!)
        grid.set_samples(self.position_list)  # Set the samples
        return grid
    
    def set_grid_params(self, grid, run):
        '''Sets custom Parameters of Grid object. OVERWRITE THIS IN SUBCLASSES'''
        _, _ = self.get_effective_nr(run)  # Get the effective Number of the Run
        grid.chrom_l = self.chrom_l
        grid.gridsize = self.gridsize  # 60  # 180/2  # 160 # 180 # 98
        grid.rec_rate = self.rec_rate  # Everything is measured in CentiMorgan; Float!
        grid.dispmode = self.dispmode  # normal/uniform/laplace/laplace_refl/demes/raphael       
        grid.IBD_detect_threshold = self.IBD_detect_threshold  # Threshold over with IBD blocks are detected (in cM)
        grid.IBD_treshold = self.IBD_treshold  # Threshold for which IBD blocks are filtered (in cM)
        grid.delete = self.delete  # If TRUE: blocks below threshold are deleted.
        grid.healing = self.healing
        grid.post_process = self.post_process
        grid.set_sigma(self.sigma)  # Setting dispersal requires that Grid Drawer is reset as well!
        grid.drawlist_length = self.drawlist_length  # Variable for how many random Variables are drawn simultaneously.
        grid.output = self.output
        grid.max_t = self.max_t
        return grid
    
    #################################
    # Methods to analyze Data
    
    def create_mle_object(self, run, grid):
        '''Creates the MLE Object'''
        mle_ana = grid.create_MLE_object(reduce_start_list=self.reduce_start_list, bin_pairs=self.bin_pairs)
        mle_ana.create_mle_model(self.mle_model, grid.chrom_l, start_param=self.start_param, diploid=False)
        return mle_ana
    
    ################################# 
    def simulateBlocks(self, run, save_blocks=False):
        """Run IBD block Simulation. Create and Return the Grid object."""
        if self.output == True:
            eff_scenario, eff_run_nr = self.get_effective_nr(run)
            print("Doing Run %i" % run)
            print("(Run %i for Scenario %i)" % (eff_run_nr, eff_scenario))
        
        # ## Start with creating the grid:
        grid = self.create_grid(run)
        
        if self.output == True:
            print("\nGrid Parameters:")
            print("Sigma: %.2f" % grid.sigma)
            print("Grid Size: %i" % grid.gridsize)
            print("Nr. of Samples successfully set: %i" % len(grid.start_list))
            print("Disp. Mode: %s" % grid.dispmode)
            print("Delete: %r" % grid.delete)
            print("Healing: %r" % grid.healing)
            print("Post Processing: %r \n" % grid.post_process)
            print("IBD Threshold Detection: %.4f" % grid.IBD_detect_threshold)
            print("IBD Threshold: %.4f" % grid.IBD_treshold)
            print("Maximum Runtime: %i" % grid.max_t)
            print("Chromosome Length: %.5f M" % (grid.chrom_l / grid.rec_rate))
        
        if run == 0:  # Saves important parameters that were used.
            self.save_parameters()
            
        grid.update_t(grid.max_t)  # Do the actual run
        
        if save_blocks == True:
            self.save_ibd_blocks(run, grid)
            
        return grid
           
    def single_run(self, run, load_blocks=False, save_blocks=False):
        '''Do a single run. 
        Scenario gives the number of the scenario which is run.
        data_set_nr give the data_set number which has to be run'''
        
        # First simulate, or load blocks:
        if load_blocks == False:
            grid = self.simulateBlocks(run, save_blocks)
            
        elif load_blocks == True:
            grid.IBD_blocks = self.load_ibd_blocks(run)  # Load the blocks
        
        ############################################################################
        # Maximum Likelihood Analysis:
        # Create the Inference Object:
        
        self.estimation_params(run, grid)  # Do the Parameter Estimates.
        
        # Optional: Save addtional Infos about the run
        if data_set_nr == 0:
            print("Please implement this. Please.")
        
        if self.output == True:
            print("Run Complete! Past Harald is so proud.")
    
    def estimation_params(self, run, grid):
        '''Function where the Parameter Estimation as well as Saving is done.
        OVERWRITE'''
        mle_ana = self.create_mle_object(run, grid)  # Create the MLE Object  
        mle_ana.mle_analysis_error()  # Analyse the samples
        
        # Saves the Estimates
        ci_s = mle_ana.ci_s
        estimates = mle_ana.estimates
        self.save_estimates(estimates, ci_s, run)

######################################################################################### 


class MultiSelfing(MultiRun):
    '''
    Tests 50 Runs for four 5 different Selfing Strengths.
    '''
    # position_list = [(235 + i * 2, 235 + j * 2, 0) for i  # 
    #         in range(15) for j in range(15)]
    
    grid_type = "selfing"  # Which Type of Grid: classic/growing/hetero/selfing
    
    position_list = [(230 * 2 + i * 2, 230 * 2 + j * 2, 0) for i  # Multiply factor of two to make grid big enough!
             in range(20) for j in range(20)]
    selfing_rates = [0, 0.5, 0.7, 0.8, 0.9, 0.95]  # The Parameters for selfing
    max_ts = [400, 500, 600, 700, 800, 1000]  # Max t
    
    # Single Grid Parameters:
    start_params = [0.5, 1.0]  # A bit off to be sure.
    sigma = 2.99  # Sigma used in the Simulations.
    chrom_l = 150
    rec_rate = 100
    gridsize = 496 * 2
    IBD_treshold = 3.0
    
    min_len = 3.0  # Minimum Block length to analyze (cM).
    max_len = 12.0  # Maximum Block length to analyze (cM).
    
    def set_grid_params(self, grid, run):
        '''Sets custom Parameters of Grid object. OVERWRITE THIS IN SUBCLASSES'''
        eff_scenario, _ = self.get_effective_nr(run)  # Get the effective Number of the Run
        assert(eff_scenario < len(self.selfing_rates))  # Sanity Check
        
        selfing_rate = self.selfing_rates[eff_scenario]
        max_t = self.max_ts[eff_scenario]
        grid.selfing_rate = selfing_rate
        grid.max_t = max_t
        
        if self.output == True:
            print("Selfing Rate: %.4f" % grid.selfing_rate)
            print("Max. t: %i" % grid.max_t)
        grid.chrom_l = self.chrom_l  # 150 is standard
        grid.gridsize = self.gridsize  # Multiply factor of 2 to make grid big enough!
        grid.rec_rate = self.rec_rate  # Everything is measured in CentiMorgan; Float!
        grid.dispmode = "laplace"  # normal/uniform/laplace/laplace_refl/demes/raphael       
        grid.IBD_detect_threshold = 0.0  # Threshold over with IBD blocks are detected (in cM)
        grid.IBD_treshold = self.IBD_treshold  # Threshold for which IBD blocks are filtered (in cM)
        grid.delete = False  # If TRUE: blocks below threshold are deleted.
        grid.healing = True
        grid.post_process = True
        grid.set_sigma(self.sigma)  # Set Sigma, and resets the drawer object.
        grid.drawlist_length = self.drawlist_length  # Variable for how many random Variables are drawn simultaneously.
        grid.output = self.output
        return grid
    
    #################################
    # Methods to analyze Data
    
    def create_mle_object(self, grid):
        '''Creates the MLE Object'''
        mle_ana = grid.create_MLE_object(reduce_start_list=self.reduce_start_list, bin_pairs=self.bin_pairs)
        mle_ana.create_mle_model(self.mle_model, grid.chrom_l, start_param=self.start_param, diploid=False)
        mle_ana.mle_object.min_len = self.min_len
        mle_ana.mle_object.max_len = self.max_len
        if self.output == True:
            print("Minimum length analyzed: %.4f cm" % mle_ana.mle_object.min_len)
            print("Maximum length analyzed: %.4f cm" % mle_ana.mle_object.max_len)
        return mle_ana
    
    def estimation_params(self, run, grid):
        '''Function where the Parameter Estimation as well as Saving is done.
        OVERWRITE'''
        mle_ana = self.create_mle_object(grid)  # Create the MLE Object  
        mle_ana.mle_object.print_block_nr()  # Analyse the samples again  
        mle_ana.mle_analysis_error()  # Analyse the samples
        
        # Saves the Estimates
        ci_s = mle_ana.ci_s
        estimates = mle_ana.estimates
        self.save_estimates(estimates, ci_s, run)
        
        #########################################
        # Do the Estimation with shortened Blocks and Chromosome
        eff_scenario, _ = self.get_effective_nr(run)  # Get the effective Number of the Run
        s = self.selfing_rates[eff_scenario]  # Extract the Selfing Rate
        # Find the right Correction Factor:
        cf = (2.0 - 2.0 * s) / (2.0 - s)  # Fraction of effective Recombination Events
        if self.output == True:
            print("\nCorrecting Length of Blocks. Correction Factor: %.4f" % cf)
        grid.correct_length(c=cf)  # Make Blocks (and chromosome) shorter
        mle_ana = self.create_mle_object(grid)  # Recreate the MLE Object
        
        # Apply the correction Factor also to bins: 
        # mle_ana.mle_object.min_len = mle_ana.mle_object.min_len * cf
        # mle_ana.mle_object.max_len = mle_ana.mle_object.max_len * cf
        # mle_ana.mle_object.bin_width = mle_ana.mle_object.bin_width * cf
        # mle_ana.mle_object.create_bins()  # Actually re-calculate bins (it's done in constructor)
        mle_ana.mle_object.print_block_nr()  # Analyse the samples again
        
        mle_ana.mle_analysis_error()  # Analyse the samples
        
        # Saves the Estimates
        ci_s = mle_ana.ci_s
        estimates = mle_ana.estimates
        self.save_estimates(estimates, ci_s, run, filename="corrected")  # Save the corrected Estimates
        


#########################################################################################


class MultiSelfingIBD(MultiSelfing):
    '''
    Generate IBD block data for 4x25 replicates.
    Save the Block data.
    '''
    
    grid_type = "selfing"  # Which Type of Grid: classic/growing/hetero/selfing
    
    position_list = [(230 * 2 + i * 2, 230 * 2 + j * 2, 0) for i  # Multiply factor of two to make grid big enough!
             in range(20) for j in range(20)]
    selfing_rates = [0, 0.5, 0.8, 0.95]  # The Parameters for selfing
    max_ts = [250, 300, 400, 500]  # Max t. Only care about quite long blocks
    
    # Single Grid Parameters:
    start_params = [0.5, 1.0]  # A bit off to be sure.
    sigma = 1.98  # Sigma used in the Simulations.
    chrom_l = 150
    rec_rate = 100
    gridsize = 496 * 2
    IBD_treshold = 4.0
    
    min_len = 3.0  # Minimum Block length to analyze (cM).
    max_len = 12.0  # Maximum Block length to analyze (cM).
    
    
    ###########################################
    def save_ibd_blocks(self, run, grid):
        '''Save linearized IBD blocks. 
        Save (np.array(pair_dist), np.array(pair_IBD), np.array(pair_nr), np.array(start_list))  
        run: Which run.''' 
        print("Saving IBD blocks...")
        full_path = self.folder + self.subfolder + "/blocks" + str(run).zfill(2) + ".p"
        self.check_directory(full_path)
        
        # Post-Process IBD blocks:
        eff_scenario, _ = self.get_effective_nr(run)
        
        s = self.selfing_rates[eff_scenario]
        print("Selfing Rate: %.2f" % s)
        cf = (2.0 - 2.0 * s) / (2.0 - s)  # Fraction of effective Recombination Event
        
        k = len(grid.IBD_blocks)
        grid.IBD_blocks = [i for i in grid.IBD_blocks if ((i[1] * cf) > self.IBD_treshold)] # Filter to Minimum Length
        print("Reducing from %i to %i effective IBD blocks" % (k,len(grid.IBD_blocks)))
        
        print("Linearizing IBD sharing...")
        binned_IBD = grid.give_lin_IBD(bin_pairs=True)
        
        pickle.dump(binned_IBD, open(full_path, "wb"))
        print("IBD blocks successfully saved. Well done!!")
        
    def create_grid(self, run):
        '''Create Grid object; with right Parameters. 
        Overwrite so that grid is rescaled with right length.'''
        grid = factory_Grid(model=self.grid_type)  # Creates empty grid (with whatever defaults there are over there)
        grid = self.set_grid_params(grid, run=run)  # Set the Parameters.
        grid.reset_grid()  # Delete everything and re-initializes Grid (but store Parameters!)
        
        # Recalculate the length of the Chromosome!!
        eff_scenario, _ = self.get_effective_nr(run)
        s = self.selfing_rates[eff_scenario]
        cf = (2.0 - 2.0 * s) / (2.0 - s)  # Fraction of effective Recombination Event
        grid.chrom_l = grid.chrom_l / cf  # Rescale the Chromosome so that "effective" length remains the same
        
        grid.set_samples(self.position_list)  # Set the samples
        return grid

#########################################################################################


# ## Here are Methods that can create and analyze Data Sets:
def factory_multirun(mode="default", folder="", subfolder="", replicates=0):
    '''Produces the right Multirun Object'''
    
    # Choose the Scenario which is to be run:
    if mode == "default":
        multirun = MultiRun(folder=folder, subfolder=subfolder, replicates=replicates)
    
    elif mode == "selfing":
        multirun = MultiSelfing(folder=folder, subfolder=subfolder, replicates=replicates)
    
    elif mode == "selfing_blocks":
        multirun = MultiSelfingIBD(folder=folder, subfolder=subfolder, replicates=replicates)
         
    else:
        raise ValueError("Subclass does not match known Subclasses. Check spelling!")
    
    return multirun

    
# Some testing:
if __name__ == "__main__":
    # data_set_nr = 99
    data_set_nr = int(sys.argv[1])  # Which data-set to use
    # mr = factory_multirun(mode="default", folder="/classic", replicates=10)
    # mr = factory_multirun(mode="selfing", folder="/selfing_3-12cm_sigma3", replicates=50)
    # mr.single_run(run=data_set_nr, save_blocks=False)
    
    # To simulate a single Run of IBD blocks:
    mr = factory_multirun(mode="selfing_blocks", folder="/selfing_block_save", replicates=50)
    mr.simulateBlocks(run=data_set_nr, save_blocks=True)
