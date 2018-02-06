'''
Created on 27.01.2015
The Grid class; basically two matrices (one per chromosome) of lists for blocks sitting there.
Contains methods for updating one generation back in time and looking for IBD
Also a class inheriting from Grid to a Grid allowing for growing/declining populations
by varying the number of chromosomes at every grid
@author: hringbauer
'''

import sys
sys.path.append('../analysis_popres/')
from blockpiece import BlPiece, Multi_Bl
from operator import attrgetter
from random import random
from parent_draw import DrawParent
import bisect
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from analysis import torus_distance
# from analysis_popres.mle_multi_run import MLE_analyse
from mle_multi_run import MLE_analyse  # @UnresolvedImport
# from mle_multi_run import MLE_analyse
from random import shuffle


###################################################################################


class Grid(object):
# Object for the Data-Grid. Contains matrix of lists for chromosomal pieces and methods to update it.    
    chrom_l = 150  # Length of the chromosome (in cM!)
    gridsize = 496  # 60  # 180/2  # 160 # 180 # 98
    sample_steps = 2  # 6/2  # 4
    grid = []  # Will become the Grid-Matrix for Chromosomes
    grid1 = []  # Will become the Grid-Matrix for previous generation
    IBD_blocks = []  # Detected IBD-blocks 
    rec_rate = 100.0  # Everything is measured in CentiMorgan; Float!
    dispmode = "laplace"  # normal/uniform/laplace/laplace_refl/demes/raphael    #laplace_refl
    sigma = 1.98  # 1.98  #0.965 #sigma = 1.98      
    IBD_detect_threshold = 4.0  # Threshold over with IBD blocks are detected (in cM)
    IBD_treshold = 4.0  # Threshold for which IBD blocks are filtered (in cM)
    delete = True  # If TRUE: blocks below threshold are deleted
    healing = False  # Whether recombination breakpoints are healed (in Multiblock Generation).
    post_process = False  # Whether to postprocess IBD list (I.e. merge up).
    ps_spacing = 0.01  # The max. spacing of gaps between IBD blocks that are merged (in cM)
    start_list = []  # Remember where initial chromosomes sat
    update_list = []  # Positions which need updating
    t = 0  # Time in generations back
    drawlist_length = 100000  # Variable for how many random Variables are drawn simultaneously
    output = False  # Whether to output statistics of the run in text form.
    
    
    drawer = 0  # Object for drawing parents   
    
    def __init__(self):  # Initializes an empty grid
        self.grid = np.empty((self.gridsize, self.gridsize, 2), dtype=np.object)  # Create empty array of objects, one for each chromosome
        self.grid1 = np.empty((self.gridsize, self.gridsize, 2), dtype=np.object)  # Creates empty array of object for previous generation
        
        drawer = DrawParent(self.drawlist_length, self.sigma, self.gridsize)  # Generate Drawer object
        self.drawer = drawer.choose_drawer(self.dispmode)
    
    def print_stats(self):
        '''Function that outputs Stats. Overwrite'''
        print("Nr. of samples: %i" % len(self.start_list))
        print("Grid Width: %i" % self.gridsize)
        print(r'$\sigma$: %.3f' % self.sigma)
        print("Dispersal mode: %s" % self.dispmode)
        print("Healing: %r" % self.healing)
        print("Deleting: %r \n" % self.delete)
        
    def set_gridwidth(self, grid_width):
        '''Changes the width of the grid - and everything associated with it'''
        self.gridsize = grid_width
        self.grid = np.empty((self.gridsize, self.gridsize, 2), dtype=np.object)  # Create empty array of objects, one for each chromosome
        self.grid1 = np.empty((self.gridsize, self.gridsize, 2), dtype=np.object)  # Creates empty array of object for previous generation
        
        drawer = DrawParent(self.drawlist_length, self.sigma, self.gridsize)  # Generate Drawer object
        self.drawer = drawer.choose_drawer(self.dispmode)
        
    def set_sigma(self, sigma):
        '''Changes sigma of the grid - and everything associated with it'''
        self.sigma = sigma
        self.grid = np.empty((self.gridsize, self.gridsize, 2), dtype=np.object)  # Create empty array of objects, one for each chromosome
        self.grid1 = np.empty((self.gridsize, self.gridsize, 2), dtype=np.object)  # Creates empty array of object for previous generation
        
        drawer = DrawParent(self.drawlist_length, self.sigma, self.gridsize)  # Generate Drawer object
        self.drawer = drawer.choose_drawer(self.dispmode)
    
    def set_samples(self, position_list=0):
        '''Sets sample chromosomes on the grid'''          
        if position_list == 0:  # In case no position list given        
            position_list = [(i + self.sample_steps / 2, j + self.sample_steps / 2, 0) for i 
                             in range(0, self.gridsize, self.sample_steps) for j in range(0, self.gridsize, self.sample_steps)]
        # print(position_list[0])
        self.set_chromosome(position_list) 
        # self.IBD_matrix = np.zeros((l, l, self.chrom_l * 10), dtype=np.int_)  # Matrix for IBD status in 0.1 cM steps
    
    def set_random_samples(self, k):
        '''Picks k random samples'''
        position_list = [(i + self.sample_steps / 2, j + self.sample_steps / 2, 0) for i in 
                         range(0, self.gridsize, self.sample_steps) for j in range(0, self.gridsize, self.sample_steps)]
        shuffle(position_list)  # Randomize position List
        self.set_samples(position_list[:k])  # Set the samples
                     
    def set_chromosome(self, positions):  # Initializes Chromosome on the given list of positions (List with entry (pos_x,pos_y,chrom) )
        for i in positions:
            self.update_list.append(i)               
            self.grid[i] = [BlPiece(i, 0, self.chrom_l)]  # Create chromosome block
            self.start_list.append(i)
    
    def give_start_list_positions(self):
        '''Gives the geogrraphic positions of the startlist as nx2 array'''
        position_list = np.array([[pos[0], pos[1]] for pos in self.start_list])
        return position_list
        
    def add_block(self, position, start, end):
        '''Adds desired block to grid'''
        if self.grid[position] == None:  # In case nothing is there already add an empty list
            self.grid[position] = []
            self.update_list.append(position)
            
        self.grid[position].append(BlPiece(position, start, end))  # Generates and appends desired block piece
        
    def add_block1(self, position, block):
        '''Quickly adds block without recombination event'''    
        if self.grid1[position] == None:  # In case nothing is there already add an empty list
            self.grid1[position] = []
            self.update_list.append(position)  # Write position in update List 
        self.grid1[position].append(block)  # Generates and appends desired block piece
        
        
    def add_block_rec(self, position, block, start, end):
        '''Adds block hit by recombination / More complicated since Multiblocks possible'''
        # Check whether block too short
        if self.delete == True:
            if (end - start) < self.IBD_detect_threshold:  # If smaller than IBD_Treshold STOP
                return
        
        if self.grid1[position] == None:  # In case nothing is there already add an empty list
            self.grid1[position] = []
            self.update_list.append(position)  # Write position in update List
        
        if isinstance(block, Multi_Bl):  # If already complicated block
            subblocks = block.sub_blocks  # Extract subblocks
            newblocks = [[max(subblock.start, start), min(subblock.end, end), subblock.origin] for subblock in subblocks]
            newblocks = [block for block in newblocks if (block[1] - block[0]) > (self.delete * self.IBD_detect_threshold)]  # Only positive lengths and above treshold
            red_subblocks = [BlPiece(i[2], i[0], i[1]) for i in newblocks]
            if red_subblocks:  # Only append blocks if they are actually there
                self.grid1[position].append(Multi_Bl(red_subblocks, healing=self.healing))
                    
        elif isinstance(block, BlPiece):  # Update simple block
            self.grid1[position].append(BlPiece(block.origin, start, end))    
               
    def reset_grid(self):
        '''Method to reset the Grid and delete all blocks.'''
        self.grid = np.empty((self.gridsize, self.gridsize, 2), dtype=np.object)
        self.update_list = []
        self.t = 0
        self.IBD_blocks = []
        self.start_list = []
        
    def update_IBD_blocks_demes(self, deme_size):
        '''Updates the position of IBD-blocks to be in center of the deme-size; and update start list'''
        for i in range(0, len(self.IBD_blocks)):
            origin1 = self.IBD_blocks[i][2]  # Extract coordinates
            origin2 = self.IBD_blocks[i][3]
            origin1 = (self.mean_deme_position(origin1[0], deme_size), self.mean_deme_position(origin1[1], deme_size), origin1[2])  # Modify coordinates accordingly
            origin2 = (self.mean_deme_position(origin2[0], deme_size), self.mean_deme_position(origin2[1], deme_size), origin2[2])                    
            self.IBD_blocks[i] = (self.IBD_blocks[i][0], self.IBD_blocks[i][1], origin1, origin2, self.IBD_blocks[i][4])  # Modify whole entry 
        
        for i in range(0, len(self.start_list)):  # Also update start list
            self.start_list[i] = (self.mean_deme_position(self.start_list[i][0], deme_size), self.mean_deme_position(self.start_list[i][1], deme_size), self.start_list[i][2])         
    
    def create_new_grid(self, nr_inds_pn=2):
        '''Generates an empty grid to fill up with stuff'''
        return np.empty((self.gridsize, self.gridsize, 2 * nr_inds_pn), dtype=np.object)  # Delete Update grid    
                  
    def generation_update(self):  
        '''Updates a single generation'''       
        update_list = self.update_list  # Make working copy of update list
        self.update_list = []  # Delete update list
        
        for position in update_list:
            x, y = position[0], position[1]
            value = self.grid[position]
            
            # In case of a single block send it to updater:    
            if len(value) == 1:  
                self.update_single_block(value[0], (x, y))
             
            # In case of multiple blocks detect IBDs and do whole chromosome break points                   
            elif len(value) >= 2:  
                self.grid[position].sort(key=attrgetter('start'))  # First sort list of blocks according to their start position:
                self.IBD_blocks += self.IBD_search(position)  # Do IBD detection
                self.merge_blocks(position)  # Merge Blocks
                
                rec_points, ancestry = self.create_break_points((x, y))  # Gets random recombination break points and ancestry of blocks
                    
                for block in self.grid[position]:
                    i = bisect.bisect_right(rec_points, block.start)  # The first rec-point greater than start of the block
                    bl_start = block.start  # The first new block
                    if rec_points[i] >= block.end:
                        self.add_block1(ancestry[i], block)
                        continue
                    
                    bl_end = rec_points[i]                        
                        
                    while bl_end < block.end:
                        self.add_block_rec(ancestry[i], block, bl_start, bl_end)  # ancestry[i] is ancestry before breakpoint
                        i += 1
                        bl_start = bl_end
                        bl_end = rec_points[i]
                        
                    self.add_block_rec(ancestry[i], block, bl_start, block.end)  # Do the last block, possibly end of chromosome
                        
        self.grid = self.grid1  # Update the grid
        self.t += 1                              
    
    def get_parents_pos(self, x, y): 
        '''Yield the parental chromosomes given position (x,y)'''
        (x1, y1) = tuple(self.drawer.draw_parent((x, y)))  # Draw first parental position    
        chrom_1 = (random() < 0.5)  # Draw random boolean for first parental chromosome
        chrom_2 = not chrom_1
        pos1 = (x1, y1, int(chrom_1))  # Make Boolean Integer so that indexing works
        pos2 = (x1, y1, int(chrom_2))
        return (pos1, pos2)  # Return the position of the two parental chromosomes
                      
    def update_single_block(self, block, (x, y)):
        '''Updates the given block to its given parental positions'''
        recpoint = block.start  # Save last recombination points
        
        pos1, pos2 = self.get_parents_pos(x, y)  # Get parental positions
        
        r = np.random.exponential(scale=self.rec_rate)  # First rec. point
        if (recpoint + r) >= block.end:  # If only one block
                self.add_block1(pos1, block)
                return  # Finished
        while True:
            self.add_block_rec(pos1, block, recpoint, recpoint + r)  # Add block
            
            recpoint += r  # Update to new start
            r = np.random.exponential(scale=self.rec_rate)  # Next recombination
            if (recpoint + r) >= block.end:  # Break if over limit
                self.add_block_rec(pos2, block, recpoint, block.end)  # Add final block
                return
            self.add_block_rec(pos2, block, recpoint, recpoint + r)  # Add block
            
            recpoint += r  # Update to new start
            r = np.random.exponential(scale=self.rec_rate)  # Next recombination
            if (recpoint + r) >= block.end:  # Break if over limit
                self.add_block_rec(pos1, block, recpoint, block.end)  # Add final block
                return               
    
    def update_t(self, t):
        '''Updates the Grid t generations'''
        
        # Print Output if needed:
        if self.output == True:
            self.print_stats()
        start = timer()
        for i in range(0, t):
            print("Doing step: " + str(i))
            self.grid1 = self.create_new_grid()  # Make new empty update grid 
            self.generation_update()
        end = timer()
        
        if self.post_process == True:
            self.post_process_IBD(spacing=self.ps_spacing)
            self.filter_IBD_blocks(self.IBD_treshold)  # Filter to blocks above threshold
        
        print("Time elapsed: %.3f" % (end - start))
        print("IBD Blocks found: " + str(len(self.IBD_blocks)))      
            
            
    def create_break_points(self, (x, y)):
        '''Create a set of breakpoints for the whole chromosome and returns it as list
        Also get according parent position list'''
        rec_point = 0  # The first rec point sits at 0 ofc
        rec_points = [0]
        while True:  # Generate List of breakpoints
            r = np.random.exponential(scale=self.rec_rate)
            rec_point += r
            
            if rec_point < self.chrom_l:
                rec_points.append(rec_point)
            
            else:  # If end reached return full list of recombination points + chromosome ends
                rec_points.append(self.chrom_l)
                break
            
        pos1, pos2 = self.get_parents_pos(x, y)
        n = len(rec_points)
        ancestry = [pos1, pos2] * (n / 2) + [pos1] * (n % 2)  # Create full lenth ancestral pos vector
        return (rec_points, ancestry)
    
    def IBD_search(self, location):
        '''Takes list of blocks and their position at given position as input and returns list of IBD-segments above threshold '''       
        block_list = self.grid[location]
        block_list.sort(key=attrgetter('start'))  # First sort list of blocks according to their start position:
        
        position = 0  # Current search position
        IBD_list = []
        
        # Check for pairwise overlaps:    
        n = len(block_list)  # Access length of blocks to avoid new blocks in loop
        for i in range(0, n):  # Check every possible overlap with this block
            block = block_list[i]
            position = block.end
            for j in range(i + 1, n):  # Check with all higher blocks
                candidate = block_list[j]
                if candidate.start <= position:  # Detect overlap
                    length = (min(position, candidate.end) - candidate.start)
                    if length > self.IBD_detect_threshold:  # Trigger IBD detection procedure
                        IBD_blocks = self.IBD_overlap(block, candidate)  # Get overlaps and all sub-blocks
                        IBD_list += IBD_blocks
                        # candidate.update_length(position - (self.IBD_detect_threshold - 1), candidate.end)  # To avoid late double findings delete overlap for the second block.
                else:
                    break  # Stop search for this block (start of following blocks beyond its end)
        return IBD_list
    
    def merge_blocks(self, location):
        block_list = self.grid[location]
        # Go along chromosome and add new multi-blocks:
        end = block_list[0].end  # First do the first blocks
        subblocks = [block_list[0]]
        merged_blocks = []
        
        for i in block_list[1:]:
            if (i.start > end):  # If gap
                merged_blocks.append(Multi_Bl(subblocks, healing=self.healing))
                end = i.end
                subblocks = [i]
            else:
                subblocks.append(i)  # Else append to overlapping block list
                
            if i.end > end:  # Extend end if necessary
                end = i.end
        merged_blocks.append(Multi_Bl(subblocks, healing=self.healing))  # For last block.
        self.grid[location] = merged_blocks  # Set blocks to sorted blocks

    def IBD_overlap(self, block1, block2):
        '''Detects overlap between block1 and block2 (can be multiblocks)'''
        IBD_list = []
        
        # Extract possible subblocks
        bl1, bl2 = [], []
        if isinstance(block1, Multi_Bl):
            bl1 += block1.sub_blocks
        else: bl1.append(block1)
                            
        if isinstance(block2, Multi_Bl):
            bl2 += block2.sub_blocks
        else: bl2.append(block2)
        
        # Check all possible pairs of subblocks
        for b1 in bl1:
            for b2 in bl2:
                end = min(b1.end, b2.end)
                start = max(b1.start, b2.start)
                length = end - start
                if length >= self.IBD_detect_threshold:
                    IBD_list.append((start, length, b1.origin, b2.origin, self.t))           
        return(IBD_list)
        
    def mean_deme_position(self, position, deme_size):
        '''Return the middle position of the deme under question. Same as used in deme_drawer'''
        return((deme_size * np.around(position / float(deme_size) + 0.001)) % self.gridsize)  # 8-12->10 
    
    def plot_distribution(self):
        '''Plots the distribution of the Chromosomes on current grid'''      
        x_list, y_list, colors, size = [], [], [], []
        # First extract the data from the last slice
        
        for (x, y, chrom), value in np.ndenumerate(self.grid):  # Iterate over all positions @UnusedVariable
            if value:  # Basically pythonic for if list not empty
                for block in value:
                    if isinstance(block, Multi_Bl):
                        block_list = block.sub_blocks
                    else:   
                        block_list = [block]
                        
                    for block in block_list:
                        x_list.append(x)
                        y_list.append(y)
                        colors.append(20 * block.origin[0] + block.origin[1])
                        size.append(block.end - block.start)
            
        size = [5 * s for s in size]  # Heuristic scale factor
        
        # Now do the plot: (marker="|")
        plt.scatter(x_list, y_list, c=colors, s=size , alpha=0.5)
        plt.xlim([0, self.gridsize - 1])
        plt.ylim([0, self.gridsize - 1])
        plt.title("Generation " + str(self.t))
        # plt.text(1,1,"Generation " + str(self.t))
        plt.show()
        return((x_list, y_list, colors, size))  # Return for possible plots
    
    def give_anc_blocks(self):
        '''Give back list of all blocks in Grid'''
        block_list = []  # The List into which Ancestry is saved into.
        
        for (x, y, chrom), value in np.ndenumerate(self.grid):  # Iterate over all positions
            if value:  # Basically pythonic for if list not empty
                for block in value:
                    if isinstance(block, Multi_Bl):
                        blocks = block.sub_blocks
                    else:   
                        blocks = [block]
                    
                    # Now iterate over every block
                    for block in blocks:
                        block_list.append([x, y, chrom, block.start, block.end, block.origin])
                        
        return block_list 
    

#############################################################################
    # Methods to create MLE object#
       
    def create_MLE_object(self, bin_pairs=False, min_dist=0, reduce_start_list=False, plot=False):
        '''Return initialized MLE-sharing object. 
        Bin_pairs: Whether pairs of individuals shall be grouped
        Min_dist: Minimal pairwise distance.
        '''
        pair_dist, pair_IBD, pair_nr, pos_list = self.give_lin_IBD(bin_pairs=bin_pairs,
                                                         min_dist=min_dist, reduce_start_list=reduce_start_list)  # Get the relevant data
        pair_dist[pair_dist == 0] = 0.001  # To avoid numerical instability for identical pairs
        # Initialize POPRES-MLE-analysis object. No error model used!
        mle_analyze = MLE_analyse(0, pair_dist, pair_IBD, pair_nr, error_model=False, position_list=pos_list)  
        
        # mle_analyze.position_list = self.give_start_list_positions()
        if plot == True:
            mle_analyze.plot_cartesian_position(barrier=[100, 0])
        return mle_analyze
    
    def give_lin_IBD(self, bin_pairs=False, reduce_start_list=False, min_dist=0):
        '''Method which returns pairwise distance, IBD-sharing and pw. Number.
        Used for full MLE-Method. 
        bin_pairs: Pool pairs with same distances. (and calculate pairwise Nr. accordingly) 
        reduce_start_list: Pool with respect to start-list; i.e. individuals with same start-list
        geographical coordinates get pooled. Calculates Number between them.
        min_dist: inimal distance used in analysis.
        Returns Numpy array of pairwise Distances, pairwise IBD sharing, pairwise Nr.; and pairwise'''
        start_list = self.start_list
        ibd_blocks = self.IBD_blocks
        
        # In case of reduced start-list; only extract unique position values.
        if reduce_start_list == True:
            orig_start_list = [(x[0], x[1]) for x in start_list]  # Only extract geographic Positions!
            start_list = [tuple(x) for x in set(tuple([x[0], x[1]]) for x in orig_start_list)]  # Extract all unique geographic Positions
            print("Length of Reduced Start List: %i" % len(start_list))
            # Also overwrite the Geographic Positions in the IBD Blocks:
            ibd_blocks = [[bpair[0], bpair[1], bpair[2][:2], bpair[3][:2]] for bpair in ibd_blocks]
            # Check against original-start-list to create Pairwise Nr
            pop_nr = [orig_start_list.count(x) for x in start_list]
        
        else:
            pop_nr = np.ones(len(start_list))
          
        l = len(start_list) 
        pair_IBD = np.zeros((l * (l - 1) / 2))  # List of IBD-blocks per pair
        pair_IBD = [[] for _ in pair_IBD]  # Initialize with empty lists
        pair_nr = -np.ones((l * (l - 1) / 2))
        
        # First Created Pop-Nr. Vec and then calculate all pairwise Comparisons
        for i in xrange(l):
            for j in xrange(i):
                pair_nr[(i * (i - 1)) / 2 + j] = pop_nr[i] * pop_nr[j]  # Nr of Pairwise Comparisons
        assert(np.min(pair_nr) > 0)  # Sanity Check
        
        # Iterate over all IBD-blocks
        for bpair in ibd_blocks:
            ibd_length = bpair[1]  # Get length in centiMorgan
            ind1 = start_list.index(bpair[2])
            ind2 = start_list.index(bpair[3])    
            j, i = min(ind1, ind2), max(ind1, ind2) 
            if i != j:
                pair_IBD[(i * (i - 1)) / 2 + j].append(ibd_length)  # Append an IBD-block  
        
        # Get distance Array of all blocks
        pair_dist = [torus_distance(start_list[i][0], start_list[i][1],
                                    start_list[j][0], start_list[j][1], self.gridsize) for i in range(0, l) for j in range(0, i)]
        
        if bin_pairs == True:  # Pool data if wanted (speeds up MLE)
            pair_dist, pair_IBD, pair_nr = self.pool_lin_IBD_shr(pair_dist, pair_IBD, pair_nr)
            
        pair_dist, pair_IBD, pair_nr = np.array(pair_dist), np.array(pair_IBD), np.array(pair_nr)  # Make everything a Numpy array.
        
        if min_dist > 0:  # In case where geographical correction is needed.
            inds = np.where(pair_dist > min_dist)  # Extract indices where Pair_Dist is bigger than min_dist.
            pair_dist, pair_IBD, pair_nr = pair_dist[inds], pair_IBD[inds], pair_nr[inds]
        
        assert(len(pair_dist) == len(pair_IBD))
        assert(len(pair_dist) == len(pair_nr))
        print("Pair Dist.:")
        print(pair_dist)
        print("Pair Nr.:")
        print(pair_nr)
        print("Pair IBD:")
        print(pair_IBD)
        return (np.array(pair_dist), np.array(pair_IBD), np.array(pair_nr), np.array(start_list)) 
        
    def pool_lin_IBD_shr(self, pw_dist, pair_IBD, pair_nr):
        '''Bins pairs of same length into one distance pair.
        This does not change the likelihood function but speeds up calculation'''
        distances = sorted(set(pw_dist))  # Produce the keys in a sorted fashion
        
        new_pair_IBD = [[] for _ in distances]  # Initialize the new shortened arrays
        new_pair_nr = [0 for _ in distances]
        
        for j in range(len(distances)):
            r = distances[j]
            for i in range(len(pw_dist)):  # Iterate over all pairs
                if pw_dist[i] == r:  # If Match
                    new_pair_IBD[j] += list(pair_IBD[i])  # Append the shared blocks
                    new_pair_nr[j] += pair_nr[i]  # Add the number of individuals
                    
        print("Nr. of all pairs: %i" % np.sum(new_pair_nr))
        print("Nr of total blocks for analysis: %i" % np.sum([len(i) for i in new_pair_IBD]))
        return(distances, new_pair_IBD, new_pair_nr) 
    
    def post_process_IBD(self, spacing=0.05):
        '''Post process IBD sharing. 
        Pool together IBD blocks that are less than 
        spacing cM apart. Update IBD block list'''
    
        ibd_list = self.IBD_blocks
        bl_list_final = []  # The pruned IBD List. Empty Container
        
        if self.output == True:
            print("Post Processing...")
        start0 = timer()
        # Update the End to absolute end of block:
        ibd_list = [(x[0], x[0] + x[1], x[2], x[3], x[4]) for x in ibd_list]
        k = len(ibd_list)
        
        # Make List of geographic Position of Individuals. Sort them so that same pair of INDs always in the same order!
        geo_inds = [min(x[2], x[3]) + max(x[2], x[3]) for x in ibd_list]    
        
        def unique_rows(data):
            '''Gives back unique rows in data and the indices needed to reconstruct the original thing'''
            data = np.copy(data)  # Make copy to avoid GC getting stuck.
            uniq, indices = np.unique(data.view(data.dtype.descr * data.shape[1]), return_inverse=True)
            return np.copy(uniq.view(data.dtype).reshape(-1, data.shape[1])), np.copy(indices)  
        
        _, inds = unique_rows(geo_inds)  # Extract "unique" Indices
        nr_unique_prs = np.max(inds) + 1  # The Nr of unique pairs (including 0)
        
        bl_ls = [[] for _ in xrange(nr_unique_prs)]  # List of Lists for IBD-Blocks
        
        for i in xrange(len(inds)):
            ind = inds[i]  # Get Index
            bl_ls[ind].append(ibd_list[i])  # Append the block to its unique Position.
            
        def merge_blocks(block_ls, spacing):
            '''Merge blocks between Individuals.
            spacing: Maximal spacing of blocks to be fused.
            block_ls: List of blocks - [[start, end, time]
            I.: First detect all same pairs that share IBD and pool their blocks.
            II.: Then merge these blocks if needed.'''
            assert spacing >= 0  # Sanity Check
    
            block_ls.sort()  # Sort blocks by Start Point
             
            block_ls_final = []  # Empty Container for the final Block List
            start, end, t = block_ls[0]  # Temporary Variables
            
            for bl in block_ls[1:]:
                if (bl[0] - end) < spacing:  # If Overlap
                    t = min(t, bl[2])  # Set time to minimum
                    end = max(bl[1], end)  # Extend
                    
                else:
                    block_ls_final.append((start, end, t))  # Append another Block
                    start = bl[0]
                    end = bl[1]
                    t = bl[2]
            block_ls_final.append((start, end, t))  # Append another Block
            return block_ls_final
        
        
        for blocks in bl_ls:
            t = np.min([x[4] for x in blocks])  # Take the first coalesced chunk as time.
            input_ls = [[x[0], x[1], x[4]] for x in blocks]  # Extract list of block Starts and Ends.
            blocks_final = merge_blocks(input_ls, spacing)  # Do the Merging
            bl = blocks[0]
            
            for start, end, t in blocks_final:
                bl_list_final.append((start, end, bl[2], bl[3], t))

    
        # Restore 2nd entry to relative length of block:
        bl_list_final = [(x[0], x[1] - x[0], x[2], x[3], x[4]) for x in bl_list_final]
        end = timer()
        if self.output == True:
            print("Time for Post-Processing: %.5f s" % (end - start0))
            print("Merged from %i to %i IBD blocks." % (k, len(bl_list_final)))
        self.IBD_blocks = bl_list_final
        
    def filter_IBD_blocks(self, min_l=4, max_l=None):
        '''Filter IBD block list to  blocks above and below some threshold'''
        ibd_blocks = self.IBD_blocks
        # Do the Minimum Cut-Off:
        k = len(ibd_blocks)
        ibd_blocks = [x for x in ibd_blocks if x[1] >= min_l]
        
        # Maximum Cut-Off:      
        if max_l:
            ibd_blocks = [x for x in ibd_blocks if x[1] <= max_l]
        self.IBD_blocks = ibd_blocks
        
        # Some output what was done
        if self.output == True:
            print("Filtering blocks above %.4f cM" % min_l)
            print("From %i to %i blocks" % (k, len(self.IBD_blocks)))
            
        
        
        
#####################################################################################################
class Grid_Grow(Grid):
    '''Class for producing a growing grid'''
    nr_inds_pn = 1  # The Number of chromosomes per node
    
    def __init__(self, **kwds):
        super(Grid_Grow, self).__init__(**kwds)  # Initialize the grid        
    # To Do: Update specific aspects 
    
    def set_chr_pn(self, t_back):
        '''Method to set individuals per node in generation t''' 
        # mu = 200.0 / t_back
        mu = t_back
        # mu = 10  # 10 before change for Hybride Zone Sim (5)
        self.nr_inds_pn = np.around(mu)
        
    def update_t(self, t):
        '''Updates the Grid t generations'''
        start = timer()
        for i in range(0, t):
            print("Doing step: " + str(i))
            self.set_chr_pn(self.t + 1)  # Set Nr of individuals per node t generations back
            self.grid1 = self.create_new_grid(self.nr_inds_pn)  # Make new empty update grid
            self.generation_update()
        end = timer()
        print("Time elapsed: %.3f" % (end - start))
        print("IBD Blocks found: " + str(len(self.IBD_blocks)))  
    
    def get_parents_pos(self, x, y):
        '''Override original method to get parental chromosome position'''
        (x1, y1) = tuple(self.drawer.draw_parent((x, y)))  # Draw parental position 
        p = 2 * np.random.randint(self.nr_inds_pn)  # Draw parent individual begin chromosome
        chrom_1 = (random() < 0.5)  # Draw random boolean for first parental chromosome
        chrom_2 = not chrom_1
        pos1 = (x1, y1, p + chrom_1)
        pos2 = (x1, y1, p + chrom_2)
        return (pos1, pos2)  # Return the position of the two parental chromosomes   


class Grid_Selfing(Grid):
    '''Grid Class which allows for selfing.'''
    selfing_rate = 0.0  # The Selfing rate, i.e. the chance than an individual has only one parent.
    update_list = []  # Positions which need updating. HERE: Individuals instead of chromosomes!
    
    IBD_detect_threshold = 4.0  # Threshold over with IBD blocks are detected (in cM)
    IBD_treshold = 4.0  # Threshold for which IBD blocks are filtered (in cM)
    
    delete = True  # Default that short blocks are not deleted
    healing = False  # Whether broken up blocks are healed.
    post_process = False  # Whether to do some Post-Processing. Overwrite to True.
    
    def __init__(self, **kwds):
        super(Grid_Selfing, self).__init__(**kwds)
        # Now: Maybe some specific Tasks
    
    def extract_unique_positions(self):
        '''Extracts unique geographic Position of Update List.
        UPDATE LATER ON MAYBE FOR INDIVIDUALS in demes.'''
        position_list = tuple(set([(l[0], l[1]) for l in self.update_list]))  # Extract unique Geographic positions
        return position_list
    
    def print_stats(self):
        '''Function that outputs Stats. Overwrite orignal output.'''
        print("Nr. of samples: %i" % len(self.start_list))
        print("Grid Width: %i" % self.gridsize)
        print("Selfing Rate: %.3f" % self.selfing_rate)
        print(r'$\sigma$: %.3f' % self.sigma)
        print("Dispersal mode: %s" % self.dispmode)
        print("Healing: %r" % self.healing)
        print("Deleting: %r \n" % self.delete)
        
    def generation_update(self):
        '''Overwrites update of single generation.
        Mostly the same, but ONE UPDATE LIST'''
        
        update_list = self.extract_unique_positions()  # Get unique geographic positions.
        self.update_list = []  # Delete update list
        
        # Decide whether to self:
        selfings = np.random.random(len(update_list)) < self.selfing_rate  # Generates random list.
        
        for i in xrange(len(update_list)):
            # 0 Outbreeding 1 Selfing.
            x, y = update_list[i][0], update_list[i][1]  # Extract Positions.
            
            par_pos0 = tuple(self.drawer.draw_parent((x, y)))  # Get Position of 1st Parent
            
            
            if selfings[i] == True:
                self.update_chromosome((x, y, 0), par_pos0)
                self.update_chromosome((x, y, 1), par_pos0)
                
            else:
                par_pos1 = tuple(self.drawer.draw_parent((x, y)))  # Draw other Parent as well
                
                self.update_chromosome((x, y, 0), par_pos0)
                self.update_chromosome((x, y, 1), par_pos1)  
                        
        self.grid = self.grid1  # Update the grid
        self.t += 1 
        
    def update_chromosome(self, position, parent_pos):
        '''Updates Chromosome - i.e. list of blocks at postion to parent position.
        Position: [x,y,0]
        Parent_pos: [x,y] '''
        blocks = self.grid[position[0], position[1], position[2]]
        
        # In case of a single block send it to updater:    
        if not blocks:  # In case of no or empty list
            pass  # Do nothing
        
        elif len(blocks) == 1:  
            self.update_single_block(blocks[0], (parent_pos[0], parent_pos[1]))
         
        # In case of multiple blocks detect IBDs and do whole chromosome break points                   
        elif len(blocks) >= 2:  
            # Heal Recombination breakpoints:
            self.grid[position].sort(key=attrgetter('start'))  # Sort list of blocks according to their start position:
            self.IBD_blocks += self.IBD_search(position)  # Do IBD detection
            self.merge_blocks(position)  # Merge Blocks
            
            rec_points, ancestry = self.create_break_points((parent_pos[0], parent_pos[1]))  # Gets random recombination break points and ancestry of blocks
                
            for block in self.grid[position]:
                i = bisect.bisect_right(rec_points, block.start)  # The first rec-point greater than start of the block
                bl_start = block.start  # The first new block
                if rec_points[i] >= block.end:
                    self.add_block1(ancestry[i], block)
                    continue
                
                bl_end = rec_points[i]                        
                    
                while bl_end < block.end:
                    self.add_block_rec(ancestry[i], block, bl_start, bl_end)  # ancestry[i] is ancestry before breakpoint
                    i += 1
                    bl_start = bl_end
                    bl_end = rec_points[i]
                    
                self.add_block_rec(ancestry[i], block, bl_start, block.end)  # Do the last block, possibly end of chromosome 
                
    def get_parents_pos(self, x, y): 
        '''Somewhat Dirty Overwrite for selfing - Always sets parents geographic position to the same Position
        since position is drawn already elsewhere.'''  
        chrom_1 = (random() < 0.5)  # Draw random boolean for first parental chromosome
        chrom_2 = not chrom_1
        pos1 = (x, y, int(chrom_1))  # Make Boolean Integer so that indexing works
        pos2 = (x, y, int(chrom_2))
        return (pos1, pos2)  # Return the position of the two parental chromosomes
        
        
    
class Grid_Heterogeneous(Grid):
    '''Grid Class where coalesence probability depends on the Side of the Barrier.'''
    nr_inds_left = 0  # Nr of diploid Individuals on the left at the current state
    nr_inds_right = 0  # Nr of diploid Individuals on the right at the current state
    # nr_const = 5 # TEMPORARY NUMBER
    barrier_pos = 50  # Where to find the Barrier.
    dispersal_params = []  # Enter the Parameters for Dispersal here.
    dispmode = "mig_mat"
    sigmas = np.array([0.5, 0.5])  # Dispersal Left, Dispersal Right, Position of the Barrier.
    start_inds = np.array([5, 5])  # The Number of Individuals in the first generation.
    nr_inds = np.array([5, 5])  # Nr. of current individuals to the left and to the right of the Barrier.
    beta = 0  # Growth Rate Parameter
    mm_mode = "isotropic"
    
    def __init__(self, **kwds):
        super(Grid_Heterogeneous, self).__init__(**kwds)  # Initialize the grid   
        drawer = DrawParent(self.drawlist_length, self.sigma, self.gridsize)  # Generate Drawer object
        self.drawer = drawer.choose_drawer(self.dispmode)
        self.drawer.set_params(self.sigmas, self.nr_inds, self.barrier_pos)
        self.drawer.init_manual(self.drawlist_length, self.sigmas,
                                self.nr_inds, self.gridsize, balance=self.mm_mode)  # Initializes the drawer correctly.
    
    def reset_grid(self):
        '''Resets Grid and Drawer'''
        self.grid = np.empty((self.gridsize, self.gridsize, np.max(self.start_inds) * 2), dtype=np.object)
        self.update_list = []
        self.t = 0
        self.IBD_blocks = []
        self.start_list = []
        drawer = DrawParent(self.drawlist_length, self.sigma, self.gridsize)  # Generate Drawer object
        self.drawer = drawer.choose_drawer(self.dispmode)
        self.drawer.set_params(self.sigmas, self.nr_inds, self.barrier_pos)
        self.drawer.init_manual(self.drawlist_length, self.sigmas,
                                self.nr_inds, self.gridsize, balance=self.mm_mode)  # Initializes the drawer correctly.
        self.nr_inds_left = self.start_inds[0]
        self.nr_inds_right = self.start_inds[1]
        
    def set_chr_pn(self, t_back):
        '''Method to set individuals per node in generation t''' 
        # Does the Population Growth Scenario
        # mu = 200.0 / t_back
        # mu = t_back
        # mu = self.nr_const  # 10 before change for Hybride Zone Sim (5)
        
        self.nr_inds_left = np.max([np.around(self.start_inds[0] * t_back ** (-self.beta)), 1.0]).astype("int")
        self.nr_inds_right = np.max([np.around(self.start_inds[1] * t_back ** (-self.beta)), 1.0]).astype("int")
        self.nr_inds = np.array([self.nr_inds_left, self.nr_inds_right])  # Set the Number of current Individuals
        self.max_inds = np.max([self.nr_inds_left, self.nr_inds_right])  # The Number of chromosomes per node.
        
        # For Debugging
        # print(t_back)
        # print(self.nr_inds_left)
        # print(self.nr_inds_right)
        
    def update_t(self, t):
        '''Updates the Grid t generations'''
        start = timer()
        for i in range(0, t):
            print("Doing step: " + str(i))
            self.set_chr_pn(self.t + 1)  # Set Nr of individuals per node t generations back
            print(self.nr_inds_left)
            print(self.nr_inds_right)
            self.grid1 = self.create_new_grid(nr_inds_pn=self.max_inds)  # Make new empty update grid
            self.generation_update()
        end = timer()
        print("Time elapsed: %.3f" % (end - start))
        print("IBD Blocks found: " + str(len(self.IBD_blocks)))  
    
    def get_parents_pos(self, x, y):
        '''Override original method to get parental chromosome position'''
        # (x1, y1) = position_update_raphael((x, y), self.grid_size, self.barrier_pos, self.dispersal_params)
        (x1, y1) = tuple(self.drawer.draw_parent((x, y)))  # Draw first parental position 
        if x1 < self.barrier_pos:
            p = 2 * np.random.randint(self.nr_inds_left)  # Draw parent individual begin chromosome
        elif x1 >= self.barrier_pos:
            p = 2 * np.random.randint(self.nr_inds_right)  # Draw parent individual begin chromosome
        
        chrom_1 = int((random() < 0.5))  # Draw random boolean for first parental chromosome
        chrom_2 = int(not chrom_1)
        pos1 = (x1, y1, p + chrom_1)
        pos2 = (x1, y1, p + chrom_2)
        return (pos1, pos2)  # Return the position of the two parental chromosomes
      
    
      
    
########################################################################################################

def factory_Grid(model="classic"):
    '''Factory method to give back Grid'''
    if model == "classic":
        return Grid()
    elif model == "growing":
        return Grid_Grow()
    elif model == "hetero":
        return Grid_Heterogeneous()
    elif model == "selfing":
        return Grid_Selfing()
    else:
        raise ValueError("Enter Valid Model. Check your Spelling!")
    

    
    
