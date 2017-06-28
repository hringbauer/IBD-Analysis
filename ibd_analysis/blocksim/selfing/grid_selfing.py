'''
Created on 27.01.2015
The Grid class; basically two matrices (one per chromosome) of lists for blocks sitting there.
Contains methods for updating one generation back in time and looking for IBD
@author: hringbauer
'''
from blockpiece import BlPiece, Multi_Bl
from operator import attrgetter
from random import random
from parent_draw import DrawParent
import bisect
import numpy as np
# import matplotlib.pyplot as plt
from timeit import default_timer as timer

counter = 0  # Debug Variable!!!!

class Grid(object):
# Object for the Data-Grid. Contains matrix of lists for chromosomal pieces and methods to update it.    
    chrom_l = 150  # Length of the chromosome
    gridsize = 100
    sample_steps = 4
    rec_rate = 100.0  # Everything is measured in CentiMorgan
    dispmode = "laplace"  # normal/uniform/laplace/demes
    sigma = 1.98
    IBD_treshold = 4.0
    delete = True  # If TRUE: blocks below threshold are deleted
    healing = False
    drawlist_length = 100000  # Variable for how many random Variables are drawn simultaneously
    selfing_rate = 0  # The rate of selfing
    
    
    t = 0  # General time Variable
    drawer = 0  # Object for drawing parents
    start_list = []  # Remember where initial chromosome sat
    update_list = []  # Positions which need updating
    IBD_blocks = []  # Detected IBD-blocks (start, length, ind1, ind2, time) length in cM
    IBD_blocks1 = []  # Detected IBD-blocks by all coalescence events
    grid = []  # Will become the Grid-Matrix for Chromosomes
    grid_1 = []  # Will become the Grid-Matrix for previous generation
    IBD_matrix = []  # Matrix for genetic IBD status
    pair_dist = []  # Vector for pairwise distances
    pair_IBD = []  # Vector for pairwise IBD. In Morgan!
    
    def __init__(self):  # Initializes an empty grid
        self.grid = np.empty((self.gridsize, self.gridsize, 2), dtype=np.object)  # Create empty array of objects, one for each chromosome
        self.grid1 = np.empty((self.gridsize, self.gridsize, 2), dtype=np.object)  # Creates empty array of object for previous generation
        
        drawer = DrawParent(self.drawlist_length, self.sigma, self.gridsize)  # Generate Drawer object
        self.drawer = drawer.choose_drawer(self.dispmode)
    
    def set_samples(self, position_list=0):
        '''Sets sample chromosomes on the grid. SETTING SAMPLES HAS TO BE DONE VIA THIS FUNCTION'''
        if position_list == 0:  # In case no position list given        
            position_list = [(i + self.sample_steps / 2, j + self.sample_steps / 2, 0) for i in range(0, self.gridsize, self.sample_steps) for j in range(0, self.gridsize, self.sample_steps)]
        # position_list = [(i + self.sample_steps / 2, j + self.sample_steps / 2, 0) for i in range(48, 75, self.sample_steps) for j in range(48, 75, self.sample_steps)]
        
        l = len(position_list) 
        
        self.set_chromosome(position_list) 
        self.IBD_matrix = np.zeros((l, l, self.chrom_l * 10), dtype=np.int_)  # Matrix for IBD status in 0.1 cM steps
        self.pair_IBD = np.zeros((l * (l - 1) / 2), dtype=np.object)  # List of IBD-blocks per pair
        pair_dist = [self.torus_distance(self.start_list[i][0], self.start_list[i][1], self.start_list[j][0], self.start_list[j][1]) for i in range(0, l) for j in range(0, i)]
        self.pair_dist = np.array(pair_dist)  # Vector for all pairwise distances
                        
    def set_chromosome(self, positions):  # Initializes Chromosome on the given list of positions (List with entry (pos_x,pos_y,chrom) )
        chromosome_index = 0
        for i in positions:
            self.update_list_add(i[0], i[1])  # Add position to update List               
            self.grid[i] = [BlPiece(i, 0, self.chrom_l, index=chromosome_index)]  # Create chromosome block
            self.start_list.append(i)
            chromosome_index += 1
    
    def add_block(self, position, start, end):
        '''Adds desired block to grid'''
        if self.grid[position] == None:  # In case nothing is there already add an empty list
            self.grid[position] = []
            self.update_list_add(position[0], position[1])
            
        self.grid[position].append(BlPiece(position, start, end))  # Generates and appends desired block piece
        
    def add_block1(self, position, block):
        '''Quickly adds block without recombination event'''    
        self.update_list_add(position[0], position[1])  # Write position in update List 
        self.grid1[position].append(block)  # Generates and appends desired block piece
        
        
    def add_block_rec(self, position, block, start, end):
        '''Adds block hit by recombination / More complicated since Multiblocks possible'''
        
        # Check if block too short
        if self.delete == True:
            if (end - start) < self.IBD_treshold:  # If smaller than IBD_Treshold STOP
                return
        
        self.update_list_add(position[0], position[1])  # Write position in update List
        
        if isinstance(block, Multi_Bl):  # If already complicated block
            subblocks = block.sub_blocks  # Extract subblocks
            newblocks = [[max(subblock.start, start), min(subblock.end, end), subblock.origin, subblock.index] for subblock in subblocks]
            newblocks = [block for block in newblocks if (block[1] - block[0]) > (self.delete * self.IBD_treshold)]  # Only positive lengths and above treshold
            red_subblocks = [BlPiece(i[2], i[0], i[1], i[3]) for i in newblocks]
            if red_subblocks:  # Only append blocks if they are actually there
                self.grid1[position].append(Multi_Bl(red_subblocks))
                    
        elif isinstance(block, BlPiece):  # Update simple block
            self.grid1[position].append(BlPiece(block.origin, start, end, block.index))    
    
    def update_list_add(self, x, y):
        '''Add position in grid to update list if nothing is already there'''
        if self.grid1[x, y, 0] == None and self.grid1[x, y, 1] == None:
            self.grid1[x, y, 0] = []  # Make empty lists in positions where blocks will go in.
            self.grid1[x, y, 1] = []
            self.update_list.append((x, y))  # Add position to update list
                   
    def reset_grid(self):
        '''Method to reset the Grid and delete all blocks.'''
        self.grid = np.empty((self.gridsize, self.gridsize, 2), dtype=np.object)
        self.update_list = []
        self.t = 0
        self.IBD_blocks = []
        self.start_list = []
        self.pair_dist = []
        self.pair_IBD = []
        
    def update_IBD_blocks_demes(self, deme_size):
        '''Update the position of IBD-blocks to be in center of the deme-size; and update start list'''
        for i in range(0, len(self.IBD_blocks)):
            origin1 = self.IBD_blocks[i][2]  # Extract coordinates
            origin2 = self.IBD_blocks[i][3]
            origin1 = (self.mean_deme_position(origin1[0], deme_size), self.mean_deme_position(origin1[1], deme_size), origin1[2])  # Modify coordinates accordingly
            origin2 = (self.mean_deme_position(origin2[0], deme_size), self.mean_deme_position(origin2[1], deme_size), origin2[2])                    
            self.IBD_blocks[i] = (self.IBD_blocks[i][0], self.IBD_blocks[i][1], origin1, origin2, self.IBD_blocks[i][4])  # Modify whole entry 
        
        for i in range(0, len(self.start_list)):  # Also update start list
            self.start_list[i] = (self.mean_deme_position(self.start_list[i][0], deme_size), self.mean_deme_position(self.start_list[i][1], deme_size), self.start_list[i][2])         
              
    def generation_update(self):  
        '''Updates a single generation''' 
        self.grid1 = np.empty((self.gridsize, self.gridsize, 2), dtype=np.object)  # Delete Update grid 
        update_list = self.update_list  # Make working copy of update list
        self.update_list = []  # Delete update list: Important: Do this at the beginning
               
        for position in update_list:
            x, y = position[0], position[1]
            parent_pos = self.drawer.draw_parent((x, y))  # Draw the position of the first parent
            
            selfing = np.random.random() < self.selfing_rate  # Whether this individual was produced by selfing
            
            if selfing == True:    
                self.update_chromosome((x, y, 0), parent_pos)  # Update both chromosomes; they are updated to the same parent position
                self.update_chromosome((x, y, 1), parent_pos)
                
            elif selfing == False:
                parent_pos1 = self.drawer.draw_parent((x, y))  # Draw parent of the second chromosome
                self.update_chromosome((x, y, 0), parent_pos)  # Update both chromosomes; they are updated to differing positions
                self.update_chromosome((x, y, 1), parent_pos1)    
                        
        self.grid = self.grid1  # Update the grid
        self.t += 1                             
    
    def update_chromosome(self, position, parent_pos):
            '''Updates chromosome in grid to given parent position'''         
            value = self.grid[position]  # Extract underlying list of blocks
            
            # Check wether anything to do at all
            if value == None:
                return
            
            # In case of a single block send it to updater:    
            if len(value) == 1:  
                self.update_single_block(value[0], parent_pos)  # Updates a single block to its parent position
             
            # In case of multiple blocks detect IBDs and do whole chromosome break points                   
            elif len(value) >= 2:  
                self.grid[position].sort(key=attrgetter('start'))  # First sort list of blocks according to their start position:
                self.IBD_search(position)  # Do IBD detection
                self.merge_blocks(position)  # Merge Blocks into superblocks
                
                rec_points, ancestry = self.create_break_points(parent_pos)  # Gets random recombination break points and ancestry of blocks
                    
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
                
                                    
    def update_single_block(self, block, parent_pos):
        '''Updates the given block to its given parental positions'''
        recpoint = block.start  # Save last recombination points
        x1, y1 = parent_pos[0], parent_pos[1]
            
        chrom_1 = random() < 0.5  # Draw random boolean for first parental chromosome
        r = np.random.exponential(scale=self.rec_rate)  # First rec. point
        if (recpoint + r) >= block.end:  # If only one block
                self.add_block1((x1, y1, chrom_1), block)
                return  # Finished
            
        chrom_2 = not chrom_1  # Do second chromosome from now on
        
        while True:
            self.add_block_rec((x1, y1, chrom_1), block, recpoint, recpoint + r)  # Add block
            recpoint += r  # Update to new start
            r = np.random.exponential(scale=self.rec_rate)  # Next recombination
            if (recpoint + r) >= block.end:  # Break if over limit
                self.add_block_rec((x1, y1, chrom_2), block, recpoint, block.end)  # Add final block
                return
            
            self.add_block_rec((x1, y1, chrom_2), block, recpoint, recpoint + r)  # Add block
            recpoint += r  # Update to new start
            r = np.random.exponential(scale=self.rec_rate)  # Next recombination
            if (recpoint + r) >= block.end:  # Break if over limit
                self.add_block_rec((x1, y1, chrom_1), block, recpoint, block.end)  # Add final block
                return       
            
    
    def update_t(self, t):
        '''Updates the Grid t generations'''
        start = timer()
        for i in range(0, t):
            print("Doing step: " + str(i))
            self.generation_update()
        end = timer()
        print("Time elapsed: %.3f" % (end - start))
        print("Effective IBD Blocks found: " + str(len(self.IBD_blocks1)))   
            
            
            
    def create_break_points(self, parent_pos):
        '''Creates a set of breakpoints for the whole chromosome and returns it as list and a list of ancestry'''
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
            
        c = random() < 0.5  # Random start chromosome
        
        n = len(rec_points)
        parent1 = tuple(parent_pos) + (c,)
        parent2 = tuple(parent_pos) + (not c,)
        ancestry = [parent1, parent2] * (n / 2) + [parent1] * (n % 2)
        return (rec_points, ancestry)
    
    def IBD_search(self, location):
        '''Take list of blocks and their position at given position as input and return list of IBD-segments above threshold '''       
        block_list = self.grid[location]
        block_list.sort(key=attrgetter('start'))  # First sort list of blocks according to their start position:
        
        position = 0  # Current search position
        
        # Check for pairwise overlaps:    
        n = len(block_list)  # Access length of blocks do avoid new blocks in loop
        for i in range(0, n):  # Check every possible overlap with this block
            block = block_list[i]
            position = block.end
            for j in range(i + 1, n):  # Check with all higher blocks
                candidate = block_list[j]
                if candidate.start <= position:  # If overlap:
                    length = (min(position, candidate.end) - candidate.start)
                    if length > 0:
                        self.IBD_mat_update(block, candidate)  # Update IBD-status
                    if length > self.IBD_treshold:  # Trigger IBD detection procedure
                        self.IBD_overlap(block, candidate)  # Get overlaps and all sub-blocks
                        # candidate.update_length(position - (self.IBD_treshold - 1), candidate.end)  # To avoid late double findings delete overlap for the second block. NOT NEEDED WITH SUPERBLOCKS
                else:
                    break  # Stop search for this block (start of following blocks beyond its end)
    
    def merge_blocks(self, location):
        '''Merges blocks into container blocks and heals breakpoints. Needs sorted blocks by start position'''
        block_list = self.grid[location]
        # Go along chromosome and add new multi-blocks:
        end = block_list[0].end  # First do the first blocks
        subblocks = [block_list[0]]
        merged_blocks = []
        
        for i in block_list[1:]:
            if (i.start > end):  # If gap
                if self.healing == True:
                    subblocks = self.rec_heal(subblocks)
                merged_blocks.append(Multi_Bl(subblocks))  # Create a new container block from collected overlapping blocks
                end = i.end
                subblocks = [i]
            else:
                subblocks.append(i)  # Else append to overlapping block list
                
            if i.end > end:  # Extend end if necessary
                end = i.end
        if self.healing == True:
            subblocks = self.rec_heal(subblocks)  
        merged_blocks.append(Multi_Bl(subblocks))  # For last block.
        self.grid[location] = merged_blocks  # Set blocks to merged blocks
    
    def rec_heal(self, blocks):
        '''Heal recombination events within the subblocks and return list of all healed subblocks'''
        subblocks = []  # Container for all sub-blocks
        healed = False  # Indicator variable whether something was healed
        subblocks1 = []  # Container for output
        
        for block in blocks:  # First extract all subblocks
            if isinstance(block, Multi_Bl):
                subblocks += block.sub_blocks
            else: subblocks.append(block)
        subblocks.sort(key=attrgetter('start'))  # First sort list of blocks according to their start position
            
        for i in range(0, len(subblocks)):  # Iterate over all pairs of subblocks
            for j in range(i + 1, len(subblocks)):
                if subblocks[i].end == subblocks[j].start and subblocks[i].origin == subblocks[j].origin:  # If same subblocks merge again
                    healed = True
                    subblocks[j].start = subblocks[i].start  # Extended the second block
            if healed == False:
                subblocks1.append(subblocks[i])  # Only append subblock when it was not healed
            healed = False
        return subblocks1
        
        
    def IBD_overlap(self, block1, block2):
        '''Detect overlap between block1 and block2 (can be multiblocks) and adds results to IBD-list'''
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
                if length >= self.IBD_treshold:
                    self.IBD_blocks.append((start, length, b1.origin, b2.origin, self.t))           
        return(IBD_list)
    
    def IBD_mat_update(self, block1, block2):
        ''' Update IBD-status matrix'''
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
                if b1.index == b2.index: continue  # To avoid self detection
                end = min(b1.end, b2.end)
                start = max(b1.start, b2.start)
                IBD_start = np.floor(10 * start)  # Self IBD-matrix has steps of 0.1 cm!
                IBD_end = np.ceil(10 * end)
                
                self.IBD_matrix[min(b1.index, b2.index), max(b1.index, b2.index), IBD_start:IBD_end] = self.t  # Set IBD_matrix to true for lower triangle; IBD_ned never reaches end of chromosome
            
    def analyze_IBD_mat(self):
        '''Analyze the IBD-matrix for long IBD blocks and update IBD1-list accordingly. And also the IBD-vector.'''
        self.IBD_blocks1 = []
        self.pair_IBD.fill(0)  # Reset IBD-array!
        
        start_mat = np.zeros((len(self.start_list), len(self.start_list)))
        start_mat[:, :] = 0.0
        
        old_mat = self.IBD_matrix[:, :, 0]
        for locus in range(1, len(self.IBD_matrix[0, 0, :])):  # Iterate over all loci
            current_mat = self.IBD_matrix[:, :, locus]
            
            unequal = (old_mat != current_mat) & (np.minimum(old_mat, current_mat) == 0)  # Jump in coalescent time; and one of them ancestral (=0 here)
            IBD_end = unequal & (current_mat == 0) & (start_mat <= (locus - self.IBD_treshold * 10))  # Create Matrix where long enough IBD-blocks end
            IBD_ind = np.nonzero(IBD_end)  # Indices where IBD blocks detected
            
            start = [start_mat[IBD_ind[0][i], IBD_ind[1][i]] for i in range(0, len(IBD_ind[0]))]  # Extract start loci
            coal_time = [old_mat[IBD_ind[0][i], IBD_ind[1][i]] for i in range(0, len(IBD_ind[0]))]  # Extract coalescence times
            self.IBD_blocks1 += [(start[i] / 10.0, (locus - start[i]) / 10.0, self.start_list[IBD_ind[0][i]], self.start_list[IBD_ind[1][i]], coal_time[i]) for i in range(0, len(IBD_ind[0]))]
            
            for l in range(0, len(IBD_ind[0])):
                j, i = min(IBD_ind[0][l], IBD_ind[1][l]), max(IBD_ind[0][l], IBD_ind[1][l])
                self.pair_IBD_add(i, j, (locus - start[l]) / (10.0 * self.rec_rate))  # Normalize Block lengths accordingly
                
                # self.pair_IBD[i * (i - 1) / 2 + j] = (locus - start[l]) / 10.0  # Set chromosome in linearized array
                
            # Update where jumps
            indices = np.nonzero(unequal)
            start_mat[indices[0], indices[1]] = locus
            old_mat[:, :] = current_mat[:, :]
            
            # Output for the user
            print("Doing locus: %.1f" % locus)
            print("New IBD-Blocks found: %.1f \n" % len(IBD_ind[0]))
            
    def conv_IBD_list_to_pair_IBD(self):
        '''Convert the existing IBD-list into the IBD-pair vector'''
        self.pair_IBD.fill(0)  # Reset IBD-array!
        
        for bpair in self.IBD_blocks:
            ibd_length = bpair[1] / self.rec_rate  # Get length in Morgan
            ind1 = self.start_list.index(bpair[2])
            ind2 = self.start_list.index(bpair[3])       
            self.pair_IBD_add(ind1, ind2, ibd_length)  # Call adding function
                
    def pair_IBD_add(self, i, j, ibd_length):
        '''Add to pair_IBD_list a new IBD-block. If list-entry is empty - create new list. If not - add entry. IBD-lenth is in Centimorgan'''
        j, i = min(i, j), max(i, j)
        
        if self.pair_IBD[i * (i - 1) / 2 + j] == 0:  # In case nothing is there create a new list
            self.pair_IBD[i * (i - 1) / 2 + j] = [ibd_length]
        else:  # In case there is already a block create a new entry
            self.pair_IBD[i * (i - 1) / 2 + j].append(ibd_length)  # Append an IBD-block
                        # self.pair_dist = np.append(self.pair_dist, self.pair_dist[i * (i - 1) / 2 + j])  # Append a new 
                            
    def mean_deme_position(self, position, deme_size):
        '''Return the middle position of the deme under question. Same as used in deme_drawer'''
        return((deme_size * np.around(position / float(deme_size) + 0.001)) % self.gridsize)  # 8-12->10   

    def torus_distance(self, x0, y0, x1, y1):
        # Calculates the Euclidean distance on a Torus
        torus_size = self.gridsize
        dist_x = abs(x0 - x1)
        dist_y = abs(y0 - y1)
        distance = np.sqrt(min(dist_x, torus_size - dist_x) ** 2 + min(dist_y, torus_size - dist_y) ** 2)
        return(distance)
