'''
Created on Mar 12, 2015
A class containing methods for quick parent draw.
@author: Harald Ringbauer
'''

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../analysis_popres/')
from analysis_popres.hetero_sharing import migration_matrix
from scipy.sparse import find
from time import time
# from hetero_sharing import migration_matrix
# from position_update_raphael import position_update_raphael


class DrawParent(object):
    """Class for drawing parents. Subclasses with more specific drawers inherit from it"""
    draw_list_len = 0
    sigma = 0  # What sigma should be
    grid_size = 0
    i = 0
    draw_list = np.array([])
    params = []
    
    def __init__(self, draw_list_len, sigma, grid_size):
        self.draw_list_len = draw_list_len
        self.sigma = sigma
        self.grid_size = grid_size
        self.i = 0  # Sets counter to 0
        self.draw_list = self.generate_draw_list()
        
    def draw_parent(self, mean):
        '''Get parent via offset from offset list'''
        delta = self.draw_parental_offset()  # Get next available element from parental offset_list
        new_pos = (mean + delta) % self.grid_size  # Do the torus correction
        return(new_pos)
        
    def generate_draw_list(self):
        pass
    
    def draw_parental_offset(self):
        # Returns Parental offspring from list
        self.i += 1
        if self.i >= self.draw_list_len:
            self.draw_list = self.generate_draw_list()
            self.i = 0
        return self.draw_list[self.i, :]
            
    def choose_drawer(self, which_drawer):
        '''Return wished drawer object'''
        args = (self.draw_list_len, self.sigma, self.grid_size)
        
        if which_drawer == "normal":
            return NormalDraw(*args)
            
        elif which_drawer == "laplace":
            return LaplaceDraw(*args)
        
        elif which_drawer == "laplace_refl":
            return LaplaceDrawRefl(*args)
            
        elif which_drawer == "uniform":
            return UniformDraw(*args)
            
        elif which_drawer == "demes":  # Special Deme mode
            return DemeDraw(self.draw_list_len, self.sigma, self.grid_size)
        
        elif which_drawer == "raphael":
            return RaphaelDraw(*args)
        
        elif which_drawer == "mig_mat":
            return HeterogeneousDraw(*args)
        
        else:
            print(which_drawer)
            raise ValueError('Dispersal mode unknown')
        
    def set_params(self, *args):
        '''Method to set local Parameters. In all subclasses is overwritten'''
        
        pass
                 
# Bunch of drawers which inherit from DrawParent           
class NormalDraw(DrawParent):
    '''For normal drawing'''
    
    def generate_draw_list(self):
        draw_list = np.around(np.random.normal(scale=self.sigma, size=(self.draw_list_len, 2)))
        return(draw_list.astype(int))
 
 
class UniformDraw(DrawParent):
    '''For uniform drawing'''
    half_length = 0
    
    def __init__(self, *args):
        self.half_length = args[1] * np.sqrt(3)
        DrawParent.__init__(self, *args)
        
    def generate_draw_list(self):
        draw_list = np.around(np.random.uniform(low=-self.half_length, high=self.half_length, size=(self.draw_list_len, 2)))
        return(draw_list.astype(int))


class LaplaceDraw(DrawParent):
    '''For Laplace drawing'''
    scale = 0
    
    def __init__(self, *args):
        self.scale = args[1] / np.sqrt(2)
        DrawParent.__init__(self, *args)

    
    def generate_draw_list(self):
        draw_list = np.around(np.random.laplace(scale=self.scale, size=(self.draw_list_len, 2)))
        return(draw_list.astype(int))
    
class LaplaceDrawRefl(DrawParent):
    '''For Laplace Drawing where lineages get reflected'''
    scale = 0
    
    def __init__(self, *args):
        self.scale = args[1] / np.sqrt(2)
        DrawParent.__init__(self, *args)
    
    def generate_draw_list(self):
        draw_list = np.around(np.random.laplace(scale=self.scale, size=(self.draw_list_len, 2)))
        return(draw_list.astype(int))
    
    def draw_parent(self, mean):  # Also overwrite the draw_parent function - to account for reflections
        '''Get parent via offset from offset list'''
        delta = self.draw_parental_offset()  # Get next available element from parental offset_list
        
        new_pos = (mean + delta)    
        
        while True:
            if new_pos[0] >= self.grid_size:  # For axis 0
                new_pos[0] = ((self.grid_size - 1) - (new_pos[0] - self.grid_size))
                
            elif new_pos[0] < 0:
                new_pos[0] = -new_pos[0]  # Reflection at 0
            
            
            if new_pos[1] >= self.grid_size:  # For axis 1
                new_pos[1] = ((self.grid_size - 1) - (new_pos[1] - self.grid_size))
                
                
            elif new_pos[1] < 0:
                new_pos[1] = -new_pos[1]  # Reflection at 0
        
            if (0 <= new_pos[0] < self.grid_size) and (0 <= new_pos[1] < self.grid_size):  # Continue if more reflections needed
                return new_pos
        
    
        
class DemeDraw(DrawParent):
    p = 0  # Probability of movement
    dis_prop = 0
    deme_size = 5
    steps = np.array([-1, 0, 1])  # For Deme Model: Steps
    
    def __init__(self, *args):
        self.steps = self.deme_size * self.steps
        self.dis_prop = (args[1] ** 2) / (2.0 * self.deme_size ** 2)  # Caluculate Dispersal Probability for Deme-Model
        self.p = np.array([self.dis_prop, 1 - 2 * self.dis_prop, self.dis_prop])
        DrawParent.__init__(self, *args)
    
    def draw_parent(self, mean):
        '''Get parent via offset from offset list. Here in deme model also update parental position to middle of theme'''
        mean = self.deme_size * np.around(np.array(mean) / float(self.deme_size) + 0.001)  # Set point to deme middle (8-12>10)
        delta = self.draw_parental_offset()  # Get next available element from parental offset_list
        new_pos = (mean + delta) % self.grid_size  # Do the torus correction
        return(new_pos)
                   
    def generate_draw_list(self):
        draw_list = np.random.choice(self.steps, p=self.p, size=(self.draw_list_len, 2))  # First do the deme offset
        draw_list += np.random.choice(self.deme_size, size=(self.draw_list_len, 2)) - self.deme_size / 2  # Then the fine scale offset within next deme
        return(draw_list.astype(int))     
    


class RaphaelDraw(DrawParent):
    '''Class that updates Positions according to Raphaels Code
    params[0]: Sigma left. params[1]: Sigma Right. params[2]: Barriers position'''
    il, ir, ic = 0, 0, 0  # Indices of the List from which to draw  
    ol_list, or_list, oc_list = [], [], []  # List of all the Offsets on the Left and on the Right   
    sigma_left = 0.5
    sigma_right = 0.5
    nr_inds_left = 5
    nr_inds_right = 5
    barrier_pos = 0  # Where to find the Barrier
    
    def __init__(self, *args):
        DrawParent.__init__(self, *args)
        self.il, self.ir = 0, 0  # Set the indicies to 0
    
    def init_manual(self, drawlist_length, sigmas, nr_inds, gridsize):
        ''' Could do something'''
        pass
    
       
    def set_params(self, sigmas, nr_inds, barrier):
        '''Sets the Parameters'''
        self.sigma_left = sigmas[0]
        print("Sigma Left set: %.4f" % self.sigma_left)
        self.sigma_right = sigmas[1]
        print("Sigma Right Set: %.4f" % self.sigma_right)
        self.barrier_pos = int(barrier)  # Make it an integer; to be sure
        print("Barrier Position Set: %i" % self.barrier_pos)
        self.nr_inds_left = nr_inds[0]
        print("Nr. Individuals left: %.4f" % self.nr_inds_left)
        self.nr_inds_right = nr_inds[1]
        print("Nr. Individuals right: %.4f" % self.nr_inds_right)
    
    
    def draw_parent(self, mean):
        x0 = mean[0]  # Extract the x-Coordinate
        barrier_pos = self.barrier_pos  # load Barrier Position
        
        # Draw Depending on which side of the barrier
        # Do the left hand side:
        if x0 < barrier_pos:
            os = self.give_offset_left()
                
        # Do the right hand side:
        elif x0 > barrier_pos:
            os = self.give_offset_right()

        # if (x0>barrier_pos) and (new_pos[0]<barrier_pos) or (x0<barrier_pos) and (new_pos[0]>barrier_pos):
        elif x0 == barrier_pos:
            os = self.give_offset_center()
            
        else: 
            raise ValueError("You are a stupid Moron. You broke the program")
        
        new_pos = (mean + os) % self.grid_size  # Add offset and take Grid Boundaries into account
        return new_pos
    
    def give_offset_left(self):
        '''Gives the Offset on the left.
        Cycle through corresponding List'''
        if self.il >= len(self.ol_list):
            self.ol_list = self.offsets(self.sigma_left)
            self.il = 0  # Reset Index
        os = self.ol_list[self.il]
        self.il += 1
        return os
        
        
    def give_offset_right(self):
        '''Gives the Offset on the right.
        Cycle through corresponding List'''
        if self.ir >= len(self.or_list):
            self.or_list = self.offsets(self.sigma_right)
            self.ir = 0
        os = self.or_list[self.ir]
        self.ir += 1
        return os
    
    def give_offset_center(self):
        '''Gives the Offset if in the center.
        Cycle through corresponding List'''
        if self.ic >= len(self.oc_list):
            self.oc_list = self.offsets_center(self.sigma_left, self.sigma_right)
            self.ic = 0
        os = self.or_list[self.ic]
        self.ic += 1
        return os
        

    def offsets(self, sigma):
        '''Draw List of all Offsets'''
        p = self.sigma_to_p(sigma)
        rand_nrs = np.random.random((self.draw_list_len, 2))  # Draws a lot of random Nbrs between 0 and 1
        off_sets = np.zeros((self.draw_list_len, 2)).astype("int")
        off_sets[rand_nrs > 1 - p] = 1
        off_sets[rand_nrs < p] = -1
        return off_sets
    
    def offsets_center(self, sigma_left, sigma_right):
        '''Draw List of Center Offsets'''
        p_l = self.sigma_to_p(sigma_left)  # The Probability of going to the left.  
        p_r = self.sigma_to_p(sigma_right)  # The probability of going to the right.
        p_u = (p_l + p_r) / 2.0  # The probability of going up or down.
        
        rand_nrs = np.random.random((self.draw_list_len, 2))  # Draws a lot of random Nbrs between 0 and 1
        off_sets = np.zeros((self.draw_list_len, 2)).astype("int")
        
        # Enter Vertical Offsets
        off_sets[rand_nrs[:, 1] > 1 - p_u, 1] = 1
        off_sets[rand_nrs[:, 1] < p_u, 1] = -1
        
        # Enter Horizontal Offsets:
        off_sets[rand_nrs[:, 0] > 1 - p_r, 1] = 1
        off_sets[rand_nrs[:, 0] < p_l, 1] = -1
        return off_sets
        
    def sigma_to_p(self, sigma):
        '''Converts Sigma to Probability to go into neighboring deme'''
        return sigma ** 2 / 2.0


class HeterogeneousDraw(DrawParent):
    '''
    Updates positions according to isotropic migration model from Nagylaki
    barrier position is always L/2
    '''
    pop_sizes = []
    sigma = []
    barrier_pos = 0
    Migration_matrix = []
    cum_sums = []
    jump_inds = []
    
    def __init__(self, *args):
        '''Initializes Parent.'''
        pass  # Requiress manual Initializiation with init_manual()!!        
        
    
    def init_manual(self, draw_list_len, sigmas, pop_sizes, grid_size):
        '''Hack: Initializes Manually'''
        self.draw_list_len = draw_list_len
        self.sigma = sigmas
        assert(grid_size % 2 == 0)
        self.grid_size = grid_size
        self.i = 0  # Sets counter to 0
        self.draw_list = self.generate_draw_list()  # Generates Draw List
        self.pop_sizes = np.maximum(pop_sizes, [0, 0])
        # DrawParent.__init__(self, draw_list_len, sigmas, grid_size + grid_size%2)
        self.Migration_matrix = migration_matrix(self.grid_size, self.sigma ** 2, self.pop_sizes)
        self.pre_calculate_cum_sums()   # Precalculates the cumulative sums.
    
    def pre_calculate_cum_sums(self):
        '''Pre-Calculates all cumulative Sums. Stores unique Values in suitable format.
        Is best suited for a very sparse Migration Matrix!!'''
        tic = time()
        # Step 1: Pre-Calculate sparse Matrix:
        l = np.shape(self.Migration_matrix)[0]
        
        cum_sums = [[] for _ in range(l)]  # The values of the cum_sums
        jump_coords = [[] for _ in range(l)]  # The indices of the jumps
        
        print("Doing Pre-Calculations for cumulative Sums")
        # range(l)
        for i in range(l):
            row = self.Migration_matrix[:, i]  # Get the ith row of the Migration Matrix
            coords, _, vals = find(row) # Extracts indices and values of the migration matrix
            cumsum = np.cumsum(vals)  # Calculates the cumulative Sum

            # Set the Values:
            jump_coords[i] = coords
            cum_sums[i] = cumsum  # Sets the cumulative Sum
        
        # Step 2: Keep only unique Lists. To be implemented
        self.cum_sums = cum_sums
        self.jump_coords = jump_coords
        
        toc = time()
        print("Time taken for pre-calculating Cum-Sums: %.4f" % (toc-tic))
        print("Nr of total entries in precalculated cum-sums:")
        print(np.sum([len(raw) for raw in self.cum_sums]))
        
    def generate_draw_list(self):
        ''' Generates a list of seeds, ie random numbers between 0 and 1. '''
        return np.random.random(self.draw_list_len)
    
    def draw_parent(self, current):
#         current = current[0] + self.grid_size * current[1]  # convert to x + L* y coordinates
#         cumulative_density = np.cumsum(self.Migration_matrix[:, current].todense())
#         seed = self.draw_seed()
#         new = np.argmax(cumulative_density > seed)
#         parent = np.array([new % self.grid_size, np.int(np.floor(new / self.grid_size))])
        
        current = current[0] + self.grid_size * current[1]  # convert to x + L* y coordinates
        cum_sum = self.cum_sums[current] # Get the right cum_sum
        seed = self.draw_seed()
        ind = np.argmax(cum_sum > seed) # Get the first index bigger than the seed
        new = self.jump_coords[current][ind] # Get the matching Jump Target
        parent = np.array([new % self.grid_size, np.int(np.floor(new / self.grid_size))]) # Get the Parental Position
        return parent
        
        # print("Delete This")
        # print("Shape of Migration Matrix:")
        # print(np.shape(self.Migration_matrix[:,current].todense()))
        # print(cumulative_density)
        # we look for the first position for which the cumulative density becomes greater than the seed
        # convert back to (x, y) coordinates
    
    def set_params(self, sigmas, nr_inds, barrier):
        '''Sets Parameters.'''
        self.sigmas = sigmas
        self.pop_sizes = nr_inds
        self.barrier_pos = barrier
    
    def draw_seed(self):
        # Returns seed from list
        self.i += 1
        if self.i >= self.draw_list_len:
            self.draw_list = self.generate_draw_list()
            self.i = 0
        return self.draw_list[self.i]


def tester_for_refl(grid_size=10):
    '''This is a quick tester for reflected lineages - to see whether it behaves as it should''' 
    drawer = DrawParent(10000, 5, 40)  # Generate Drawer object
    drawer = drawer.choose_drawer("laplace_refl")
    n = 1000
    
    x_list = [0 for _ in range(n)]
    y_list = [0 for _ in range(n)]
    x, y = 5, 5
    
    for i in range(1, 1000):
        print(x)
        print(y)
        x, y = drawer.draw_parent((x, y))
        x_list[i] = x
        y_list[i] = y
    
    plt.figure()
    x_values = [i for i in range(0, n)]
    print("Maximum: ")
    print(max(x_list))
    print("Minimum: ")
    print(min(x_list))
    plt.plot(x_values, x_list)
    plt.show()

# tester_for_refl()  
             
         
# Some Code for testing Purposes        
# Tester for Heterogeneous Draw:
#if __name__ == "__main__":
#     print("Initializing Test")
#     drawer = HeterogeneousDraw()
#     drawer.init_manual(draw_list_len=1000, sigmas=np.array([0.5, 0.5]), pop_sizes=np.array([5, 5]), grid_size=100)
#     mig_mat_list = drawer.Migration_matrix[:, 5]
#     print(np.shape(mig_mat_list))
#     print(mig_mat_list)
#     res = find(mig_mat_list)
#     x_coord, y_coord, vals = find(mig_mat_list)
#     print(vals)
#     print(np.cumsum(vals))
    # print(mig_mat_list[0])
    # print(drawer.Migration_matrix[:,5])
    # print("Calculated Migration Matrix")
    # print(np.shape(drawer.Migration_matrix))
    # print(drawer.Migration_matrix[:,2])
    
    # cumulative_density = np.cumsum(drawer.Migration_matrix[:, 2].todense())
    # print(np.shape(cumulative_density))
    # print(cumulative_density[:5])
    
    # drawer.pre_calculate_cum_sums()
def test_heterogeneous_draw():
    '''Tester for heterogeneous_draw'''
    drawer = HeterogeneousDraw()
    drawer.init_manual(draw_list_len=1000, sigmas=np.array([0.1, 0.5]), pop_sizes=np.array([5, 5]), grid_size=100)
    pos=[150,20]
    parents = [drawer.draw_parent([60, 20]) for _ in range(10000)]
    x_off_sets = [(parent[0]-pos[0]) for parent in parents]
    y_off_sets = [(parent[1]-pos[1]) for parent in parents]
    #print(np.corrcoef([x_off_sets, y_off_sets]))
    print("STD x-Axis: %.4f" % np.std(x_off_sets))
    print("STD y-Axis: %.4f" % np.std(y_off_sets))
    print("Test finished.")
    
    
# Tester for Heterogeneous Draw. Draws randomly 10000 offsets; and checks whether axial sigma is right:
if __name__ == "__main__":
    test_heterogeneous_draw()
    
    
    
      
