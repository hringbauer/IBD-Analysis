'''
Created on Mar 12, 2015
A class containing methods for quick parent draw.
@author: Harald Ringbauer
'''

import numpy as np
import matplotlib.pyplot as plt
#from position_update_raphael import position_update_raphael


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
        
        else:
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
    barrier_pos = 0  # Where to find the Barrier
    
    def __init__(self, *args):
        DrawParent.__init__(self, *args)
        self.il, self.ir = 0, 0  # Set the indicies to 0
        
    def set_params(self, params):
        '''Sets the Parameters'''
        self.sigma_left = params[0]
        print("Sigma Left set: %.4f" % self.sigma_left)
        self.sigma_right = params[1]
        print("Sigma Right Set: %.4f" % self.sigma_right)
        self.barrier_pos = int(params[2])  # Make it an integer; to be sure
        print("Barrier Position Set: %i" % self.barrier_pos)
        
    def draw_parent(self, mean):
        x0 = mean[0]  # Extract the x-Coordinate
        barrier_pos = self.barrier_pos # load Barrier Position
        
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
        if sigma>1:
            raise ValueError("Sigma MUST be smaller than one for grid model.")
        return sigma ** 2 / 2.0


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
             
         
         
# Tester for reflected drawer        