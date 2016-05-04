'''
Created on Mar 12, 2015
A class containing methods for quick parent draw.
@author: Harald Ringbauer
'''

import numpy as np


class DrawParent(object):
    """Class for drawing parents"""
    draw_list_len = 0
    sigma = 0  # What sigma should be
    grid_size = 0
    i = 0
    draw_list = np.array([])
    
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
            
        elif which_drawer == "uniform":
            return UniformDraw(*args)
            
        elif which_drawer == "demes":  # Special Deme mode
            return DemeDraw(self.draw_list_len, self.sigma, self.grid_size)
        
        else:
            raise ValueError('Dispersal mode unknown')
                 
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
             
         
         
         
