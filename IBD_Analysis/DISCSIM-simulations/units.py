'''
Created on Mar 4, 2015

@author: Contains class to transform between different time units.
'''

from math import pi
from math import sqrt

class Unit_Transformer(object):
    # Class for transforming units. Initialized with Discsim parameters
    grid_size=0
    u=0
    r=0
    sigma=0
    
    
    def __init__(self, grid_size, u, r):
        # Transforms from DISCSIM model parameters to standard model parameters
        self.grid_size=grid_size    # Total grid_size
        self.u = u  # Impact
        self.r = r  # Radius
        
    def to_gen_time(self,t):
        # Transforms in model time to generation time        
        death_rate=self.r**2 * pi * self.u              # Calculate death rate
        return(t*death_rate)                            # Gives back time in generations
    
    def to_model_time(self,t):
        # Transforms gen time to model time
        death_rate=self.r**2 * pi * self.u
        return(t/death_rate)
        
    def sigma_calculator(self):
        # Calculates expected sigma
        #self.sigma=0.25*self.r**4 * pi                  # Thats per model time
        self.sigma=sqrt(0.5*self.r**2)
        return self.sigma
    
    def give_D(self):
        '''Calculates the Density'''
        return (1/(2*pi*self.u*self.r**2))
        
