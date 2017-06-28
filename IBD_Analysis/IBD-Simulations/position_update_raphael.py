'''
Created on Jun 9, 2017
This is the Function to draw the position of the ancestor
for every chromosome.
@author: Harald
'''


def position_update_raphael(old_position, barrier_pos, grid_size, params):
    '''This is the Function with which to update Positions
    Old Position is a vector of length 2: x and y coordinates
    The Ouput has to be of the same form: A list of length 2 of the new Positions.
    *Params is a vector of parameters which are needed for updating.
    They are pre_specified and can be manually changed in the Grid a class.
    Grid_Size: The axial Size of the Grid of Individuals.'''
    sigma_left = params[0]
    sigma_right = params[1]
    x0,y0 = old_position  # Load the oldposition
    
    if x0<barrier_pos:
        sigma=sigma_left
        
    if x0>=barrier_pos:
        sigma=sigma_right
    
    
    
    new_pos = [0,0]  # Update this to your function!
    
    assert 0<=new_pos[0]<grid_size
    assert 0<=new_pos[1]<grid_size
    return new_pos
    
    
    