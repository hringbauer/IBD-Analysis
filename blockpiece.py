'''
Created on 27.01.2015
The class for the pieces of blocks. Basically just the relevant attributes
@author: hringbauer
'''

class BlPiece(object):
# Class for block pieces, container for relevant attributes

    origin = ()  # Coordinates of the original individual
    start = 0  # Starting coordinates
    end = 0  # End coordinates
    index=0 # Index of chromosome
    screwed = False  # Variable which indicate this block will not be followed anymore

    def __init__(self, origin, start, end,index=0):
        self.origin = origin
        self.start = start
        self.end = end
        self.index=index
        
    def update_length(self, start, end):
        self.start = start
        self.end = end
        
class Multi_Bl(BlPiece):
    # Subclass of Block, simply carries blocks of multiple origins and total start and end value
    
    def __init__(self, sub_blocks):
        self.origin = ()  # No origin since only subblocks have that
        self.start = min(i.start for i in sub_blocks)
        self.end = max(i.end for i in sub_blocks)
        self.sub_blocks = []  # IMPORTANT: Since lists are mutable, we have to reset them
        
        for block in sub_blocks:  # Subblocks can be Multi-Blocks!
            if isinstance(block, Multi_Bl):
                self.sub_blocks += block.sub_blocks
            else: self.sub_blocks.append(block)