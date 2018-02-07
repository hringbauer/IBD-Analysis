'''
Created on 27.01.2015
The class for the pieces of blocks. Basically just the relevant attributes
@author: hringbauer
'''
from operator import attrgetter

class BlPiece(object):
# Class for block pieces, container for relevant attributes

    origin = ()  # Coordinates of the original individual
    start = 0  # Starting coordinates
    end = 0  # End coordinates
    index = 0  # Index of chromosome
    screwed = False  # Variable which indicate this block will not be followed anymore

    def __init__(self, origin, start, end, index=0):
        self.origin = origin
        self.start = start
        self.end = end
        self.index = index
        
    def update_length(self, start, end):
        self.start = start
        self.end = end
        
class Multi_Bl(BlPiece):
    # Subclass of Block, simply carries blocks of multiple origins and total start and end value
    heals = 0 # Nr of Heals
        
    def __init__(self, sub_blocks, healing=False):
        self.origin = ()  # No origin since only subblocks have that
        self.start = min(i.start for i in sub_blocks)
        self.end = max(i.end for i in sub_blocks)
        self.sub_blocks = []  # IMPORTANT: Since lists are mutable, we have to reset them
        
        
        for block in sub_blocks:  # Subblocks can be Multi-Blocks!
            if isinstance(block, Multi_Bl):
                self.sub_blocks += block.sub_blocks
            else: self.sub_blocks.append(block)
        
        # Heal former Recombination Breakpoints
        self.sub_blocks.sort(key=attrgetter('start'))  # Sort subblocks according to start position
        if healing == True:
            self.rec_heal()
            
    
    def rec_heal(self):
        '''Heal recombination events within the subblocks'''
        subblocks1 = []  # Container for output
        
        for i in range(0, len(self.sub_blocks)):  # Iterate over all pairs of subblocks
            healed = False
            for j in range(i + 1, len(self.sub_blocks)):
                if self.sub_blocks[i].end == self.sub_blocks[j].start and self.sub_blocks[i].origin == self.sub_blocks[j].origin:  # If same subblocks merge again
                    healed = True
                    self.heals +=1 # Increase Heal Counter
                    self.sub_blocks[j].start = self.sub_blocks[i].start  # Extended the second block
            if healed == False:
                subblocks1.append(self.sub_blocks[i])  # Only append subblock when it was not healed
        self.sub_blocks = subblocks1         
    
    
