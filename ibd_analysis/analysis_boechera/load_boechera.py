'''
Created on May 31, 2018
Class to load Boechera Data. Changes everything to effective cM,
according to f_is. Here the transformation happens!
Interacts with Boechera Data to give the three holy vectors
@author: harald
'''

import pandas as pd
import numpy as np


class LoadBoechera(object):
    '''
    Loads and pre-processes the data.
    '''
    
    # Folders of the Data:
    folder = "./Boechera_Data/"  # The data folder for the data
    ibd_file = "ibd_blocks.csv"
    coordinates_file = "coords.csv"
    
    # Important Parameters for Boechera Analysis
    f_is = 0.0  # The f_is, for adequate length corrections.
    gss_pure = [0, 0, 0, 0, 0, 0]  # Vector of lengths of Chromosomes in MORGAN (!)
    
    # Important Parameters
    coords = 0  #
    df_ibd = 0  # Pandas Dataframe with IBD sharing
    
    debug = False  # Debug Mode
    
    def __init__(self):
        '''
        Constructor
        '''
        print("Initialising LoadBoechera object. F_IS is: %.2f" % self.f_is)
        assert(len(self.gss_pure) == 6)  # Make sure that one has the right number of linkage groups
        
        # Save pre-processed Coordinates to .csv files
        self.df_coords = pd.read_csv(self.folder + self.coordinates_file)
        self.df_ibd = pd.read_csv(self.folder + self.ibd_file)
        
    def filter_nb_valley(self, lat_min=38.94):
        """Filte out neighboring Valley. Everything lower than lat_min is deleted.
        Deletes Individuals from Coordinate List as well as Block Dataframe"""
        inds_main = self.df_coords["Latitude"] > lat_min
        
        print("Deleting %i individuals" % np.sum(~inds_main))
        print(self.df_coords[~inds_main])
        
        # Identify bad labels
        inds_bad = set(self.df_coords[~inds_main]["ID"])  # Extract the set of "deleted" individuals
        # Deletee the bad columns
        self.df_coords = self.df_coords[inds_main]
        
        # Delete from IBD sharing Dataframe
        bad_raws = self.df_ibd[["Ind1", "Ind2"]].isin(inds_bad).any(axis=1)  # Identify bad raws
        print("\nDeleting %i raws from IBD dataframe" % np.sum(bad_raws))
        self.df_ibd = self.df_ibd[~bad_raws]  # Delete them!
        
    def del_rel_pairs(self, cut_off=300):
        """Delete highly related pairs
        cut_off: Where to make the cut!"""
        raise NotImplementedError("Implement this!!")
        
    def give_chrom_lens(self):
        """Gives the effective chromosome lengths"""
        gss_mod = self.gss_pure * (1 - self.f_is)
        return gss_mod
        
    def give_lin_block_sharing(self, min_dist=0, min_len=0, pr=False):
        """Gives the effective block sharing, return the three important vectors:
        pw_IBD, pw_dist, pw_nr"""
        
        # ## First get all the distances of the blocks.
        
        l = len(self.df_coords)  # Nr of individuals in coordinates
        k = int(l * (l - 1) / 2)  # Length of the pairwise lists
        pw_IBD = [[] for _ in range(k)]  # Initialize with empty lists
        
        # Adjust and filter for Block Length:
        self.df_ibd["IBDlen"] = self.df_ibd["IBDlen"] * (1 - self.f_is)  # correct IBD length!!
        self.df_ibd = self.df_ibd[self.df_ibd["IBDlen"] > min_len]
        
        # Create the vector of all pairwise distances and IDs, to quickly look up inds!
        ids = self.df_coords["ID"]
        # Create Dataframe with all pairs of Inds:
        inds = np.array([[i, j] for j in range(0, l) for i in range(0, j)])  # Create list of pw. Indices, i, j in form 01 02 12 03

        coords1 = self.df_coords[["Easting", "Northing"]].iloc[inds[:, 0]].values
        coords2 = self.df_coords[["Easting", "Northing"]].iloc[inds[:, 1]].values
        
        # PW. Distance
        pw_dist = np.linalg.norm(coords1 - coords2, axis=1)
        
        # Send Individuals Labels to their indices
        d = dict(zip(ids, range(len(ids))))
        i1 = self.df_ibd["Ind1"].map(d)
        i2 = self.df_ibd["Ind2"].map(d)
                
        for i1, i2, l in zip(i1, i2, self.df_ibd["IBDlen"]):
            assert(i2 > i1)  # Sanity check
            i = (i2 * (i2 - 1)) / 2 + i1
            pw_IBD[i].append(l)  # Append length of shared IBD block
        
        pw_nr = np.ones(k)  # Initialize to one
        
        # Sanity checks!
        print(k)
        print(len(pw_nr))
        print(len(pw_dist))
        print(len(pw_IBD))
        assert(len(pw_dist) == len(pw_IBD))
        assert(len(pw_dist) == len(pw_nr))
        if self.debug == True:
            print("Pair Dist.:")
            print(pw_dist[:10])
            print("Pair Nr.:")
            print(pw_nr[:10])
            print("Pair IBD:")
            print(pw_IBD[:10])
        
        # Make everything a numpy array
        pw_IBD = np.array(pw_IBD)
        
        return (pw_dist, pw_IBD, pw_nr)
        
