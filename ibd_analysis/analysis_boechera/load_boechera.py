'''
Created on May 31, 2018
Class to load Boechera Data. Changes everything to effective cM,
according to f_is. Here the transformation happens!
Interacts with Boechera Data to give the three holy vectors.
Later on, this will be a class that generally loads the Data and produces the three holy Vectors
for a given .ibd file as well as a given coordinate file.
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
    ibd_file = "ibd_blocks.csv"  # Pandas Dataframe. Ultimately .csv
    coordinates_file = "coords.csv"  # Pandas Dataframe. Ultimately. csv
    
    # Important Parameters for Boechera Analysis
    f_is = 0.9  # The f_is, for adequate length corrections of IBD blocks as well as chromosomes.
    
    gss_pure = np.array([120.60, 186.52, 115.52, 211.88, 83.34, 97.02])  # Vector of lengths of LGs in MORGAN (!) 46.25 of the 0th LG!!
    
    lat_min = 38.94  # The cutoff of the longitute. Everything smaller than this is deleted
    # Important Parameters
    df_coords = 0  # Pandas Dataframe with Coordinates
    df_ibd = 0  # Pandas Dataframe with IBD sharing
    
    min_d = 200  # Minmum and Maximum Distance (m)
    max_d = 1500  # Maximum Distance (m)
    min_len = 5  # Minimum Length of uncorrected Blocks (cM)
    max_len = 300  # Maximum Length of uncorrected Blocks (cM)
    max_rel = 400  # Related for more than 600 cM
    
    debug = False  # Debug Mode. If yes print more output.
    
    def __init__(self, f_fac):
        '''
        Constructor
        '''
        self.f_is = f_fac
        print("Initialising LoadBoechera object. F_IS is: %.2f" % self.f_is)
        assert(len(self.gss_pure) == 6)  # Make sure that one has the right number of linkage groups
        self.load_data_fresh()  # Load the Data
        
    def load_data_fresh(self):
        """Load the Data"""
        self.df_coords = pd.read_csv(self.folder + self.coordinates_file)
        self.df_ibd = pd.read_csv(self.folder + self.ibd_file)
           
    def filter_nb_valley(self, lat_min=None):
        """Filte out neighboring Valley. Everything lower than lat_min is deleted.
        Deletes Individuals from Coordinate List as well as Block Dataframe"""
        if lat_min == None:
            lat_min = self.lat_min
            
        inds_main = self.df_coords["Latitude"] > self.lat_min
        
        print("\nDeleting %i individuals:" % np.sum(~inds_main))
        print(self.df_coords[~inds_main])

        inds_bad = set(self.df_coords[~inds_main]["ID"])  # Extract the set of "deleted" individuals        
        self.df_coords = self.df_coords[inds_main]  # Delete the bad columns
        
        # Delete from IBD sharing Dataframe
        bad_raws = self.df_ibd[["Ind1", "Ind2"]].isin(inds_bad).any(axis=1)  # Identify bad raws
        print("\nDeleting %i raws from IBD dataframe" % np.sum(bad_raws))
        self.df_ibd = self.df_ibd[~bad_raws]  # Delete them!
        
    def del_rel_pairs(self, pw_IBD, pw_nr=None, max_rel=None):
        """Delete highly related pairs
        cut_off: Sum of individuals IBD sharing, where to make the cutoff.
        Keep in mind units of cut_off!"""
        if max_rel == None:
            max_rel = self.max_rel
        
        if pw_nr == None: 
            pw_nr = np.ones(len(pw_IBD))
            
        bl_sum = list(map(np.sum, pw_IBD))  # Calculate all the lengths
        bl_sum = bl_sum / pw_nr  # Normalize IBD sharing to per pair:
        
        ids = (bl_sum < max_rel)
        print("Kicking out %i pairs related more than %i cM" % (len(ids) - np.sum(ids), max_rel))
        return ids
    
    def min_dist(self, pw_dists, min_d=None, max_d=None):
        """Return boolean array of pairs of pw_dists between min_dist and max_dist"""
        # Load Attributes not given.
        if min_d == None:
            min_d = self.min_d
        if max_d == None:
            max_d = self.max_d
        
        ids_min = (pw_dists > min_d)
        ids_max = np.ones(len(pw_dists))
        if max_d != None:  # Could be still None
            ids_max = (pw_dists < max_d)
        
        ids = ids_min * ids_max  # Where both conditions are fulfilled
        return ids
        
    def give_chrom_lens(self):
        """Gives the effective chromosome lengths"""
        gss_mod = self.gss_pure * (1 - self.f_is)
        return gss_mod
        
    def give_lin_block_sharing(self, min_dist=0, min_len=None, max_len=None, max_rel=None):
        """Gives the effective block sharing, return the three important vectors:
        pw_IBD, pw_dist, pw_nr. Filtering is done Before adjusting for length
        min_dist: Minimum Distance of shared IBD blocks
        max_rel: How much maximum sharing there is for a particular individual pair"""
        
        # ## First get all the distances of the blocks.
        l = len(self.df_coords)  # Nr of individuals in coordinates
        k = int(l * (l - 1) / 2)  # Length of the pairwise lists
        pw_IBD = [[] for _ in range(k)]  # Initialize with empty lists
        
        # Filter and adjust for Block Length. Filtering is done BEFORE adjusting for length
        if min_len == None:
            min_len = self.min_len
        if max_len == None:
            max_len = self.max_len
        
        print("Extracting blocks of right length between %.2f and %.2f" % (min_len, max_len))
        ii = len(self.df_ibd)
        self.df_ibd = self.df_ibd[self.df_ibd["IBDlen"] > min_len]
        self.df_ibd = self.df_ibd[self.df_ibd["IBDlen"] < max_len]
        print("Filtered to %i from %i blocks!" % (len(self.df_ibd), ii))
        
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
            i = (i2 * (i2 - 1)) / 2 + i1  # Instert at the right linearization
            pw_IBD[i].append(l)  # Append length of shared IBD block
        
        pw_nr = np.ones(k)  # Initialize to one
        
        if self.debug == True:
            print("Pair Dist.:")
            print(pw_dist[:10])
            print("Pair Nr.:")
            print(pw_nr[:10])
            print("Pair IBD:")
            print(pw_IBD[:10])
        
        pw_IBD = np.array(pw_IBD)  # Make everything a numpy array
        
        ids_dist = self.min_dist(pw_dist)  # Extract Indices of Individuals with right lengths
        ids_nonrel = self.del_rel_pairs(pw_IBD)  # Extract Indices of non-related Individuals
        ids = ids_dist * ids_nonrel
        
        print("\nTotal pairs: %i" % len(pw_dist))
        print("Pairs within pw. Dist: %i" % np.sum(ids_dist))
        print("Non-Related Indviduals: %i" % np.sum(ids_nonrel))
        print("Both conditions: %i" % np.sum(ids))
        
        pw_dist, pw_IBD, pw_nr = pw_dist[ids], pw_IBD[ids], pw_nr[ids]  # Extract the right indices.
        f = (1 - self.f_is)
        pw_IBD = [[i * f for i in l] for l in pw_IBD]  # Apply the correction factor.
        
        # Sanity checks: Whether all lengths are the same
        assert(len(pw_dist) == len(pw_IBD))
        assert(len(pw_dist) == len(pw_nr))
        print("\nSuccessfully loaded %i pw. comparisons" % len(pw_dist))
        return (pw_dist, pw_IBD, pw_nr)
        
