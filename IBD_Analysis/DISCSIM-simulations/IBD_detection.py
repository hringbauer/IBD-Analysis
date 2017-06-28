'''
Created on Mar 5, 2015
@author: Harald Ringbauer
'''

import numpy as np
import matplotlib.pyplot as plt

from analysis import torus_distance
from mle_multi_run import MLE_analyse

class IBD_Detector(object):
    '''Class which analyzes Output of DISC-sim for shared blocks.'''
    
    tau = []
    pi = []
    inds = 0
    info_mat = []
    IBD_treshold = 40  # Number of loci for which IBD_treshold is reached.
    IBD_blocks = []
    gridsize = 0
    start_list = []
    rec_rate = 0.001
    t_ancestral = 500  # What coalescent time is "ancient"
    chrom_l = 0  # Length of the chromosome
    t = 0  # What time is it (back)
    
    def __init__(self, tau, pi, rec_rate, gridsize, start_list, IBD_treshold, t, chrom_l):
        self.tau = tau
        self.pi = pi
        self.inds = len(pi[0]) / 2
        self.gridsize = gridsize
        self.start_list = start_list
        self.rec_rate = 1 / rec_rate
        self.IBD_treshold = IBD_treshold
        self.t = t
        self.chrom_l = chrom_l
        
    def coal_list(self, locus):
        '''Traverses coalescence tree at given locus; gives back 
        coalescence list for every individual''' 
        coal_list = [0] + [[] for i in range(0, self.inds)]  # Empty coalescence list @UnusedVariable
        
        pi = self.pi[locus]  # Load data from locus
        tau = self.tau[locus]
        
        kids = [[]] + [[i] for i in range(1, self.inds + 1)] + [[] for i in range(0, self.inds)]  # List of kids for this node
        
        for i in range(1, len(pi)):
            parent = pi[i]  # Load parent
            if parent == 0: continue  # Node with no parent->stop
            new_childs = kids[i]  # Load children List
            existing_childs = kids[parent]
            t = tau[parent]  # Load coalescence time
            
            # Iterate over all possible Coalescence events:
            for i in new_childs:
                for j in existing_childs:
                    coal_list[min(i, j)].append([max(i, j), t])  # Always add at minimum of the individuals
                    coal_list[max(i, j)].append([min(i, j), t])  # Symmetrie!!
            kids[parent] += new_childs  # Add existing children
            
        return coal_list
        
    def coal_mat_get(self, locus):
        '''Returns Coalescence matrix at given locus'''
        info_mat = np.ones((self.inds, self.inds + 1)) * 10000  # Produces Matrix for coalescence times with originally ancient coalescence
        # Gets info_mat with coalescence infos for given locus
        pi = self.pi[locus]  # Load data from locus
        tau = self.tau[locus]
        kids = [[]] + [[i] for i in range(1, self.inds + 1)] + [[] for i in range(0, self.inds)]  # List of kids for this node
        
        for i in range(1, len(pi)):
            parent = pi[i]  # Load parent
            if parent == 0: continue  # Node with no parent->stop
            new_childs = kids[i]  # Load children List
            existing_childs = kids[parent]
            t = tau[parent]  # Load coalescence time
            
            # Iterate over all possible Coalescence events:
            for i in new_childs:
                for j in existing_childs:
                    info_mat[min(i, j), max(i, j)] = t  # Add coalescence Time; ONLY TO PAIR WITH I<J
            kids[parent] += new_childs  # Add children to node
        return info_mat
    
    def info_mat_init(self):
        self.info_mat = np.zeros((self.inds, self.inds + 1, 2))  # First two entries are indexing individuals, last one: COALESCENCE_TIME, beginning locus index
        self.info_mat[:, :, 0] = 10000
        self.info_mat[:, :, 1] = -1
        self.IBD_blocks = []  # Delete existing IBD_List
        
    def IBD_detection(self):
        '''IBD-Detection Algorithm. Needs tau and pi, sets self.IBD_list'''
        self.info_mat_init()    
        for i in range(0, len(self.pi)):
            self.info_mat_update(i)
        self.info_mat = []  # Delete INFO_mat, not needed anymore 
    
    def IBD_detection_eff(self):
        '''IBD-Detection Algorithm. Needs tau and pi, sets self.IBD_list. Uses effective Recombination'''
        self.info_mat_init()    
        for i in range(0, len(self.pi)):
            self.info_mat_update_eff(i)
        self.info_mat = []  # Delete INFO_mat, not needed anymore 
        
    def info_mat_update(self, locus):
        '''Updates info_mat for given Locus and coalescence matrix (assuming previous ones have been done already)'''
        coal_mat = self.coal_mat_get(locus)  # Get coalescence Matrix
        unequal = coal_mat != self.info_mat[:, :, 0]  # Find indices where there is jump in coalescence time.       
        
        # IBD detection
        IBD_end = unequal & (self.info_mat[:, :, 1] < (locus - self.IBD_treshold)) & (self.info_mat[:, :, 0] != 10000)  # Create Boolean matrix where IBD_blocks end
        IBD_ind = np.nonzero(IBD_end)  # Indices where IBD blocks detected
        start = [self.info_mat[IBD_ind[0][i], IBD_ind[1][i], 1] for i in range(0, len(IBD_ind[0]))]  # Extract start times
        t = [self.info_mat[IBD_ind[0][i], IBD_ind[1][i], 0] for i in range(0, len(IBD_ind[0]))]  # Extract coalesence times (in model time)
        self.IBD_blocks += [(start[i], locus - start[i], self.start_list[IBD_ind[0][i] - 1], self.start_list[IBD_ind[1][i] - 1], t[i]) for i in range(0, len(IBD_ind[0]))]
        
        # Output for the user
        print("\n Doing locus: %.1f" % locus)
        print("New IBD-Blocks found: %.1f" % len(IBD_ind[0]))
            
        # Update where jumps
        indices = np.nonzero(unequal)  # Indices where jumps
        self.info_mat[indices[0], indices[1], 0] = coal_mat[indices]  # New coalescence time 
        self.info_mat[indices[0], indices[1], 1] = locus  # New start locus
    
    def info_mat_update_eff(self, locus):
        '''Updates info_mat for given locus and coalescence matrix, and extracts extended IBD_blocks with effective recombination'''
        coal_mat = self.coal_mat_get(locus)  # Get coalescence Matrix 
        
        # Find indices where effective recombination: Jump in recomb time AND one of the times is ancient
        unequal = (coal_mat != self.info_mat[:, :, 0]) & (np.maximum(coal_mat, self.info_mat[:, :, 0]) > self.t_ancestral) 
        
        # IBD detection
        IBD_end = unequal & (self.info_mat[:, :, 1] < (locus - self.IBD_treshold)) & (self.info_mat[:, :, 0] < 10000)  # Create Boolean matrix where IBD_blocks end
        IBD_ind = np.nonzero(IBD_end)  # Indices where IBD blocks detected
        
        start = [self.info_mat[IBD_ind[0][i], IBD_ind[1][i], 1] for i in range(0, len(IBD_ind[0]))]  # Extract start times
        t = [self.info_mat[IBD_ind[0][i], IBD_ind[1][i], 0] for i in range(0, len(IBD_ind[0]))]  # Extract coalesence times (in model time)
        self.IBD_blocks += [(start[i], locus - start[i], self.start_list[IBD_ind[0][i] - 1], self.start_list[IBD_ind[1][i] - 1], t[i]) for i in range(0, len(IBD_ind[0]))]
        
        # Output for the user
        print("\n Doing locus: %.1f" % locus)
        print("New IBD-Blocks found: %.1f" % len(IBD_ind[0]))
            
        # Update where jumps
        indices = np.nonzero(unequal)  # Indices where jumps
        self.info_mat[indices[0], indices[1], 0] = coal_mat[indices]  # New coalescence time 
        self.info_mat[indices[0], indices[1], 1] = locus  # New start locus
        
    def detect_inbreeding(self, loop_time):
        '''Detect shared long blocks between neighboring individuals, prints fraction of genome that shows short loops.
        Also prints ROH stats. Start list has to consist of neighbours!'''
        treshold_len = input("ROH-treshold length (in loci): ")
        roh_blocks = []
        # Generate list of coalescence times with next individual in start list at all loci
        t_list = np.array([self.get_mrca_t(ind, ind + 1, locus) for locus in range(0, len(self.tau)) for ind in range(1, len(self.tau[0]) / 2, 2)])
        print("Got t_list...")
        t_mat = np.reshape(t_list, (len(self.tau), len(self.tau[0]) / 4))  # Locus x Individual List
        block_starts = np.zeros(len(self.tau[0]) / 4)
        
        for i in range(1, len(self.tau)):  # Iterate over all loci
            jumps = t_mat[i, :] != t_mat[i - 1, :]  # Detect all jumps
            indices = np.nonzero(jumps)  # Extract indices where jump
            
            longjumps = jumps & (block_starts <= (i - treshold_len)) & (t_mat[i - 1, :] < 10000)  # Detect all long jumps (with finite coal. time)
            indices_longjumps = np.nonzero(longjumps)  # Extract their indices
            roh_blocks += list(i - block_starts[indices_longjumps])
            block_starts[indices] = i
        
        print("Detected ROH blocks above %.2f consecutive loci: %.1f" % (treshold_len, len(roh_blocks)))  
        frac_ROH = sum(roh_blocks) / (len(self.tau) * len(self.tau[0]) / 4.0)
        print("Genomic fraction of detected long ROH runs: % .4f" % frac_ROH)
        
        recent_ancestry = (t_list < loop_time)
        inbred_fraction = np.sum(recent_ancestry) / float(len(t_list))
        print("Inbreeding fraction: %.4f:" % inbred_fraction)
        
        # Plot CDF
        plt.figure()
        counts, bin_edges = np.histogram(t_list, bins=1000, range=[0, 200])
        cdf = np.cumsum(counts)
        plt.plot(bin_edges[1:], cdf / float(len(t_list)))
        plt.xlabel('Coalescence Time')
        plt.ylabel('CDF of ancestry before t')
        plt.show()
        
        # Plot Histogram of ROH-blocks
        plt.figure()
        plt.hist(roh_blocks, bins=50)
        plt.xlabel('Block Length')
        plt.ylabel('ROH-blocks')
        plt.show()
        
    def get_mrca_t(self, ind1, ind2, locus):
        '''Calculates time of most recent common ancestor between inds at locus i'''
        anc1 = self.pi[locus][ind1]  # Get ancestor 
        anc2 = self.pi[locus][ind2]  # Get ancestor
        
        while True:
            if anc1 == anc2 or anc1 * anc2 == 0: break 
            # Update the younger locus
            if anc1 > anc2:
                anc2 = self.pi[locus][anc2]
            elif anc1 < anc2:
                anc1 = self.pi[locus][anc1]
        
        if anc1 * anc2 == 0: return 100000  # In case ancient coalescence
        else: return self.tau[locus][anc1]
                         
    def delete_history(self):
        '''Deletes PI and TAU, not needed for IBD mle_multi_run'''
        self.pi = []
        self.tau = []
        
        
#############################################################################
    # Methods to create MLE object#
       
    def create_MLE_object(self, bin_pairs=True):
        '''Return initialized MLE-sharing object'''
        pair_dist, pair_IBD, pair_nr = self.give_lin_IBD(bin_pairs=bin_pairs)  # Get the relevant data
        return MLE_analyse(0, pair_dist, pair_IBD, pair_nr, error_model=False)  # Initialize POPRES-MLE-analysis object
    
    def give_lin_IBD(self, bin_pairs=False):
        '''Method which returns pairwise distance, IBD-sharing and pw. Number.
        Used for full MLE-Method. Require exisiting IBD-list. 
        If bin==True pool same distances. Return arrays'''
        l = len(self.start_list) 
        pair_IBD = np.zeros((l * (l - 1) / 2))  # List of IBD-blocks per pair
        pair_IBD = [[] for _ in pair_IBD]  # Initialize with empty lists
        
        # Iterate over all IBD-blocks
        for bpair in self.IBD_blocks:
            ibd_length = bpair[1] * 100 / self.rec_rate  # Get length in centiMorgan
            ind1 = self.start_list.index(bpair[2])
            ind2 = self.start_list.index(bpair[3])    
            j, i = min(ind1, ind2), max(ind1, ind2) 
            pair_IBD[i * (i - 1) / 2 + j].append(ibd_length)  # Append an IBD-block  
        
        # Get distance Array of all blocks
        pair_dist = [torus_distance(self.start_list[i][0], self.start_list[i][1],
                                    self.start_list[j][0], self.start_list[j][1], self.gridsize) for i in range(0, l) for j in range(0, i)]
        pair_nr = np.ones(len(pair_dist))
        
        if bin_pairs == True:  # Pool data if wanted (speeds up MLE)
            pair_dist, pair_IBD, pair_nr = self.pool_lin_IBD_shr(pair_dist, pair_IBD, pair_nr)
        return (np.array(pair_dist), np.array(pair_IBD), np.array(pair_nr)) 
    
    def pool_lin_IBD_shr(self, pw_dist, pair_IBD, pair_nr):
        '''Bins pairs of same length into one distance pair.
        This does not change the likelihood function but speeds up calculation'''
        distances = sorted(set(pw_dist))  # Produce the keys in a sorted fashion
        
        new_pair_IBD = [[] for _ in distances]  # Initialize the new shortened arrays
        new_pair_nr = [0 for _ in distances]
        
        for j in range(len(distances)):
            r = distances[j]
            for i in range(len(pw_dist)):  # Iterate over all pairs
                if pw_dist[i] == r:  # If Match
                    new_pair_IBD[j] += list(pair_IBD[i])  # Append the shared blocks
                    new_pair_nr[j] += pair_nr[i]  # Add the number of individuals
                    
        print("Nr. of all pairs: %i" % np.sum(new_pair_nr))
        print("Nr of total blocks for analysis: %i" % np.sum([len(i) for i in new_pair_IBD]))
        return(distances, new_pair_IBD, new_pair_nr) 
    
        
                
            
        
                    
            
            
        
        
    
