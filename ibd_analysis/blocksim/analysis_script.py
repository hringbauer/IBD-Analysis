import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import socket as socket
import os as os
import sys as sys
import itertools as it
import multiprocessing as mp

socket_name = socket.gethostname()
print(socket_name)

if socket_name.startswith("bionc"):
    print("Leipzig Cluster detected!")
    path = "/mnt/archgen/users/hringbauer/git/IBD-Analysis/ibd_analysis/blocksim/" # Update if another cluster / setup!
    sys.path.insert(0, path) # Work in blocksim  Path
    os.chdir(path)  # Set the right Path (in line with Atom default)
print(os.getcwd()) 
print("CPU Count: %d" % mp.cpu_count())

data_folder = '/home/raphael/Recherche - local/IBD_POPRES/SLFV_simulations/'

scenario = 12
run = 0

folder = data_folder + 'scenario_%d/' % scenario
ibd_file = folder + 'IBD_segments_run_%d.csv' % run
loc_file = folder + 'sample_locations_run_%d.csv' % run
G = 25.0 # genome length in Morgan

### Import the MLE Analysis Object
sys.path.append('../analysis_popres/')
from mle_multi_run import MLE_analyse  # @UnresolvedImport

# %config InlineBackend.print_figure_kwargs={'facecolor' : "w"}

df_ibds = pd.read_csv(ibd_file, index_col=0)
df_iids = pd.read_csv(loc_file, index_col=0)
print("Loaded %i IBDs" % (len(df_ibds)))
print("Loaded %i IIDs" % (len(df_iids)))
m = np.max(df_ibds["end"])
print("Highest IBD End Point: %.4f M" % m)

print(df_ibds.head())

def pw_dist(x,y):
    """Return pw. distance between two locations"""
    d = np.sqrt(np.sum((x-y)**2))
    return d

def get_mle_object(df_iids, df_ibds):
    """Create and return MLE object given IID Dataframe and pw. IBD Dataframe.
    Return MLE_analyse object.
    df_iids: Dataframe of indivdiuals, with x and y coordinates
    df_ibds: Dataframe of IBD segments, with length, indivdiual 1 and indivdiual 2.
    Has to match df_iids"""

    positions, inverse, counts = np.unique(df_iids[["x", "y"]].values, axis = 0, 
                                            return_inverse= True, return_counts= True)
    l = len(positions)

    ### Create IBD Vectors
    n_pairs = (l * (l + 1) / 2)  # List of IBD-blocks per pair
    pair_IBD = [[] for _ in range(n_pairs)]  # Initialize with empty lists
    pair_nr = [counts[i] * counts[j] if i != j 
               else counts[i] * (counts[i]-1) / 2
               for i in range(l) for j in range(i+1)]
    assert(np.min(pair_nr) >= 1)  # Sanity Check

    # Get distance Array of all blocks
    pair_dist = [pw_dist(positions[i], positions[j]) for i in range(l) for j in range(i+1)]

    ### Get IBD Segments
    for _, row in df_ibds.iterrows():
        ibd_length = row["length"] * 100  # Get length in centiMorgan
        ind1 = int(row["individual1"])
        ind2 = int(row["individual2"])
        i1, i2 = inverse[ind1], inverse[ind2]
        j, i = min(i1, i2), max(i1, i2)
        pair_IBD[(i * (i + 1)) / 2 + j].append(ibd_length)  # Append an IBD-block  

    assert(len(pair_dist) == len(pair_IBD))  # Sanity Check
    assert(len(pair_dist) == len(pair_nr))

    pair_dist, pair_IBD, pair_nr = np.array(pair_dist), np.array(pair_IBD), np.array(pair_nr)  # Make everything a Numpy array.
    mle_analyze = MLE_analyse(0, pair_dist, pair_IBD, pair_nr, position_list=positions, error_model=False)
    return mle_analyze

def run_mle_analyze(df_iids, df_ibds, model="hetero",
                    start_params=[150, 150, 0.5, 0.5, 0.5], 
                    g=50.0, barrier_pos=0):
    """Load and run Inference for df_idds and df_ibds.
    g: Chromosome Length (in Morgan)"""
    
    mle_analyze = get_mle_object(df_iids, df_ibds) # Create MLE Objec
    
    mle_analyze.create_mle_model_barrier(model=model, g=g, start_param=np.array(start_params), 
                             diploid=False, barrier_pos=barrier_pos,
                             step=0, L=0, mm_mode="symmetric")
    mle_analyze.mle_analysis_error()


run_mle_analyze(df_iids, df_ibds, start_params=[1.0, 1.0, 1.0, 1.0], g=G, barrier_pos=0)