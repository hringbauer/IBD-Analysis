'''
Script for Array Job
Runs Simulation and Inference
of IBD with Barrier.
'''

#################################
### Imports  

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

path = "/mnt/archgen/users/hringbauer/git/IBD-Analysis/ibd_analysis/blocksim/"
sys.path.insert(0, path) # Work in blocksim  Path
os.chdir(path)  # Set the right Path (in line with Atom default)
print(os.getcwd()) 
print("CPU Count: %d" % mp.cpu_count())

from multi_run_hetero import cluster_run, MultiRunHetero

################################
### Parameters

sigmas = [[0.5, 0.5], [0.5, 0.5],  [0.4, 0.8], [0.4, 0.8]]
nr_inds = [[40, 40], [20, 40], [40, 40], [20, 40]]
betas = [0.0, 0.0, 0.0, 0.0]

position_list = [[100 + i, 100 + j] for i in xrange(-10, 11, 4) for j in xrange(-6, 7, 4)]
sample_size = 10

start_params = [30, 30, 0.5, 0.5]

replicates = 25
folder_out = "/mnt/archgen/users/hringbauer/git/IBD-Analysis/ibd_analysis/blocksim/output/fixNe/"


################################
################################
### Main Code

if __name__ == '__main__':  # Only Run if File directly run.        
    i = int(sys.argv[1]) # Get the Run number
    i = int(i) - 1 # c indexing as qsub cannot run 0
    
    cluster_run(data_set_nr=i, folder_out=folder_out, replicates=replicates, simtype='classic',
                sigmas=sigmas, nr_inds=nr_inds, betas=betas, 
                position_list=position_list, sample_size=sample_size, start_params=start_params)
    
    print("Finished IBD simulation and estimation. Good Job!")