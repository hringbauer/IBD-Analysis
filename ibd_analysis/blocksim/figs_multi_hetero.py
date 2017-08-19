'''
Created on July 5th, 2017
Contains Methods to load and visualize the puptputs of multi_run_hetero
@author: Harald Ringbauer
'''


# from scipy.special import kv as kv  # Import Bessel functions of second kind

import numpy as np
import matplotlib.pyplot as plt
import cPickle as pickle
import os

# Contains various Methods to analyze the Runs of Multirun Hetero!

def load_estimates(data_folder="./testfolder",data_set_nr=1, scenario=0):
    '''Method to load and return estimates.'''
    full_path = data_folder + "/scenario" + str(scenario) + "/data_set_nr_" + str(data_set_nr).zfill(2) + ".csv"
    # If data non-existent return nothing:
    if not os.path.exists(full_path):
        return ([],[])
    data=np.loadtxt(full_path, delimiter='$').astype('float64')
    params, ci_s = data[:,0], data[:,1:]
    return (params, ci_s)
    
def load_estimates_range(data_folder="./testfolder",data_set_vec=[1], scenario=0, filter=True):
    '''Load Estimates for a Range of Datasets'''
    params=np.array([load_estimates(data_folder = data_folder, scenario=scenario, data_set_nr=i)[0] for i in data_set_vec])
    cis=np.array([load_estimates(data_folder = data_folder, scenario=scenario, data_set_nr=i)[1] for i in data_set_vec])
    
    if filter == True:
    # Maybe check for missig data HERE
        lengths = np.array(map(len, params))
        inds = ~(lengths==0)
        
        params=params[inds]
        cis=cis[inds]
    
    return params, cis
        
#def plot_estimates(data_folder="./testfolder", scenario=0, data_set_vec=range(6)):
#    '''Plot multiple Estimates from a given scenario'''
    
    
    

    
  
  
    
 
# Some testing:

if __name__ == "__main__":
    #params, ci_s = load_estimates(data_set_nr=5, scenario=0)   
    #print(params)
    #print(ci_s)   
    ps,cis=load_estimates_range(data_folder="./testfolder", scenario=0, data_set_vec=range(6))
    print(ps)
    print("Testin Successful!")
