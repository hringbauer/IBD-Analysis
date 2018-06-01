'''
Created on May 31, 2018
Choose your own adventure interface for Boechera Stricta Analysis.
@author: harald
'''

import sys
sys.path.append('../analysis_popres/')
import cPickle as pickle  # @UnusedImport
from load_boechera import LoadBoechera
from mle_multi_run import MLE_analyse   # @UnresolvedImport
import numpy as np

# ## Paths to relevant data fount in load_boechera!!!!'''

print("Hello there, sexy.")

while True:
    inp = input(("\nWhat do you want to do? \n(1) Extract Boechera Data \n(2) Analyze Data" 
    "\n(8) Save/Load data \n(0) Exit \n"))
    
    if inp == 1: 
        min_len = input("What is the minimum block length to load? (in cM)?\n")
        load_obj = LoadBoechera()
        
        # Need the following
        pw_dist, pw_IBD, pw_nr = load_obj.give_lin_block_sharing()
        analysis = MLE_analyse(pw_dist=pw_dist, pw_IBD=pw_IBD, pw_nr=pw_nr, all_chrom=True, error_model=False)
        analysis.gss = load_obj.give_chrom_lens()      
    
    elif inp == 2:
        while True:
            inp1 = input(("\n(1) Regress spec. block sharing \n(2) Regress min. block sharing"
                        "\n(3) Divserse Plots \n(4) MLE estimation "
                        "\n(0) Exit \n"))
            if inp1 == 1:
                min_len = input("What is the minimum block length? (in cM)?\n")
                max_len = input("What is the maximum block length? (in cM)?\n")
                analysis.plot_ibd_spec(min_len=min_len, max_len=max_len)
            elif inp1 == 2:
                min_len = input("What is the minimum block length? (in cM)?\n")
                analysis.plot_ibd_min(min_len, 150)
            
            elif inp1 == 3:  
                while True:
                    inp5=input("\n What Plot do you want to make? \n(0) Geographic Positions "
                    "\n(1) Plot Block Histogram \n(2) Plot IBD-Histogram \n(3) Back\n")
                    if inp5==0:
                        analysis.plot_cartesian_position(barrier=[1500,5000], angle= 0/360.0 * (2*np.pi))
                    elif inp5==1:
                        raise NotImplementedError("Implement this!")
                        analysis.visualize_block_lengths()
                    elif inp5==2:
                        analysis.show_IBD_hist() 
                    elif inp5==3:
                        break
                    else:
                        print("Invalid Input!!")
                
            elif inp1 == 4:
                while True:
                    inp2 = input("\n(1) Choose MLE-model \n(2) Run Fit\n(3) Bin plot fitted data \n(4) Log-Likelihood surface"
                            "\n(5) Jack-Knive Countries \n(7) Boots-Trap (Country Pairs) \n(8) Boots-Trap (Blocks)" 
                            "\n(9) Which times? \n(10) Analyze Residuals \n(11) Plot all fits \n (0) Exit\n")
                    if inp2 == 1:
                        inp3 = input("Which Model?\n(1) Constant \n(2) Doomsday"
                        "\n(3) Power growth \n(4) Power-Const \n(5) Heterogeneous \n(0) Back\n")
                        if inp3 == 1: analysis.create_mle_model("constant")
                        elif inp3 == 2: analysis.create_mle_model("doomsday")
                        elif inp3 == 3: analysis.create_mle_model("power_growth")
                        elif inp3 == 4: analysis.create_mle_model("ddd")
                        elif inp3 == 5: analysis.create_mle_model("hetero")
                        else: print("Invalid Input!! Please do again")
                            # elif inp2 == 4: mle_multi_run.mle_analysis_error("exp_const")
                    elif inp2 == 2: analysis.mle_analysis_error()
                    elif inp2 == 3: analysis.plot_fitted_data_error()    
                    elif inp2 == 4: analysis.plot_loglike_surface() 
                    elif inp2 == 5: analysis.jack_knife_ctries() 
                    elif inp2 == 7: 
                        bts_nr = input("How many boots traps?\n")
                        analysis.boots_trap_ctry(bts_nr)
                    elif inp2 == 8:
                        bts_nr = input("How many boots traps?\n")
                        analysis.boot_trap_blocks(bts_nr)
                    elif inp2 == 9:
                        analysis.which_times()
                    elif inp2 == 10:
                        analysis.calculate_pw_residuals()
                    elif inp2 == 11:
                        analysis.plot_allin_one()
                    elif inp2 == 0: break
                    
            elif inp1 == 9:  # Hidden option
                analysis.show_longest_blocks() 
            elif inp1 == 0:     break  
        
    elif inp == 8:
        inp1 = input("\n(1) Save processed Data \n(2) Load processed Data \n(3) Back\n")
        if inp1 == 1:
            print("Saving...\n")
            pickle.dump(analysis, open(pickle_path, "wb"))  # Pickle the data
        elif inp1 == 2:
            print("Loading...\n")
            analysis = pickle.load(open(pickle_path, "rb"))
            analysis.all_chrom = True  # Use all chromosomes
            print("Countries successfully loaded: ")
            print(analysis.countries)
        elif inp1 == 3:
            continue  # Return to main menue
        
    elif inp == 0: break
print("Sad to see you go!")


