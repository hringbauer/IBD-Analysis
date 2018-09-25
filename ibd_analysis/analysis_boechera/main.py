'''
Created on May 31, 2018
Choose your own adventure interface for Boechera Stricta Analysis.
@author: harald
'''

import sys
sys.path.append('../analysis_popres/')
import cPickle as pickle  # @UnusedImport
from load_boechera import LoadBoechera
from mle_multi_run import MLE_analyse  # @UnresolvedImport
import numpy as np

# ## Paths to relevant data fount in load_boechera!!!!'''
pickle_path = "./Boechera_Data/popres_blocks.p"  # Where the transformed data is saved to. 3.9 - 12.1
min_b, max_b, bin_w = 1.9, 10.1, 0.1  # On which blocks to do analysis [in cm]
min_b_p, max_b_p = 18, 70  # The Bin widths for the Poisson analysis. [in cm]
#plot_itvs = [[4, 6], [6, 8], [8, 10], [10, 12]]  # Which Intervals to plot in fit
plot_itvs = [[2, 4], [4, 6], [6, 8], [8, 10]]  # Which Intervals to plot in fit
plot_itvs_p = [[20,30], [30,40], [40,50], [50, 60]]
f_fac = 0.9115  # What inbreeding factor to use.

print("Hello there, sexy.")

while True:
    inp = input(("\nWhat do you want to do? \n(1) Extract Boechera Data \n(2) Analyze Data" 
    "\n(8) Save/Load data \n(0) Exit \n"))
    
    if inp == 1: 
        load_obj = LoadBoechera(f_fac)
        load_obj.filter_nb_valley()  # Filter the neighboring valley
        
        # Need the following
        pw_dist, pw_IBD, pw_nr = load_obj.give_lin_block_sharing()
        analysis = MLE_analyse(pw_dist=pw_dist, pw_IBD=pw_IBD, pw_nr=pw_nr, all_chrom=True, error_model=False)
        
        # Print set the right parameters for Multirun
        analysis.gss = load_obj.give_chrom_lens()     
    
    elif inp == 2:
        while True:
            inp1 = input(("\n(1) Regress spec. block sharing \n(2) Regress min. block sharing"
                        "\n(3) Diverse Plots \n(4) Rescale map length \n(5) MLE estimation "
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
                    inp5 = input("\n What Plot do you want to make? \n(0) Geographic Positions "
                    "\n(1) Plot Block Histogram (NI!) \n(2) Plot IBD-Histogram \n(3) Back\n")
                    if inp5 == 0:
                        analysis.plot_cartesian_position(barrier=[1500, 5000], angle=0 / 360.0 * (2 * np.pi))
                    elif inp5 == 1:
                        raise NotImplementedError("Implement this!")
                        analysis.visualize_block_lengths()
                    elif inp5 == 2:
                        analysis.show_IBD_hist() 
                    elif inp5 == 3:
                        break
                    else:
                        print("Invalid Input!!")
            
            elif inp1 == 4:
                print("Doing rescaling with F_IS = %.4f" % f_fac)
                f = 1.0 - f_fac
                
                pw_IBD_t = [[i * f for i in l] for l in pw_IBD]  # Apply the correction factor.
                analysis = MLE_analyse(pw_dist=pw_dist, pw_IBD=pw_IBD_t, pw_nr=pw_nr, all_chrom=True, error_model=False)
                analysis.gss = load_obj.give_chrom_lens(rescaling=True)     # Set the new chromosome lengths
                print("Rescaling complete!")
                
            elif inp1 == 5:
                while True:
                    inp2 = input("\n(1) Choose MLE-model \n(2) Run Fit \n(3) Bin plot fitted data" 
                            "\n(4) Bin plot poisson Model \n(5) Log-Likelihood surface"
                            "\n(7) Jack-Knive Countries \n(8) Boots-Trap (Blocks)" 
                            "\n(9) Which times? \n(10) Analyze Residuals \n(11) Plot all fits \n (0) Exit\n")
                    if inp2 == 1:
                        inp3 = input("Which Model?\n(1) Constant \n(2) Doomsday"
                        "\n(3) Power growth \n(4) Power-Const \n(5) Heterogeneous" 
                        "\n(6) Poisson \n(0) Back\n")
                        if inp3 == 1: analysis.create_mle_model("constant", start_param=[0.0001, 180])
                        elif inp3 == 2: analysis.create_mle_model("doomsday")
                        elif inp3 == 3: analysis.create_mle_model("power_growth", start_param=[1e-5, 263., -0.8])
                        elif inp3 == 4: analysis.create_mle_model("ddd")
                        elif inp3 == 5: analysis.create_mle_model("hetero")
                        elif inp3 == 6: 
                            analysis.create_mle_model("selfing_poisson", start_param=[3e-4, 150.0])
                            min_b, max_b = min_b_p, max_b_p
                        else: print("Invalid Input!! Please do again")
                        
                        print("Bin Width: %.2f" % bin_w)
                        print("Setting MLE Object from: %.1f to %.1f cM" % (min_b, max_b))
                        analysis.mle_object.reset_bins(min_b, max_b, bin_w)
                            # elif inp2 == 4: mle_multi_run.mle_analysis_error("exp_const")
                    elif inp2 == 2: 
                        analysis.mle_analysis_error()
                        analysis.save_res("estimates.csv")  # Save the results
                    elif inp2 == 3: analysis.plot_fitted_data_error(intervals=plot_itvs, f_fac=f_fac)  # Plot bock sharing in intervals
                    elif inp2 == 4: analysis.plot_fitted_data_error(intervals=plot_itvs_p)  # Plot bock sharing in intervals
                    elif inp2 == 5: analysis.plot_loglike_surface() 
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

        elif inp1 == 3:
            continue  # Return to main menue
        
    elif inp == 0: break
print("Sad to see you go!")

