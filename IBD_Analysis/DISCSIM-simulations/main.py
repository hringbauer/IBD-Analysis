'''
Created on Mar 2, 2015
Main module of the discsim-project.
Everything is run from here.
@author: Harald Ringbauer
'''

import discsim
import ercs
import cPickle as pickle
import numpy as np
from math import sqrt  # @UnusedImport
from timeit import default_timer as timer
from units import Unit_Transformer
from IBD_detection import IBD_Detector
from analysis import Analysis

# Some simulation constants:
grid_size = 160
sample_steps = 4  # Should be even!
u = 1 / (8 * np.pi)
r = sqrt(2) * 2
recombination_rate = 0.001  # Recombination rate (NOT in CM!!)
num_loci = 1500  # Total number of loci
time = 200  # GENERATION TIME
startlist = []
IBD_treshold = 40  # Nr of loci considered IBD

def main():
    '''The heart of the program.'''
    print("I missed you old buddy.")
    trans = Unit_Transformer(grid_size, u, r)
    sim = discsim.Simulator(grid_size)
    startlist = [(i + sample_steps / 2.0, j + sample_steps / 2.0) for i in range(0, grid_size, sample_steps) for j in range(0, grid_size, sample_steps)]
    # startlist = [j for i in zip(startlist, startlist) for j in i]  # Do for diploids
    sim.sample = [None] + startlist
    sim.event_classes = [ercs.DiscEventClass(r, u, rate=grid_size ** 2)]  # Events fall with constant rate per unit area
    sim.recombination_probability = recombination_rate
    sim.num_loci = num_loci
    
    sim.max_population_size = 100000
    
    while True:
        inp = int(input("\nWhat u wanna do bud?! \n (1) Run DISCSIM\n (2) Detect IBD\n (3) Analyze IBD-blocks \n "
                        "(4) Load/Save Data\n (5) MLE-Analysis\n (6) Exit\n"))
    
        if inp == 1:
            start = timer()
            for i in range(1, time):
                print("Doing step: %2.f" % (i))
                sim.run(until=(i))
            end = timer()
            
            print("\nRun time: %.2f s" % (end - start))
            print("Total Generations: %.2f" % trans.to_gen_time(time))
            print("Transformation factor: 1 Time unit is %.3f generations:" % trans.to_gen_time(1.0))
            
            pi, tau = sim.get_history()  # Extract the necessary data
            tau = trans.to_gen_time(np.array(tau))  # Vectorize and measure time in Gen time.
            
        elif inp == 2:  
            chrom_l = num_loci * recombination_rate * 100
            det = IBD_Detector(tau, pi, recombination_rate, grid_size, startlist, IBD_treshold, time, chrom_l)  # Turn on a IBD_Detector
            
            inp1 = int(input("What mode?\n (1) Classic\n (2) Effective Recombination\n (3) Calculate inbreeding fraction\n"))
            if inp1 == 1:   det.IBD_detection()  
            elif inp1 == 2: det.IBD_detection_eff()               
            elif inp1 == 3: 
                det.detect_inbreeding(input("Loop time: "))
                  
            print("Number of IBD-blocks detected %.2f" % len(det.IBD_blocks))
            
        elif inp == 3:
            if det == 0:
                print("\n No IBD anaylsis, please do one")
                continue
                
            data = Analysis(det)
            print("\nData loaded: " + str(len(data.IBD_blocks)) + " IBD Blocks\n")
            print("Sigma: %.3f" % trans.sigma_calculator())
            
            # Inner loop for this menue
            while True:    
                inp1 = int(input(" (1) Block-Statistics \n (2) Generate Histogram \n (3) Plot exponential decay \n " 
                "(4) Show IBD-List \n (5) Exponential Fit \n (6) Exit to main menu \n"))
        
                if inp1 == 1:
                    data.which_blocks()
                    data.which_times()
                elif inp1 == 2:
                    data.IBD_analysis()
                elif inp1 == 3:
                    data.plot_expdecay(logy=False)
                elif inp1 == 4:
                    print(data.IBD_blocks) 
                elif inp1 == 5:
                    data.fit_expdecay()
                elif inp1 == 6:
                    break
                else: print("Input invalid!")
                
        elif inp == 4:
            inp1 = int(input("\nSaving/Loading: \n(1) Save DISCSIM-data \n(2) Load DISCSIM-data "
                    " \n(3) Save IBD-Data \n(4) Load IBD-Data \n(5) Exit\n"))
            
            if inp1 == 1:
                print("SAAAAAVE")
                pickle.dump((pi, tau), open("data.p", "wb"))  # Pickle the data
                
            if inp1 == 2:
                print("LOOOOOOOAD")
                (pi, tau) = pickle.load(open("data.p", "rb"))
                print("LOOOOAAADED: \nLoci loaded: %.2f" % len(tau))
                print("Nodes per locus: %.2f: " % len(tau[0]))
            
            if inp1 == 3:
                print("SAVE IBD-Data...")
                det.delete_history()  # Wont need Pi and Tau anymore
                pickle.dump(det, open("IBD-data.p", "wb"))
            
            if inp1 == 4:
                print("Load IBD-Data...")
                det = pickle.load(open("IBD-data.p", "rb"))
                print("IBD blocks loaded: %.0f" % len(det.IBD_blocks))
                
        elif inp == 5:
            analysis = det.create_MLE_object(bin_pairs=True)  # Create the POPRES-MLE-Object
                 
            while True:
                inp2 = input("\n(1) Choose MLE-model \n(2) Run Fit\n(3) Bin-plot fitted data \n(4) Log-Likelihood surface"
                            "\n(5) Jack-Knive Countries \n(7) Boots-Trap (Country Pairs) \n(8) Boots-Trap (Blocks)" 
                            "\n(9) Which times? \n(10) Analyze Residuals \n(0) Exit\n")
                if inp2 == 1:
                    inp3 = input("Which Model?\n(1) Constant \n(2) Doomsday"
                    "\n(3) Power growth \n(4) Power-Const\n(0) Back\n")
                    if inp3 == 1: analysis.create_mle_model("constant", det.chrom_l, [1.0, 2.0])
                    elif inp3 == 2: analysis.create_mle_model("doomsday", det.chrom_l , [200, 2.0])
                    elif inp3 == 3: analysis.create_mle_model("power_growth", det.chrom_l, [200, 2.0, 1.0])
                    elif inp3 == 4: analysis.create_mle_model("ddd")
                    else: print("Invalid Input!! Please do again")
 
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
                elif inp2 == 0: break
          
        elif inp == 6:
            print("Good bye horses. Im flying over you.")
            break
            
        else: print("U DRUNK")
    
    
if __name__ == '__main__':
    main()
    
