'''
Created on 17.10.2014
This is the main file, controlling the other parts of the Spatial Block simulator
@author: Harald Ringbauer
'''

from grid import factory_Grid  # Factory method to create Grid Object
from analysis import Analysis
import cProfile  # @UnusedImport
# import cPickle as pickle

growing = 0  # Wether to use growing model or not
        
def main():
    '''Main loop of the Block-Simulator'''
    grid = 0
    data = 0
    print("Welcome back!")
    
    while True:
        print("\nWhat do you want to do?")
        inp = int(input(" (1) Set Blocks \n (2) Update Generation \n (3) Data Analysis \n (4) Reset Grid \n"
        " (5) Full MLE-Analysis\n (6) End program \n"))    
            
        if inp == 1:
            print("Initiating Blocks...")
            if grid == 0:  # Make new grid if not existing
                grid = factory_Grid(growing)
            grid.set_samples()
                
        if inp == 2:     
            if grid == 0:  # Make new grid if not existing
                grid = factory_Grid(growing)               
            grid.plot_distribution()    # Plot the Grid
            t = int(input("\nHow many generations?\n"))
            grid.update_t(t)   
                     
        if inp == 3:
            if grid == 0:
                print("\n Error: No grid to analyze detected, generate one")
                grid = factory_Grid(growing)
            if data == 0:
                print("Load data...")
                data = Analysis(grid)  # Generate Data Analysis Object
            print("\nData loaded: " + str(len(grid.IBD_blocks)) + " IBD Blocks\n")
            
            # Inner loop for this menu
            while True:    
                inp1 = int(input(" (1) Block-Statistics \n (2) MLE-estimation \n (3) Plot exponential decay\n"
                            " (4) Show IBD-List\n (5) Exponential Fit \n (6) Fit specific length \n (7) Exit to main menu\n"))
        
                if inp1 == 1:
                    # print(grid.IBD_blocks)
                    data.which_blocks()
                    data.which_times()
                elif inp1 == 2:
                    data.mle_estimate_error()
                elif inp1 == 3:
                    data.plot_expdecay(logy=False)
                elif inp1 == 4:
                    print(data.IBD_blocks) 
                elif inp1 == 5:
                    data.fit_expdecay()
                elif inp1 == 6:
                    interval1 = input("Interval start: \n")
                    interval2 = input("Interval end: \n")
                    interval = [interval1, interval2]
                    data.fit_specific_length(interval)
                elif inp1 == 7:
                    break
                else: print("Input invalid!")                     
            
        if inp == 4:
            if grid == 0:
                grid = factory_Grid(growing)
            grid.reset_grid()
        
        elif inp == 5:
            if grid == 0:
                print("\n Error: No grid to analyze detected, generate one")
                grid = factory_Grid(growing)
            analysis = grid.create_MLE_object(bin_pairs=True)  # Create the POPRES-MLE-Object
                
            while True:
                inp2 = input("\n(1) Choose MLE-model \n(2) Run Fit\n(3) Bin-plot fitted data \n(4) Log-Likelihood surface"
                            "\n(5) Jack-Knive Countries \n(7) Boots-Trap (Country Pairs) \n(8) Boots-Trap (Blocks)" 
                            "\n(9) Which times? \n(10) Analyze Residuals \n(0) Exit\n")
                if inp2 == 1:
                    inp3 = input("Which Model?\n(1) Constant \n(2) Doomsday"
                    "\n(3) Power growth \n(4) Power-Const\n(0) Back\n")
                    if inp3 == 1: analysis.create_mle_model("constant", grid.chrom_l, [1.0, 2.0])
                    elif inp3 == 2: analysis.create_mle_model("doomsday", grid.chrom_l , [200, 2.0])
                    elif inp3 == 3: analysis.create_mle_model("power_growth", grid.chrom_l, [200, 2.0, 1.0])
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
            
        if inp == 6:
            print("Thanks for your patience future-Harald. See you later alligator!")
            break
    
    pass

def profiling_main():
    '''Short script for profiling where the program spends time.'''
    grid = factory_Grid()
    grid.set_samples()
    grid.update_t(40)
    
if __name__ == '__main__':
    main()
    # cProfile.run('profiling_main()')
