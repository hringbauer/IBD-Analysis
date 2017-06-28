'''
Created on 17.10.2014
This is the main file, controlling the other parts of the Spatial Block simulator
@author: Harald Ringbauer
'''

from selfing.grid_selfing import Grid
from mle_multi_run import Analysis
from time import time
import numpy as np
import matplotlib.pyplot as plt
import cProfile  # @UnusedImport

       
def main():
    '''Main loop of the Block-Simulator'''
    grid = 0
    data = 0
    print("Welcome back!")
    
    while True:
        print("\nWhat do you want to do?")
        inp = int(input(" (1) Set Blocks \n (2) Update Generation \n (3) Data Analysis \n (4) Extract true IBD-blocks \n (5) Reset Grid \n (6) End program \n"))    
            
        if inp == 1:
            print("Initiating Blocks...")
            if grid == 0:  # Make new grid if not existing
                grid = Grid()
            grid.set_samples()
                
        if inp == 2:     
            if grid == 0:  # Make new grid if not existing
                grid = Grid()
                
            data = Analysis(grid)
            data.plot_distribution()
            data = 0
            t = int(input("\nHow many generations?\n"))
            grid.update_t(t)
            # grid.analyze_IBD_mat() 
            grid.conv_IBD_list_to_pair_IBD()
            print("Number of IBD-pairs: %.1f" % np.sum(grid.pair_IBD > 0))
            print("Number of classical IBD-blocks: %.1f" % len(grid.IBD_blocks))
            print("Number of along chromosome IBD-blocks: %.1f" % len(grid.IBD_blocks1))
            
            plt.figure()
            inds = np.where(grid.pair_IBD != 0)[0]
            firstblocks=[grid.pair_IBD[i][0] for i in inds]
            plt.scatter(grid.pair_dist[inds], firstblocks)
            plt.xlabel("Distance")
            plt.ylabel("IBD length")
            plt.show()
                     
        if inp == 3:
            if grid == 0:
                print("\n No grid to analyze detected, generate one")
                grid = Grid()
            if data == 0:
                print("Load data...")
                data = Analysis(grid)  # Generate Data Analysis Object
            print("\nData loaded: " + str(len(grid.IBD_blocks)) + " IBD Blocks\n")
            
            # Inner loop for this menu
            while True:    
                inp1 = int(input(" (1) Block-Statistics \n (2) Generate Histogram \n (3) Switch IBD-lists \n (4) Show IBD-List \n (5) Bessel Fit \n (6) Do MLE\n (7) Plot Bins\n (8) Exit to main menu\n"))
        
                if inp1 == 1:
                    data.which_blocks()
                    data.which_times()
                elif inp1 == 2:
                    data.IBD_analysis()
                elif inp1 == 3:
                    data.IBD_blocks, grid.IBD_blocks1 = grid.IBD_blocks1, data.IBD_blocks
                elif inp1 == 4:
                    print(data.IBD_blocks) 
                elif inp1 == 5:
                    data.fit_expdecay()
                elif inp1 == 6:
                    start = time()
                    data.mle_estimate(grid.pair_IBD, grid.pair_dist)
                    print("Run time: %.4f:" % (time() - start))
                elif inp1 == 7:
                    data.plot_fitted_data()
                elif inp1 == 8:
                    break
                else: print("Input invalid!")
                    
        if inp == 4:
            grid.analyze_IBD_mat()
                
        if inp == 5:
            if grid == 0:
                grid = Grid()
            grid.reset_grid()
            
        if inp == 6:
            print("Thanks for your patience future-Harald. See you later alligator!")
            break
    pass

def profiling_main():
    '''Short script for profiling'''
    grid = Grid()
    grid.set_samples()
    grid.update_t(200)
    
if __name__ == '__main__':
    main()
    # cProfile.run('profiling_main()')
