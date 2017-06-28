'''
This method makes plots how ancestral material spreads out.
@author: Harald
'''

from grid import Grid
from analysis import Analysis
import matplotlib.pyplot as plt


def main():
    '''Main method'''
    grid = Grid()  # Set grid
    grid.reset_grid()  # Delete everything
    
    grid.delete=False
    grid.chrom_l=500
    
    grid.set_chromosome([(30,50,0),(70,50,0)])
    
    data= Analysis(grid)
    (x_list,y_list,colors,size)=data.plot_distribution()    # Extracts Data for plots
      
    grid.update_t(5)  # Updatas plot
    data = Analysis(grid)
    (x_list5,y_list5,colors5,size5)=data.plot_distribution()

    
    grid.update_t(5)
    data = Analysis(grid)
    (x_list10,y_list10,colors10,size10)=data.plot_distribution()
    
    grid.update_t(10)
    data = Analysis(grid)
    (x_list20,y_list20,colors20,size20)=data.plot_distribution()   



    f, axarr = plt.subplots(2, 2,sharey=True,sharex=True)  # @UnusedVariable
    #axarr.set_title("Estimated Dispersal rate")
    axarr[0, 0].scatter(x_list, y_list, c=colors, s=size , alpha=0.5)
    axarr[0, 0].set_title('Generation: 0')
    
    axarr[0, 1].scatter(x_list5, y_list5, c=colors5, s=size5 , alpha=0.5)
    axarr[0, 1].set_title('Generation: 5')
    
    axarr[1, 0].scatter(x_list10, y_list10, c=colors10, s=size10, alpha=0.5)
    axarr[1, 0].set_title('Generation: 10')
    
    axarr[1, 1].scatter(x_list20, y_list20, c=colors20, s=size20 , alpha=0.5)
    axarr[1, 1].set_title('Generation: 20')
    f.text(0.52, 0.04, 'x-Axis', ha='center')
    f.text(0.04, 0.5, 'y-Axis', va='center', rotation='vertical')

    for i in range(0,2): 
        for j in range(0,2):        # Set the axis so that common
            axarr[i,j].set_xlim([0,100])
            axarr[i,j].set_ylim([0,100])

    #plt.delaxes(axarr[2,1])
    # Fine-tune figure; hide x ticks for top plots and y ticks for right plots
    #plt.setp([a.get_xticklabels() for a in axarr[0, :]], visible=False)
    #plt.setp([a.get_yticklabels() for a in axarr[:, 1]], visible=False)
    plt.show()

main()