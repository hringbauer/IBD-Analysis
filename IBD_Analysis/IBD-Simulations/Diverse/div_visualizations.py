'''
Created on Nov 11, 2016
Class for various visualizations
@author: hringbauer
'''
import numpy as np
from matplotlib import pyplot as plt

def vis_clumped_inds():
    '''Method that visualizes clumped individuals'''
    n = 98
    
    a = np.array([(i, j) for i in range(2, n, 4) for j in range(2, n, 4)])  # Start List for evenly spaced inds 
    
    b2 = np.array([(i, j) for i in range(4, n, 8) for j in range(4, n, 8)])
    b21 = np.array([(i, j + 1) for i in range(4, n, 8) for j in range(4, n, 8)])
    b22 = np.array([(i + 1, j) for i in range(4, n, 8) for j in range(4, n, 8)])
    b23 = np.array([(i + 1, j + 1) for i in range(4, n, 8) for j in range(4, n, 8)])
    b = np.concatenate((b2, b21, b22, b23), axis=0)  # Concatenate the arrays
    
    c = np.array([(i, j) for i in range(6, n, 12) for j in range(6, n, 12)])
    c1 = np.array([(i + 1, j) for i in range(6, n, 12) for j in range(6, n, 12)])
    c2 = np.array([(i + 2, j) for i in range(6, n, 12) for j in range(6, n, 12)])
    c3 = np.array([(i, j + 1) for i in range(6, n, 12) for j in range(6, n, 12)])
    c4 = np.array([(i + 1, j + 1) for i in range(6, n, 12) for j in range(6, n, 12)])
    c5 = np.array([(i + 2, j + 1) for i in range(6, n, 12) for j in range(6, n, 12)])
    c6 = np.array([(i, j + 2) for i in range(6, n, 12) for j in range(6, n, 12)])
    c7 = np.array([(i + 1, j + 2) for i in range(6, n, 12) for j in range(6, n, 12)])
    c8 = np.array([(i + 2, j + 2) for i in range(6, n, 12) for j in range(6, n, 12)])
    c = np.concatenate((c, c1, c2, c3, c4, c5, c6, c7, c8), axis=0)  # Concatenate the arrays
    
    d = np.array([(i, j) for i in range(8, n, 16) for j in range(8, n, 16)])
    d1 = np.array([(i + 1, j) for i in range(8, n, 16) for j in range(8, n, 16)])
    d2 = np.array([(i + 2, j) for i in range(8, n, 16) for j in range(8, n, 16)])
    d3 = np.array([(i + 3, j) for i in range(8, n, 16) for j in range(8, n, 16)])
    d4 = np.array([(i, j + 1) for i in range(8, n, 16) for j in range(8, n, 16)])
    d5 = np.array([(i + 1, j + 1) for i in range(8, n, 16) for j in range(8, n, 16)])
    d6 = np.array([(i + 2 , j + 1) for i in range(8, n, 16) for j in range(8, n, 16)])
    d7 = np.array([(i + 3, j + 1) for i in range(8, n, 16) for j in range(8, n, 16)])
    d8 = np.array([(i, j + 2) for i in range(8, n, 16) for j in range(8, n, 16)])
    d9 = np.array([(i + 1, j + 2) for i in range(8, n, 16) for j in range(8, n, 16)])
    d10 = np.array([(i + 2, j + 2) for i in range(8, n, 16) for j in range(8, n, 16)])
    d11 = np.array([(i + 3, j + 2) for i in range(8, n, 16) for j in range(8, n, 16)])
    d12 = np.array([(i, j + 3) for i in range(8, n, 16) for j in range(8, n, 16)])
    d13 = np.array([(i + 1, j + 3) for i in range(8, n, 16) for j in range(8, n, 16)])
    d14 = np.array([(i + 2, j + 3) for i in range(8, n, 16) for j in range(8, n, 16)])
    d15 = np.array([(i + 3, j + 3) for i in range(8, n, 16) for j in range(8, n, 16)])
    d = np.concatenate((d, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15), axis=0)  # Concatenate the arrays
    
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True, sharey=True)
    ax1.set_title('Sample Distribution')
    ax1.scatter(a[:, 0], a[:, 1], s=30, color="r")
    ax2.scatter(b[:, 0], b[:, 1], s=20, color="g")
    ax3.scatter(c[:, 0], c[:, 1], s=15, color="y")
    ax4.scatter(d[:, 0], d[:, 1], s=10, color="k")
    f.subplots_adjust(wspace=0, hspace=0)
    # f.subplots_adjust(vspace=0)
    # plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
    plt.xlim([0, 96])
    plt.ylim([0, 96])
    plt.show()
    
def vis_range_boundary():
    '''Method that visualizes the sample distribution for range boundaries'''
    a = np.array([(25 + i * 4, 25 + j * 4, 0) for i 
        in range(13) for j in range(13)])  # Introduced this for grant
    
    
    plt.figure()  
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True, sharey=True)
    ax1.set_title(('Sample Distribution'))
    ax1.scatter(a[:, 0], a[:, 1], s=20, color="r")
    p1, p2, p3, p4 = (4, 4), (4, 94), (94, 94), (94, 4)
    ax1.plot(p1, p2, 'k-', linewidth=3)
    ax1.plot(p2, p3, 'k-',linewidth=3)
    ax1.plot(p3, p4, 'k-',linewidth=3)
    ax1.plot(p4, p1, 'k-',linewidth=3)
    ax1.set_title(('Sample Distribution'))
    
    ax2.scatter(a[:, 0], a[:, 1], s=20, color="r")
    p1, p2, p3, p4 = (14, 14), (14, 84), (84, 84), (84, 14)
    ax2.plot(p1, p2, 'k-',linewidth=3)
    ax2.plot(p2, p3, 'k-',linewidth=3)
    ax2.plot(p3, p4, 'k-',linewidth=3)
    ax2.plot(p4, p1, 'k-',linewidth=3)
    
    ax3.scatter(a[:, 0], a[:, 1], s=20, color="r")
    p1, p2, p3, p4 = (23, 23), (23, 75), (75, 75), (75, 23)
    ax3.plot(p1, p2, 'k-',linewidth=3)
    ax3.plot(p2, p3, 'k-',linewidth=3)
    ax3.plot(p3, p4, 'k-',linewidth=3)
    ax3.plot(p4, p1, 'k-',linewidth=3)
    
    ax4.scatter(a[:, 0], a[:, 1], s=20, color="r")
    p1, p2, p3, p4 = (24, 24), (24, 74), (74, 74), (74, 24)
    ax4.plot(p1, p2, 'k-',linewidth=3)
    ax4.plot(p2, p3, 'k-',linewidth=3)
    ax4.plot(p3, p4, 'k-',linewidth=3)
    ax4.plot(p4, p1, 'k-',linewidth=3)
    ax4.text(10, 10, "Reflective Boundary", fontsize=24)
    

    f.subplots_adjust(wspace=0, hspace=0)
    # f.subplots_adjust(vspace=0)
    # plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
    plt.xlim([0, 100])
    plt.ylim([0, 100])
    plt.show()
    
def vis_lim_hab():
    '''Method that visualizes the sample distribution for range boundaries'''
    a = np.array([(2 + i * 4.0, 2 + j * 4.0) for i  
                 in range(15) for j in range(15)])   # Sample Distribution
    p1, p2, p3, p4 = (0, 0), (60, 0), (60, 60), (0, 60) 
    
    plt.figure()  
    plt.scatter(a[:, 0], a[:, 1], s=60, color="r")
    plt.plot(p1, p2, 'k-', linewidth=3)
    plt.plot(p2, p3, 'k-',linewidth=3)
    plt.plot(p3, p4, 'k-',linewidth=3)
    plt.plot(p4, p1, 'k-',linewidth=3)
    plt.title('Sample Distribution',fontsize=20)
    
    # plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
    plt.xlim([-1, 61])
    plt.ylim([-1, 70])
    plt.text(1, 63, r"$\sigma:$", fontsize=32, color = 'g')
    
    plt.plot((10,11), (68,68), 'g-',linewidth=6)
    plt.plot((10,12), (66,66), 'g-',linewidth=6)
    plt.plot((10,15), (64,64), 'g-',linewidth=6)
    plt.plot((10,20), (62,62), 'g-',linewidth=6)
    plt.axes().set_aspect('equal')
    #plt.plot((20,90),(),'g-',linewidth=5)
    #plt.plot((10,92),(),'g-',linewidth=5)
    #plt.plot((10,93),(),'g-',linewidth=5)
    #plt.plot((10,73), (10,10), 'g-',linewidth=5)
    #plt.plot((74,10), (76,10), 'g-',linewidth=5)
    #plt.plot((75,10), (77,10), 'g-',linewidth=5)
    plt.show()
    

#np.random.seed(6)  # Sets the random seed

def draw_samples(x, y, grid_size=100, sigma=5):
    '''Draws samples with Gaussian off-sets.
    Draws Gaussian off-set and then rounds it to nearest integer Value'''
    x_new = np.random.normal(loc=x, scale=sigma) % grid_size
    y_new = np.random.normal(loc=y, scale=sigma) % grid_size
    coords = np.around([x_new, y_new, 0])  # Rounds the Coordinates. And also puts chromosomes on certain Position
    return coords.astype(int)  # Returns the Coordinates as int
    
def draw_center(grid_size=96):
    '''Draws the centers from a grid of size n'''
    x = np.random.randint(0, grid_size)  # Draw the x-Value
    y = np.random.randint(0, grid_size)  # Draw the y-Value
    return (x, y)
    

def draw_sample_list(mean_sample_nr=10, max_samples=500, grid_size=96, sigma=5):
    '''Function that produces spatially correletad samples.
    It draws the means from Poisson - and then a Poisson number of Individuals 
    distributed around it with Gaussian Off-Sets (with STD sigma)'''
    
    samples = []  
    sample_nr = 0   # Sets the sample Number to 0
    
    while sample_nr < max_samples:  # Draw until the wished sample number is reached.
        x, y = draw_center(grid_size=grid_size)
        nr_samples = np.random.geometric(1/float(mean_sample_nr))  # Draws the mean number of samples per cluster
        #print("\nNr of Samples: %i" % nr_samples)
        sample_nr += nr_samples  # Updates the total Nr of Samples
        new_samples = [draw_samples(x, y, grid_size=grid_size, sigma=sigma) for _ in range(nr_samples)]  # Draws the new samples
        samples += new_samples
           
    samples = np.array(samples[:max_samples])  # Reduces to max_sample many individuals
    return samples

def draw_poisson_samples(max_samples, grid_size=96):
    '''Draws Poisson Distributed Random Samples on a spatial Grid'''
    position_list = [[np.random.randint(0, grid_size), np.random.randint(0, grid_size), 0] for _ in range(max_samples)]
    return np.array(position_list).astype(int)
    

def vis_samples_clumping():
    samples = np.array([(i, j) for i in range(2, 96, 4) for j in range(2, 96, 4)])  # Start List for evenly spaced individuals
    samples1 = draw_poisson_samples(max_samples = 500)
    samples2 = draw_sample_list(mean_sample_nr = 5)
    samples3 = draw_sample_list(mean_sample_nr = 50)
    
    
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True, sharey=True)
    ax1.set_title('Sample Distribution')
    ax1.scatter(samples[:, 0], samples[:, 1], s=15, color="r")
    ax2.scatter(samples1[:, 0], samples1[:, 1], s=15, color="g")
    ax3.scatter(samples2[:, 0], samples2[:, 1], s=15, color="y")
    ax4.scatter(samples3[:, 0], samples3[:, 1], s=15, color="k")
    f.subplots_adjust(wspace=0, hspace=0)
    # f.subplots_adjust(vspace=0)
    # plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
    plt.xlim([0, 96])
    plt.ylim([0, 96])
    plt.show()
    
    
# vis_clumped_inds()
# vis_range_boundary()
# vis_lim_hab()
# vis_samples_clumping()
