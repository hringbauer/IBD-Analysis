'''
Created on Mar 9, 2016
Contains the machinery to creaty plots
@author: Harald
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.distributions import norm
from mpl_toolkits.basemap import Basemap


def pg_time_dens(t, l0, r, b):
    '''Gives the time density of sharing a block
    l0 Minimum block length (in cM),r (pairwise distance),
    b growth rate'''
    return 0.00154 * t ** b * np.exp(-r ** 2 / (4.0 * t) - t * 2.0 * l0 / 100.0)
 
def pg_time_dens_G(t, l0, r, b, G): 
    '''Gives out the time density of sharing a block of minimum block length l0 (in Morgan),
    r (pairwise distance in sigma), b growth rate (D(t)=Dt^-b and genome length G in Morgan'''
    pre_fac = G / (4 * np.pi * 62.1 ** 2 * 1.78)  # Entered estimates for dispersal and density
    return pre_fac * t ** b * np.exp(-r ** 2 / (4.0 * t) - 2.0 * l0 * t)
    
  
def plot_sharing_times():
    '''Plots sharing times for blocks at various distances.
    Allows inspection of when sharing happens'''
    r = [5, 10, 15] 
    times = np.linspace(1, 120, 240)
    
    f, axarr = plt.subplots(3, 1, sharex=True)  # Create sub-plots
    
    for i in range(3):
        curr = axarr[i] 
        ri = r[i]  
        curr.plot(times, pg_time_dens_G(times, 0.04, ri, 1, 35.374 * 4), 'orange', linewidth=2, label=">4 cM")
        # curr.plot(times, pg_time_dens(times, 4, ri, 1.0), 'orange', linewidth=2, label=">4 cM")    # Original
        curr.plot(times, pg_time_dens_G(times, 0.06, ri, 1, 35.374 * 4), 'goldenrod', linewidth=2, label=">6 cM")  
        curr.plot(times, pg_time_dens_G(times, 0.08, ri, 1, 35.374 * 4), 'red', linewidth=2, label=">8 cM")  
        curr.annotate(r'Pw. Distance: %.0f $\sigma$' % ri, xy=(0.7, 0.15), xycoords='axes fraction', fontsize=15)
        curr.axvline(50, linewidth=1, color="green", label="End of Migration Period?")
        # curr.xlabel("Generations")
        # curr.ylabel("Block-Sharing density per pair")
        # curr.legend()
    axarr[2].set_xlabel("Generations", fontsize=20)
    f.text(0.06, 0.5, 'Shared blocks per pair', ha='center', va='center', rotation='vertical', fontsize=20)
    axarr[0].legend(loc="upper right")
    axarr[0].set_title("Block sharing for 1/T model")
    # plt.tight_layout()
    plt.show()
    
def plot_sharing_times1():
    '''Plots sharing times for blocks at various distances.
    Allows inspection of when sharing happens. All in one picture'''
    r = [5, 10, 15] 
    colors = ['blue', 'magenta', 'red']
    times = np.linspace(1, 100, 240)
    
    plt.figure()  # Create sub-plots
    l0, str = [], []  # @ReservedAssignment
    
    for i in range(3):
        ri = r[i]  
        color = colors[i]
        f1, = plt.plot(times, pg_time_dens_G(times, 0.04, ri, 0, 35.374 * 4), color, linestyle=':', linewidth=2, label=">4 cM")
        # curr.plot(times, pg_time_dens(times, 4, ri, 1.0), 'orange', linewidth=2, label=">4 cM")    # Original
        f2, = plt.plot(times, pg_time_dens_G(times, 0.06, ri, 0, 35.374 * 4), color, linestyle='--', linewidth=2, label=">6 cM")  
        f3, = plt.plot(times, pg_time_dens_G(times, 0.08, ri, 0, 35.374 * 4), color, linestyle='-', linewidth=2, label=">8 cM")  
        # l1=plt.annotate(r'Pw. Distance: %.0f $\sigma$' % ri, xy=(0.7, 0.15), xycoords='axes fraction', fontsize=15)
        l0.append(f3)
        str.append(r'Pw. Distance: %.0f $\sigma$' % ri)
        # curr.xlabel("Generations")
        # curr.ylabel("Block-Sharing density per pair")
        # curr.legend()
    plt.axvline(50, linewidth=1, color="green", label="End of Migration Period?")
    plt.xlabel("Generations", fontsize=24)
    plt.ylabel("Shared blocks per pair", fontsize=24)
    #plt.title("Block sharing for 1/T model")
    plt.title("Block sharing for a constant population density")
    # plt.legend(loc="upper right")
    
    # Add the legends
    leg = plt.legend((f1, f2, f3), (">4 cM", ">6 cM", ">8 cM"), loc="upper right")
    plt.gca().add_artist(leg)  # Add the legend manually to the current Axes.
    plt.legend(l0, str, loc="center right")  # Block Lengths
    # plt.tight_layout()
    plt.show()
    
def demographic_growth():
    '''Plots demographic growth models'''
    x = np.linspace(5, 150, 1000)
    y = np.ones(1000) * 0.05
    y1 = 2 / x
    y2 = 0.3 / (x ** 0.5)
    
    plt.figure()
    plt.plot(x, y, 'r-', linewidth=2, alpha=0.8, label=r"$\mu(t)=C_0$: Constant")
    plt.plot(x, y1, 'g-', linewidth=2, alpha=0.8, label=r"$\mu(t)=\frac{C_1}{t}$: Doomsday")
    plt.plot(x, y2, 'y-', linewidth=2, alpha=0.8, label=r"$\mu(t)=\frac{C_2}{\sqrt{t}}$: Power growth")
    plt.ylabel("Eff. population per area", fontsize=20)
    plt.xlabel("Generations back", fontsize=20)
    leg = plt.legend(loc='best')
    for text in leg.get_texts():
        plt.setp(text, fontsize=18)
    plt.show()
 
def empirical_growth_europe():
    '''Plots empirical estimates of growth in Europe vrs models'''
    t = np.array([1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1850, 1900])
    x = ([36, 44, 58, 79, 60, 81, 100, 120, 180, 265, 390])
    
    t_back = 2000 - t  # To get time back
    t_gen = t_back / 20  # To get mean generation time
    t_array = np.linspace(np.min(t_gen), np.max(t_gen), 1000)
    
    plt.figure()
    plt.plot(t_gen, x, 'ro', label="Historic Estimates")
    plt.plot(t_array, 4000 / t_array ** 1.28, 'g-', label=r"Growth with $\beta=1.28$")
    plt.plot(t_array, 4000 * (0.000597 / 0.0018) / t_array, 'y-', label=r"Doomsday")
    plt.xlim([0, 60])
    plt.xlabel("Generations back", fontsize=20)
    plt.ylabel("Population size", fontsize=20)
    plt.legend()
    plt.show()
    
def plot_powergrowth_vrs_doomsday():
    '''Plots the estimates from Different Growth  models against each other'''
    t = np.linspace(10, 100, 1000)
    plt.figure()
    plt.plot(t , 1.5 / t, label="Doomsday")
    plt.plot(t, 4.5 / (t ** 1.28), label="Power Growth")
    plt.ylabel("Density per km^2")
    plt.xlabel("Generations back")
    plt.legend()
    plt.show()
    
def plot_two_lineages():
    '''Plots the pairwise separation of lineages'''
    x_plot = np.linspace(-20, 20, 100000)
    y_norm = norm.pdf(x_plot - 5, scale=np.sqrt(10))  # 10 Generations back
    y_norm1 = norm.pdf(x_plot + 5, scale=np.sqrt(10))
    
    # y_norm1 = norm.pdf(x_plot, scale=np.sqrt(20))
    y_norm3 = norm.pdf(x_plot - 5, scale=np.sqrt(30))  # 30 Generations back
    y_norm31 = norm.pdf(x_plot + 5, scale=np.sqrt(30))
    
    plt.figure(1)
    plt.subplot(211)
    plt.plot(x_plot, y_norm3, 'r-', linewidth=2)
    plt.plot(x_plot, y_norm31, 'y-', linewidth=2)
    plt.annotate(r'$30$ Generations', xy=(0.01, 0.9), xycoords='axes fraction', fontsize=22)
    plt.axvline(-5, color='k', linestyle='dashed')
    plt.axvline(5, color='k', linestyle='dashed')
    plt.xticks([])
    plt.yticks([])
    plt.ylim([0, 0.15])
    
    plt.subplot(212)
    plt.plot(x_plot, y_norm, 'r-', linewidth=2)
    plt.plot(x_plot, y_norm1, 'y-', linewidth=2)
    plt.annotate(r'$10$ Generations', xy=(0.01, 0.9), xycoords='axes fraction', fontsize=22)
    plt.ylim([0, 0.15])
    plt.axvline(-5, color='k', linestyle='dashed')
    plt.axvline(5, color='k', linestyle='dashed')
    plt.xticks([-5, 5], ["", ""])
    plt.yticks([])
    plt.xlabel("r", fontsize=22)
    # plt.annotate(r'Distance', xy=(0.01, -0.1), xycoords='axes fraction', fontsize=22)
    plt.tight_layout()
    plt.show()
    
    # plt.xlabel(r'$\Delta x$', fontsize=26)
    # plt.ylabel("Probability Density", fontsize=26)
    # plt.annotate(r'$Pr(\Delta x|t)=\frac{1}{\sqrt{2\pi \sigma^2 t}} \exp(-\frac{\Delta x^2}{2\sigma^2 t})$', xy=(0.02, 0.8), xycoords='axes fraction', fontsize=42)
    # plt.legend()
    # plt.title(r'$\Delta x(t)=\frac{1}{\sqrt{2\pi \sigma^2 t}} \exp(-\frac{x^2}{2\sigma^2 t})$')
    # plt.ylim([0, 0.18])
    
def make_testmap():
    # make sure the value of resolution is a lowercase L,
    #  for 'low', not a numeral 1
    
    m = Basemap(projection='merc', llcrnrlat=30, urcrnrlat=65,
                llcrnrlon=5, urcrnrlon=40, resolution='i')
    #m = Basemap(llcrnrlon=-10.5, llcrnrlat=35, urcrnrlon=4., urcrnrlat=44.,
                #resolution='i', projection='merc', lat_0 = 39.5, lon_0 = -3.25)
    
    
    # m.drawcountries(linewidth=0.1,color='w')
    
    # m.drawmapboundary(fill_color='aqua')
    m.drawcoastlines()
    m.drawcountries(linewidth=1, color='k')
    #m.drawcountries()
    m.fillcontinents(color='coral')
    plt.show()
       
if __name__ == '__main__':
    #make_testmap()
    plot_sharing_times1()
    # plot_two_lineages()
    # demographic_growth()
    # empirical_growth_europe()
    # plot_powergrowth_vrs_doomsday()
