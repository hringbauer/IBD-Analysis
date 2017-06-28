'''
Created on 27.01.2015
Area for testing shit and such
@author: hringbauer
'''
import numpy as np

from math import sqrt
from blockpiece import BlPiece
from random import shuffle
import matplotlib.pyplot as plt
import bisect
import itertools 
from collections import Counter
from random import randint
from scipy.misc import factorial  # @UnresolvedImport
from scipy.optimize import curve_fit
from scipy.stats import gaussian_kde
from scipy.special import kv as kv
from scipy.stats import binned_statistic
from scipy.stats.distributions import norm, laplace, uniform


# sigma=0.965
sigma = 0.965  # 1.98 #0.965
scale = sigma / np.sqrt(2)
half_length = sigma * np.sqrt(3)


p = 0  # Probability of movement
deme_size = 5
steps = np.array([-1, 0, 1])  # For Deme Model: Steps
    
steps = [-deme_size, 0, deme_size]
dis_prop = (sigma ** 2) / (2.0 * deme_size ** 2)  # Caluculate Dispersal Probability for Deme-Model
p = np.array([dis_prop, 1 - 2 * dis_prop, dis_prop])
# draw_list = np.random.choice(steps, p=p, size=50000000)  # First do the deme offset
# draw_list = np.around(np.random.normal(scale=sigma, size=10000000))
# draw_list = np.around(np.random.uniform(low=-half_length, high=half_length, size=50000000))
draw_list = np.around(np.random.laplace(scale=scale, size=5000000))


print("Mean: %.2f" % np.mean(draw_list))
print("Std: %.4f" % np.std(draw_list))

# Now plot different dispersal kernels:
x_plot = np.linspace(-10, 10, 100000)
y_norm = norm.pdf(x_plot, scale=2)
y_laplace = laplace.pdf(x_plot, scale=scale)
y_uniform = uniform.pdf(x_plot, scale=2 * half_length, loc=-half_length)
    
plt.figure()
plt.plot(x_plot, y_laplace, label="Laplace: 3", linewidth=3)
plt.plot(x_plot, y_norm, label="Normal: 0", linewidth=3)
plt.plot(x_plot, y_uniform, label="Uniform: -1.2", linewidth=3, color='y')
plt.ylabel("Probability Density", fontsize=25)
plt.legend(prop={'size':25})
plt.tick_params(axis='x', labelsize=15)
plt.tick_params(axis='y', labelsize=15)
plt.show()

plt.figure()
x_plot = np.linspace(-14, 14, 100000)
y_norm = norm.pdf(x_plot, scale=np.sqrt(10))
y_norm1 = norm.pdf(x_plot, scale=np.sqrt(20))
y_norm2 = norm.pdf(x_plot, scale=np.sqrt(30))
plt.plot(x_plot, y_norm, linewidth=3, color='yellow', label="10 gens")
plt.plot(x_plot, y_norm1, linewidth=3, color='orange', label="20 gens")
plt.plot(x_plot, y_norm2, linewidth=3, color='red', label="30 gens")
plt.xlabel(r'$\Delta x$', fontsize=26)
plt.ylabel("Probability Density", fontsize=26)
# plt.annotate(r'$Pr(\Delta x|t)=\frac{1}{\sqrt{2\pi \sigma^2 t}} \exp(-\frac{\Delta x^2}{2\sigma^2 t})$', xy=(0.02, 0.8), xycoords='axes fraction', fontsize=42)
plt.annotate(r'$Pr(\Delta x|t)\approx N(0,\sigma^2 t)$', xy=(0.1, 0.8), xycoords='axes fraction', fontsize=42)
plt.legend()
# plt.title(r'$\Delta x(t)=\frac{1}{\sqrt{2\pi \sigma^2 t}} \exp(-\frac{x^2}{2\sigma^2 t})$')
plt.ylim([0, 0.18])
plt.show()

# Figure for Bessel-Decay:
plt.figure()
x_plot = np.linspace(0.000001, 30, 100000)
y_plot = x_plot * kv(1, np.sqrt(0.1) * x_plot)
y_plot1 = x_plot * kv(1, np.sqrt(0.06) * x_plot)
y_plot2 = x_plot * kv(1, np.sqrt(0.14) * x_plot)
plt.semilogy(x_plot, y_plot1, linewidth=4, color='blue', label=r'$l=3$ cM')
plt.semilogy(x_plot, y_plot, linewidth=4, color='green', label=r'$l=5$ cM')
plt.semilogy(x_plot, y_plot2, linewidth=4, color='red', label=r'$l=7$ cM')
plt.tick_params(axis='x', labelsize=20)
plt.tick_params(axis='y', labelsize=20)
plt.xlabel(r'Initial sampling distance in dispersal units', fontsize=30)
plt.ylabel("Expected Nr. of IBD-blocks >l", fontsize=30)
plt.annotate(r'$rK_1(\frac{\sqrt{2l}}{\sigma}r)$', xy=(0.08, 0.15), xycoords='axes fraction', fontsize=42)
plt.legend(prop={'size':30})
plt.show()

# Figure of Bessel-Decay for K2:
plt.figure()
x_plot = np.linspace(0.000001, 30, 100000)
y_plot = x_plot ** 2 / 0.1 * kv(2, np.sqrt(0.2) * x_plot)
y_plot1 = x_plot ** 2 / 0.05 * kv(2, np.sqrt(0.1) * x_plot)
y_plot2 = x_plot ** 2 / 0.15 * kv(2, np.sqrt(0.3) * x_plot)
plt.semilogy(x_plot, y_plot1, linewidth=4, color='blue', label=r'$l=5$ cM')
plt.semilogy(x_plot, y_plot, linewidth=4, color='green', label=r'$l=10$ cM')
plt.semilogy(x_plot, y_plot2, linewidth=4, color='red', label=r'$l=15$ cM')
plt.tick_params(axis='x', labelsize=20)
plt.tick_params(axis='y', labelsize=20)
plt.xlabel(r'Initial sampling distance in dispersal units', fontsize=30)
plt.ylabel("Expected Nr. of IBD-blocks of length l", fontsize=30)
plt.annotate(r'$\frac{r^2}{l\sigma^2} K_2(\frac{\sqrt{2l}}{\sigma}r)$', xy=(0.08, 0.15), xycoords='axes fraction', fontsize=42)
plt.legend(prop={'size':30})
plt.show()


    
# x_values = np.linspace(0.1, 10, 100)
# y_values= 3*x_values*kv(1,0.20001*x_values)
# x_plot = np.linspace(0.1,10,10000)
# 
# def bessel_decay(x,C,r):
#     # Fit to expected decay curve in 2d (C absolute value, r rate of decay)
#     return(C*x*kv(1,r*x))
#         
# # Fit with curve_fit
# parameters, cov_matrix = curve_fit(bessel_decay, x_values, y_values)  # @UnusedVariable
# perr = np.sqrt(np.diag(cov_matrix))
# print("\nAbsolute Value Guess %.5f" % parameters[0])
# print("\nDecayrate %.5f" % parameters[1])
# C,r=parameters
# print(perr)
# 
# plt.figure()
# #plt.hist(a, bins=max(a)+1-min(a), normed=1)
# plt.semilogy(x_values, y_values, 'ro')
# plt.semilogy(x_plot, C*x_plot*kv(1,r*x_plot), 'b-')
# plt.ylim(min(y_values),max(y_values))
# plt.title("Bessel fit")
# plt.show()
