'''
Created on Mar 2, 2015

@author: Harald
'''

import discsim
import ercs
import numpy as np
from math import hypot
from math import pi
from math import sqrt
import matplotlib.pyplot as plt

a=np.array([0,1,2,3,4,5])
b=np.array([0.1,2.2,3.0,3.9,4.01,4.95])

K, C = np.polyfit(a, b, 1, full=False, cov=True)

print((K,C))

x=2*np.random.random(size=(100000,2))-1
x*=5
circle=[i for i in x if (i[0]**2+i[1]**2)<=2*9]   # Should be pi/4

print("Length of cirlce: %.2f" % len(circle))

pair_dist=[circle[2*i] - circle[2*i + 1] for i in range(0,len(circle)/2)]

euc_distances=[hypot(i[0],i[1]) for i in pair_dist]
distances=[i[0] for i in pair_dist]    # x-Distances
distances_y=[i[1] for i in pair_dist]  # y-Distances   

print("Variance: %.2f" % np.var(distances))
print("Covariance: %.2f" % np.cov(distances,distances_y)[1,0])
print("Std: %.2f" % np.std(distances))
print("Mean: %.2f" % np.mean(distances))

plt.figure()
plt.hist(distances,bins=100)
plt.show()


