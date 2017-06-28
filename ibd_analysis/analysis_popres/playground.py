'''
Created on Feb 5, 2016

@author: Harald
'''
import numpy as np
import matplotlib.pyplot as plt
from bisect import bisect_left  # @UnusedImport




x=np.array([0,1,5,10,20,50,100,200,300])
p=np.array([24.5,14.7,14.1,12.3,15.4,8.6,5.8,4.6])

dx=x[1:]-x[:-1]
px=p/dx

plt.plot()
plt.bar(x[:-1],px,dx)
plt.show()

mx=x[:-1]+dx/2.0

variance=np.sum(mx**2 *p/100.0)
print(np.sqrt(variance))





n = 5
# a=np.tril(np.ones((n, n), dtype=int), 0)

x = np.linspace(0, 10, 10000)
y = np.exp(-13 - 2 * x + 4.3 * np.sqrt(x)) * 3587.00
y1 = np.exp(-13.704 - 2.095 * x + 4.381 * np.sqrt(x)) * 3587

plt.figure()
plt.plot(x, y, label="False Positive rate")
plt.plot(x, y1, label="Github false positive rate")
plt.legend()
# plt.show()

y = 1 - 1.0 / (1 + 0.077 * x * x * np.exp(0.54 * x))
y1 = 1 - 1 / (1 + 0.05718153 * x * np.exp(0.95613164 * x)) 

plt.figure()
plt.plot(x, y, label="Power")
plt.plot(x, y1, label="Github Power")
plt.grid(True)
# plt.show()
