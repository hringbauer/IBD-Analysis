'''
Created on Feb 4, 2016
Testing the IBD-csv files from github
@author: Harald
'''

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

pop_path = "/home/hringbauer/IST/BlockInference/Popres Data/ibd-pop-info.csv"
ibd_list_path = "/home/hringbauer/IST/BlockInference/Popres Data/ibd-blocklens.csv"
country_path = "/home/hringbauer/IST/BlockInference/Popres Data/country_centroids.csv"
country_string = '"Ireland"'  # Which country to analyse


raw_data_pops = np.loadtxt(pop_path, dtype='string', delimiter=',')[1:, :]  # Load the cross_reference list
raw_data_blocks = np.loadtxt(ibd_list_path, dtype='string', delimiter=',')[1:, :]
geo_list = np.loadtxt(country_path, dtype='string', delimiter=',')


print("Done")

k = len(raw_data_pops)
n = len(raw_data_blocks)

print("Total number of inds: %.1f" % k)
print("Total number of blocks: %.1f" % n)

counter = Counter(raw_data_pops[:, 1])  # Count how often certain Countries appear

for i in counter.most_common():
    print("%s: %.1f" % (str(i[0]), i[1]))

print("Average Sharing: %.2f" % (n * 2.0 / (k * (k - 1))))  # Calculates Average Block Sharing

# print(raw_data_blocks[3,1:10])
print("Median sharing (in cm): %.4f" % np.median(raw_data_blocks[:, 3].astype('float')))
print("25 percentile (in cm): %.4f" % np.percentile(raw_data_blocks[:, 3].astype('float'), 25))
print("75 percentile (in cm): %.4f" % np.percentile(raw_data_blocks[:, 3].astype('float'), 75))

print("Blocks between 3 and 3.1 centimorgan")
a = np.sum((raw_data_blocks[:, 3].astype('float') > 5.0) * (raw_data_blocks[:, 3].astype('float') < 5.1))

print("Block sharing between 3 and 3.1 centimorgan per pair: %.4f" % (2 * a / (k * (k - 1.0))))


plt.plot()
plt.hist(raw_data_blocks[:, 3].astype('float'), range=[0, 10], bins=100)

print("Maximal block found %.4f" % np.max(raw_data_blocks[:, 3].astype('float')))
plt.show()



raw_data_alb = raw_data_pops[raw_data_pops[:, 1] == country_string, :]  # Extract Albania pops
albania_ids = raw_data_alb[:, 0]  # Extract Albania IDs
k_a = len(albania_ids)        

# Exctract Albanian Albanian shared blocks
albania_sharing = raw_data_blocks[np.in1d(raw_data_blocks[:, 0], albania_ids) * np.in1d(raw_data_blocks[:, 1], albania_ids), 0]  # Get Albania Sharing
print("Average country block sharing: %.3f" % (len(albania_sharing) / (k_a * (k_a - 1) / 2.0)))

# Detect double sharing
def unique_rows(a):
    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([('', a.dtype)] * a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))

#Detect unique block sharing
long_blocks = raw_data_blocks[raw_data_blocks[:, 3].astype('float') > 6]
print(len(long_blocks[:, 0]))
print(len(unique_rows(long_blocks[:,0:2])))

