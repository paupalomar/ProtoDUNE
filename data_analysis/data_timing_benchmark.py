import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import uproot
import time
from scipy import stats

print('Initializing timing benchmark...')
t0 = time.time()

path = '/afs/cern.ch/work/d/dapullia/public/pau/combined_data.root'

f = uproot.open(path)
tree = f['ana/tree_image']

imageU1s = np.array(tree["imageU1"].array())
imageU1s[imageU1s == 0] = np.nan

t1 = time.time()
delta_t = t1-t0
delta_t_min = delta_t/60

print('Benchmark ended.')
print(f'Total timing was {delta_t} seconds, or {delta_t_min} minutes')
print(f'U1 size is {imageU1s.size}')