import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import uproot
import time
from scipy import stats

path = '/afs/cern.ch/work/d/dapullia/public/pau/combined_data.root'
f = uproot.open(path)
print(f'Keys in the file are {f.keys()}')

tree = f['ana/tree_truth']
print(f'Keys inside truth are: \n{tree.keys()}')

triggers = tree['triggerActivityFlag'].array()
print(f'Trigger activities are \n{triggers}')