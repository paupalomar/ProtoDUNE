import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import uproot
import time
from scipy import stats

print('Initializing timing benchmark')
t0 = time.time()


# path = "/exp/dune/app/users/dpullia/protodunedm_analysis/work/anaOut.root"
# path = "/pnfs/dune/scratch/users/dpullia/outana/wnp04/numu/decay2/anaOut_132.root"
path ="/afs/cern.ch/work/d/dapullia/public/pau/prod_protodunehd_beamneutrino_1_342755_3_1777954807_gen_g4_detsim_trigger_reco_stage1_reco_stage2_anaOut.root"
f = uproot.open(path)
tree=f["ana/tree_image"]
print(tree.keys())

imageU1s = np.array(tree["imageU1"].array())
imageU1s[imageU1s == 0] = np.nan

imageU2s = np.array(tree["imageU2"].array())
imageU2s[imageU2s == 0] = np.nan

imageU3s = np.array(tree["imageU3"].array())
imageU3s[imageU3s == 0] = np.nan

imageU4s = np.array(tree["imageU4"].array())
imageU4s[imageU4s == 0] = np.nan

imageV1s = np.array(tree["imageV1"].array())
imageV1s[imageV1s == 0] = np.nan

imageV2s = np.array(tree["imageV2"].array())
imageV2s[imageV2s == 0] = np.nan

imageV3s = np.array(tree["imageV3"].array())
imageV3s[imageV3s == 0] = np.nan

imageV4s = np.array(tree["imageV4"].array())
imageV4s[imageV4s == 0] = np.nan

imageZ1s = np.array(tree["imageZ1"].array())
imageZ1s[imageZ1s == 0] = np.nan

imageZ2s = np.array(tree["imageZ2"].array())
imageZ2s[imageZ2s == 0] = np.nan

imageZ3s = np.array(tree["imageZ3"].array())
imageZ3s[imageZ3s == 0] = np.nan

imageZ4s = np.array(tree["imageZ4"].array())
imageZ4s[imageZ4s == 0] = np.nan

t1 = time.time()
delta_t = t1-t0
delta_t_min = delta_t/60

print(f'Process completed. Total time: {delta_t} seconds, which is {delta_t_min} minutes')
print(f'U1 size is: {imageU1s.size}')

index=9 # there's 10 events.

# ///////////////////////////////////
#              U Planes
# ///////////////////////////////////
plt.imshow(imageU1s[index], aspect='auto', interpolation='nearest')
plt.colorbar()
plt.xlabel('channel')
plt.ylabel('time')
plt.savefig("imageU1.png")
plt.clf()

plt.imshow(imageU2s[index], aspect='auto', interpolation='nearest')
plt.colorbar()
plt.xlabel('channel')
plt.ylabel('time')
plt.savefig("imageU2.png")
plt.clf()

plt.imshow(imageU3s[index], aspect='auto', interpolation='nearest')
plt.colorbar()
plt.xlabel('channel')
plt.ylabel('time')
plt.savefig("imageU3.png")
plt.clf()

plt.imshow(imageU4s[index], aspect='auto', interpolation='nearest')
plt.colorbar()
plt.xlabel('channel')
plt.ylabel('time')
plt.savefig("imageU4.png")
plt.clf()

# ///////////////////////////////////
#              V Planes
# ///////////////////////////////////

plt.imshow(imageV1s[index], aspect='auto', interpolation='nearest')
plt.colorbar()
plt.xlabel('channel')
plt.ylabel('time')
plt.savefig("imageV1.png")
plt.clf()

plt.imshow(imageV2s[index], aspect='auto', interpolation='nearest')
plt.colorbar()
plt.xlabel('channel')
plt.ylabel('time')
plt.savefig("imageV2.png")
plt.clf()

plt.imshow(imageV3s[index], aspect='auto', interpolation='nearest')
plt.colorbar()
plt.xlabel('channel')
plt.ylabel('time')
plt.savefig("imageV3.png")
plt.clf()

plt.imshow(imageV4s[index], aspect='auto', interpolation='nearest')
plt.colorbar()
plt.xlabel('channel')
plt.ylabel('time')
plt.savefig("imageV4.png")
plt.clf()

# ///////////////////////////////////
#              Z Planes
# ///////////////////////////////////

plt.imshow(imageZ1s[index], aspect='auto', interpolation='nearest')
plt.colorbar()
plt.xlabel('channel')
plt.ylabel('time')
plt.savefig("imageZ1.png")
plt.clf()

plt.imshow(imageZ2s[index], aspect='auto', interpolation='nearest')
plt.gca().invert_yaxis()
plt.colorbar()
plt.xlabel('channel')
plt.ylabel('time')
plt.savefig("imageZ2.png")
plt.clf()

plt.imshow(imageZ3s[index], aspect='auto', interpolation='nearest')
plt.colorbar()
plt.xlabel('channel')
plt.ylabel('time')
plt.savefig("imageZ3.png")
plt.clf()

plt.imshow(imageZ4s[index], aspect='auto', interpolation='nearest')
plt.gca().invert_yaxis()
plt.colorbar()
plt.xlabel('channel')
plt.ylabel('time')
plt.savefig("imageZ4.png")