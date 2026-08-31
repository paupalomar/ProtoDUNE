import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

import os
import sys
import argparse
import glob
import random
import json
import time
import re
from pathlib import Path

tf.random.set_seed(42)

# ============================================================================================
#                                      \\ get arguments //
# ============================================================================================

parser = argparse.ArgumentParser(description='Teaching physics to a computer...')
parser.add_argument('--config', type=str, required=True, help='Path to json file.') # PUT CONFIG OF THE GIVEN MODEL I AM BENCHMARKING

args = parser.parse_args()

with open(args.config, 'r') as f:
    config = json.load(f)

# args from config

train_split = config["data_info"]["train_split"]
val_split = config["data_info"]["val_split"]
test_split = config["data_info"]["test_split"]

mc_train_sampling = config["data_info"]["mc_train_sampling"]
bkg_train_sampling = config["data_info"]["bkg_train_sampling"]
mc_val_sampling = config["data_info"]["mc_val_sampling"]
bkg_val_sampling = config["data_info"]["bkg_val_sampling"]
mc_test_sampling = config["data_info"]["mc_test_sampling"]
bkg_test_sampling = config["data_info"]["bkg_test_sampling"]

batch_size = config["data_info"]["batch_size"]

# load model configuration

bm_modelList_path = '/afs/cern.ch/user/p/ppalomar/private/neutrino_search_cnn/benchmarking/json_settings/model_list.json' #IMPORTANT TO CHANGE THIS EACH TIME TO THE CORRECT MODEL (if its not the last one trained)!

with open(bm_modelList_path, 'r') as f:
    model_list = json.load(f)

model_version = model_list["which_to_benchmark"] # Ex: "01"
group_key = f"models_{model_version[0]}x"        # Ex: "models_0x"
script_name = f"model_{model_version}"           # Ex: "model_01" (for plotting)

model_keras_name = model_list[group_key][f"{script_name}_name"]
model_output_dir = model_list[group_key][f"{script_name}_output_file"]

model_path = Path(model_output_dir) / model_keras_name

# relevant numbers

total_spillOn_POT = model_list["run_parameters"]["spill_on"]["total_POT"]
total_spillOn_time = model_list["run_parameters"]["spill_on"]["total_time"] # in hours
total_spillOff_time = model_list["run_parameters"]["spill_off"]["total_time"] # in hours


print(f'\n--- BENCHMARKING INITIALIZED ---')
print(f'Target: {script_name}')
print(f'Model path: {model_path}\n')

# create the folder if neccessary
output_dir = Path(model_output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

# ============================================================================================
#                                      \\ data pipeline //
# ============================================================================================

path_mc = glob.glob('/eos/user/p/ppalomar/sharded_datasets_def/mc_TAcut/mc_data_125_*.tfrecord')
path_bkg = glob.glob('/eos/user/p/ppalomar/sharded_datasets_def/bkg_just_spillOff/bkg_data_125_*.tfrecord')

image_keys = [
    'imageU1', 'imageV1', 'imageZ1', 
    'imageU2', 'imageV2', 'imageZ2', 
    'imageU3', 'imageV3', 'imageZ3', 
    'imageU4', 'imageV4', 'imageZ4'
]

random.Random(42).shuffle(path_mc)
random.Random(42).shuffle(path_bkg)

# extract always the same events for testing

def extract_num_events(file_list):
    file_info = {}
    for file_path in file_list:
        match = re.search(r'_(\d+)\.tfrecord$', file_path) # event number has to be in the end!

        if match:
            n_events = int(match.group(1))
            file_info[file_path] = n_events
    return file_info

mc_files_info = extract_num_events(path_mc)
bkg_files_info = extract_num_events(path_bkg)

def split_files(file_info, target_events):
    """
    file_info is a dictionary with keys being files paths and values being number of events of that file

    target_events is an array with 3 entries: [train_events, val_events, test_events]
    """
    train_files = []
    val_files = []
    test_files = []
    current_events = 0
    for key, n_events in file_info.items():
        # take train events
        if current_events < target_events[0]:
            train_files.append(key)
            current_events += n_events
        elif current_events < (target_events[0] + target_events[1]):
            val_files.append(key)
            current_events += n_events
        elif current_events < (target_events[0] + target_events[1] + target_events[2]):
            test_files.append(key)
            current_events += n_events
            
    return train_files, val_files, test_files

def count_total_events(file_info):
    total = 0
    for key, n_events in file_info.items():
        total += n_events
    return total

total_mc_events = count_total_events(mc_files_info)
total_bkg_events = count_total_events(bkg_files_info)

# count how many events we'll have in each split, according to sampling weights 
if total_mc_events < total_bkg_events:
    train_mc_events = total_mc_events * train_split
    train_bkg_events = train_mc_events * (bkg_train_sampling / mc_train_sampling)

    val_mc_events = total_mc_events * val_split
    val_bkg_events = val_mc_events * (bkg_val_sampling / mc_val_sampling)

    test_mc_events = total_mc_events * test_split
    test_bkg_events = test_mc_events * (bkg_test_sampling / mc_test_sampling)
else:
    raise ValueError('There is more MC events than BKG. Script is not prepared for this case. Update it.')

target_mc_events = [int(train_mc_events), int(val_mc_events), int(test_mc_events)]
target_bkg_events = [int(train_bkg_events), int(val_bkg_events), int(test_bkg_events)]

paths_train_mc, paths_val_mc, paths_test_mc = split_files(mc_files_info, target_mc_events)
paths_train_bkg, paths_val_bkg, paths_test_bkg = split_files(bkg_files_info, target_bkg_events)

# parsing events, extracting features

def parse_event(serialized_example):
    """
    This function is applied to every event from the .tfrecord file.
    """

    # Extract the features, TensorFlow should expect to find these in the file
    features = {
        'label': tf.io.FixedLenFeature([], tf.int64),
        'res': tf.io.FixedLenFeature([], tf.int64)
    }

    for key in image_keys:
        features[key] = tf.io.FixedLenFeature([], tf.string) # this is the image feature for every key. we add them to the features dictionary
    
    parsed_features = tf.io.parse_single_example(serialized_example, features) #this separates every feature from the data given

    # we extract the label and resolution (same for every image)
    label = tf.cast(parsed_features['label'], tf.int32)
    resolution = tf.cast(parsed_features['res'], tf.int32)

    # for every image, we extract it, we get a 1d array, and reshape it to an actual image height x width x 1
    images_dict = {}
    for key in image_keys:
        img_1d = tf.io.decode_raw(parsed_features[key], tf.float32)
        img_3d = tf.reshape(img_1d, [resolution, resolution, 1])
        img_3d = tf.where(tf.math.is_nan(img_3d), tf.zeros_like(img_3d), img_3d) #get rid of the nan's, they were just for visualization via matplotlib, just in case I forgot
        images_dict[key] = img_3d #we add the image to the main dictionary of images

    # now join images by APA's
    apa_dict = {}
    apa_dict['APA1'] = tf.concat([images_dict['imageU1'], images_dict['imageV1'], images_dict['imageZ1']], axis=-1)
    apa_dict['APA2'] = tf.concat([images_dict['imageU2'], images_dict['imageV2'], images_dict['imageZ2']], axis=-1)
    apa_dict['APA3'] = tf.concat([images_dict['imageU3'], images_dict['imageV3'], images_dict['imageZ3']], axis=-1)
    apa_dict['APA4'] = tf.concat([images_dict['imageU4'], images_dict['imageV4'], images_dict['imageZ4']], axis=-1)
    
    return apa_dict, label #return both the image array with the corresponding labels

def build_dataset(mc_list, bkg_list, weights, is_training=True):
    raw_mc_ds = tf.data.Dataset.from_tensor_slices(mc_list)
    raw_bkg_ds = tf.data.Dataset.from_tensor_slices(bkg_list)

    raw_mc_ds = raw_mc_ds.interleave(
        lambda x: tf.data.TFRecordDataset(x, compression_type='GZIP'),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False)
    raw_bkg_ds = raw_bkg_ds.interleave(
        lambda x: tf.data.TFRecordDataset(x, compression_type='GZIP'), 
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False)

    ds = tf.data.Dataset.sample_from_datasets(
                    [raw_mc_ds, raw_bkg_ds],
                    weights = weights,
                    stop_on_empty_dataset=True,
                    seed = 42
                )
    
    if is_training:
        ds = ds.shuffle(buffer_size=1000)

    ds = ds.map(parse_event, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

val_ds = build_dataset(paths_val_mc, paths_val_bkg, [mc_val_sampling, bkg_val_sampling], False)

# ============================================================================================
#                                   \\ model prediction //
# ============================================================================================
model = keras.models.load_model(model_path)

print('\n ------ Extracting predictions from model ------\n')

scores_list = []
labels_list = []

for _, label_batch in val_ds:
    labels_list.append(label_batch.numpy().flatten())

labels_test = np.concatenate(labels_list, axis=0).astype(np.int8)

ds_images_only = val_ds.map(lambda x, y: x, num_parallel_calls=tf.data.AUTOTUNE)

scores_test = model.predict(ds_images_only, verbose=1).flatten().astype(np.float32)
# ============================================================================================
#                               \\ calculate optimal threshold //
# ============================================================================================ 

# Pre-calculate expectation scaling constraints
# note: I extract the total MC POT from the .root files directly since the .tfrecord files just have triggered events
# and thus, if I summed the POT's from the .tfrecord events, wouldn't be correct. I do this calculation in another script.
#n_total_bkg_events = 93636 # this is for datasets previous to model 30
n_total_bkg_events = 92256
n_total_MC_events = 10120 
total_MC_POT = 1.0711964425954948e+18 

n_exp_bkg = ( n_total_bkg_events / total_spillOff_time ) * total_spillOn_time
n_exp_neutrinos = ( n_total_MC_events / total_MC_POT ) * total_spillOn_POT

fpr, tpr, thresholds = roc_curve(labels_test, scores_test)
roc_auc = auc(fpr, tpr)


#  ------- DIFFERENT METHODS OF THRESHOLD CALCULATION ---------

# 1. WEIGHTED YOUDEN
# Weight the FPR relative to the real BKG/Signal proportion
W = n_exp_bkg / n_exp_neutrinos
best_youden_idx = np.argmax(tpr - (W * fpr))
thresh_youden = float(thresholds[best_youden_idx]) if thresholds[best_youden_idx] < 1 else 0.0


# 2. FIXED FPR (e.g., 0.5%)
target_fpr = 0.00082
best_targetFPR_idx = np.argmin(np.abs(fpr - target_fpr))
thresh_fixed_fpr = float(thresholds[best_targetFPR_idx])

# 3. GAUSSIAN MAXIMUM SIGNIFICANCE
S_array = tpr * n_exp_neutrinos
B_array = fpr * n_exp_bkg
significance = S_array / np.sqrt(S_array + B_array + 1e-10)
best_sig_idx = np.argmax(significance)
thresh_sig = float(thresholds[best_sig_idx])

# 4. POISSION MAXIMUM SIGNIFICANCE

mu = B_array
n_on = S_array + B_array

# This is to avoid division by 0
eps = 1e-10
mu_safe = np.clip(mu, eps, None)
n_on_safe = np.clip(n_on, eps, None)

q0_array = 2 * (n_on_safe * np.log(n_on_safe / mu_safe) + mu_safe - n_on_safe)
q0_array = np.clip(q0_array, 0, None) # avoid negatives so I can compute sqrt

poisson_significance = np.sqrt(q0_array)

best_poiss_sig_idx = np.argmax(poisson_significance)
thresh_poiss_sig = float(thresholds[best_poiss_sig_idx])
max_Z = float(poisson_significance[best_poiss_sig_idx])

print(f"Optimal Threshold (Poisson): {thresh_poiss_sig:.4f} with Z = {max_Z:.2f} sigmas") # I think I got 23 sigmas

thresholds_dict = {
    "TH_Weighted_Youden": thresh_youden,
    "TH_Fixed_FPR": thresh_fixed_fpr,
    "TH_Max_Gaussian_Significance": thresh_sig,
    "TH_Max_Poisson_Significance": thresh_poiss_sig
}

# ============================================================================================
#                               \\ plot score distribution //
# ============================================================================================ 
print('\n--- GENERATING SCORE DISTRIBUTION PLOT ---')

plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'font.family': 'serif',  
    'figure.dpi': 300,       
    'savefig.bbox': 'tight'  
})

probs_bkg = scores_test[labels_test == 0]
probs_sig = scores_test[labels_test == 1]

plt.figure(figsize=(10, 6))

# Histogrames en escala logarítmica (gruix de línia pujat a 1.8 per a presentacions)
plt.hist(probs_bkg, bins=80, range=(0, 1), color='#377eb8', alpha=1, 
         label='Background', histtype='step', linewidth=1.8)
plt.hist(probs_sig, bins=80, range=(0, 1), color='#e41a1c', alpha=1, 
         label='Signal (MC)', histtype='step', linewidth=1.8)

# Línies verticals per als llindars calculats
plt.axvline(x=thresh_poiss_sig, color='green', linestyle='--', linewidth=2, 
            label=f'Poisson Thresh ({thresh_poiss_sig:.3f})')
plt.axvline(x=thresh_fixed_fpr, color='purple', linestyle='-.', linewidth=2, 
            label=f'Fixed FPR Thresh ({thresh_fixed_fpr:.3f})')

plt.xlabel('CNN Output Score (Probability)')
plt.ylabel('Number of events')
plt.yscale('log')
plt.xlim([-0.02, 1.02])
plt.grid(True, linestyle=':', alpha=0.6, axis='both')

# CORRECCIÓ DE LA LLEGENDA I EL TÍTOL
# Canviem loc='lower center' perquè ancori la base de la llegenda a la coordenada Y=1.03 (just a sobre del plot)
# Fiquem ncol=2 per fer una graella simètrica 2x2 amb els 4 elements.
plt.title('Validation Set Score Distribution & Calculated Thresholds')
plt.legend(loc='upper center', ncol=2, frameon=True, edgecolor='black')
plt.tight_layout()

plot_file_path = output_dir / f"validation_{script_name}_score_distribution.png"
plt.savefig(plot_file_path)
plt.clf()
plt.close()

print(f'---> Score distribution plot successfully saved to: {plot_file_path}')

# ============================================================================================
#                                   \\ save results //
# ============================================================================================

results_file_path = output_dir / f"{script_name}_computed_thresholds.json"

with open(results_file_path, 'w') as f:
    json.dump(thresholds_dict, f, indent=4)

print(f'\n---> Thresholds computation results successfully saved to: {results_file_path}')
print('THRESHOLD CALCULATIONS FINISHED')