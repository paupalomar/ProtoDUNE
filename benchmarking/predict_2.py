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

# load model configuration

bm_modelList_path = '/afs/cern.ch/user/p/ppalomar/private/neutrino_search_cnn/benchmarking/json_settings/model_list.json'

with open(bm_modelList_path, 'r') as f:
    model_list = json.load(f)

model_version = model_list["which_to_benchmark"] # Ex: "01"
group_key = f"models_{model_version[0]}x"        # Ex: "models_0x"
script_name = f"model_{model_version}"           # Ex: "model_01" (for plotting)

model_keras_name = model_list[group_key][f"{script_name}_name"]
model_output_dir = model_list[group_key][f"{script_name}_output_file"]

model_path = Path(model_output_dir) / model_keras_name

print(f'\n--- BENCHMARKING INITIALIZED ---')
print(f'Target: {script_name}')
print(f'Model path: {model_path}\n')

# create the folder if neccessary
output_dir = Path(model_output_dir)
output_dir.mkdir(parents=True, exist_ok=True)


# load model evaluation results file
results_file_path = f'/afs/cern.ch/work/p/ppalomar/neutrino_search_cnn/models_storage/{script_name}/{script_name}_results.json'

with open(results_file_path, 'r') as f:
    results_data = json.load(f)

exp_bkg_total = results_data["scaling_factors"]["expected_pure_bkg"]

# ============================================================================================
#                                      \\ data pipeline //
# ============================================================================================

path_spillOn = glob.glob('/eos/user/p/ppalomar/official_sharded_datasets_def/bkg_spillON_noCut/bkg_data_125_spillOn_NoCut_*.tfrecord')

image_keys = [
    'imageU1', 'imageV1', 'imageZ1', 
    'imageU2', 'imageV2', 'imageZ2', 
    'imageU3', 'imageV3', 'imageZ3', 
    'imageU4', 'imageV4', 'imageZ4'
]

random.Random(42).shuffle(path_spillOn)

def parse_event(serialized_example):
    """
    This function is applied to every event from the .tfrecord file.
    """

    # Extract the features, TensorFlow should expect to find these in the file
    features = {
        'res': tf.io.FixedLenFeature([], tf.int64),
        'reconstructedEnergy': tf.io.FixedLenFeature([], tf.float32)
    }

    for key in image_keys:
        features[key] = tf.io.FixedLenFeature([], tf.string) # this is the image feature for every key. we add them to the features dictionary
    
    parsed_features = tf.io.parse_single_example(serialized_example, features) #this separates every feature from the data given

    # we extract the label and resolution (same for every image)
    resolution = tf.cast(parsed_features['res'], tf.int32)
    energy = tf.cast(parsed_features['reconstructedEnergy'], tf.int32)
    

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
    
    return apa_dict, energy #return both the image array with the corresponding labels

ds = tf.data.TFRecordDataset(path_spillOn, compression_type='GZIP', num_parallel_reads=tf.data.AUTOTUNE)
ds = ds.map(parse_event, num_parallel_calls=tf.data.AUTOTUNE)
ds = ds.batch(64)
ds = ds.prefetch(tf.data.AUTOTUNE)

# ============================================================================================
#                                   \\ model prediction //
# ============================================================================================

model = keras.models.load_model(model_path)

print('\n ------ Extracting predictions from model ------\n')

energy_list = []

for _, energy_batch in ds:
    energy_list.append(energy_batch.numpy().flatten())

ds_images_only = ds.map(lambda x, y: x, num_parallel_calls=tf.data.AUTOTUNE)

scores_spill_on = model.predict(ds_images_only, verbose=1).flatten().astype(np.float32)
energy_spill_on = np.concatenate(energy_list, axis=0)

# save scores and energies for plots later
npz_save_path = output_dir / 'final_plots' / f'{script_name}_spillON_inference.npz'
npz_save_path.parent.mkdir(parents=True, exist_ok=True)

np.savez_compressed(
    npz_save_path,
    scores_spill_on=scores_spill_on,
    energy_spill_on=energy_spill_on
)


# Initialize dictionary to store results for all working points
results_data["real_spill_on_results"] = {}

# Loop through the different thresholds evaluated in the benchmarking script
for wp_name, wp_data in results_data["threshold_methods"].items():
    
    threshold = wp_data["threshold"]
    fpr = wp_data["test_set_metrics"]["false_positive_rate"]
    eff_sig = wp_data["test_set_metrics"]["signal_efficiency"]
    expected_neutrinos = wp_data["model_expectations_on_spill_on"]["predicted_neutrinos"]
    
    # Apply threshold mask
    y_pred_binary = (scores_spill_on > threshold).astype(int)

    n_neutrinos_found = int(np.sum(y_pred_binary==1))
    n_bkg_found = int(np.sum(y_pred_binary==0))

    # Standard physics calculations (Theoretical BKG)
    # i want to calculate how many of the n_neutrinos_found are actually neutrinos. We do it by background subtraction
    # we use the theoretical BKG propagated from spillOFF data
    estimated_cosmics_pass_selection = exp_bkg_total * fpr
    estimated_neutrinos_and_beam_bkg_pass_selection = n_neutrinos_found - estimated_cosmics_pass_selection
    # with this data, I can 'extrapolate' to find how many REAL neutrinos are in the spillOn data by dividing by the efficiency of the model
    # this number should be bigger than 387 since we expect beam bkg's to also pass
    estimated_total_pure_neutrinos = estimated_neutrinos_and_beam_bkg_pass_selection / eff_sig if eff_sig > 0 else -1.0


    # Save metrics for this specific working point
    results_data["real_spill_on_results"][wp_name] = {
        "total_events_spillON": int(len(y_pred_binary)),
        "found_neutrinos": n_neutrinos_found,
        "found_background": n_bkg_found,
        "difference_vs_expected_neutrinos": n_neutrinos_found - expected_neutrinos,
        "estimated_cosmics_pass_selection": estimated_cosmics_pass_selection,
        "estimated_neutrinos_and_beam_bkg_pass_selection": estimated_neutrinos_and_beam_bkg_pass_selection,
        "estimated_total_pure_neutrinos_in_beam": estimated_total_pure_neutrinos,
    }

# Save back to the JSON file
with open(results_file_path, 'w') as f:
    json.dump(results_data, f, indent=4)
    
print(f"\nResults for all working points successfully updated in {results_file_path}")
    