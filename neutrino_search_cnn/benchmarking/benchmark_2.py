# ============================================================================================
#                                  \\ changes from last benchmark //
# ============================================================================================
"""
Since in model_2x I have changed the inputs as 4 views (one for each APA) with 3 channels each (U, V and Z planes)
I have to change the dataset. 

Changed parse_event function.

"""

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
parser.add_argument('--config', type=str, required=True, help='Path to json file.')

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

bm_modelList_path = '/afs/cern.ch/user/p/ppalomar/private/neutrino_search_cnn/benchmarking/json_settings/model_list.json'

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

# thresholds calculated previously with validation set

thresh_path = f'/afs/cern.ch/work/p/ppalomar/neutrino_search_cnn/models_storage/{script_name}/{script_name}_computed_thresholds.json'
with open(thresh_path, 'r') as f:
    thresholds = json.load(f)

TH_points_dict = thresholds

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
        'res': tf.io.FixedLenFeature([], tf.int64),
        'reconstructedEnergy': tf.io.FixedLenFeature([], tf.float32),
        'vertexX':  tf.io.FixedLenFeature([], tf.float32),
        'vertexZ':  tf.io.FixedLenFeature([], tf.float32)
    }

    for key in image_keys:
        features[key] = tf.io.FixedLenFeature([], tf.string) # this is the image feature for every key. we add them to the features dictionary
    
    parsed_features = tf.io.parse_single_example(serialized_example, features) #this separates every feature from the data given

    # we extract the label and resolution (same for every image)
    label = tf.cast(parsed_features['label'], tf.int32)
    resolution = tf.cast(parsed_features['res'], tf.int32)
    energy = tf.cast(parsed_features['reconstructedEnergy'], tf.float32)
    vertexX = tf.cast(parsed_features['vertexX'], tf.int32)
    vertexZ = tf.cast(parsed_features['vertexZ'], tf.int32)

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
    
    return apa_dict, (label, energy, vertexX, vertexZ) #return both the image array with the corresponding labels

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

test_ds = build_dataset(paths_test_mc, paths_test_bkg, [mc_test_sampling, bkg_test_sampling], False)

# ============================================================================================
#                                   \\ model prediction //
# ============================================================================================

model = keras.models.load_model(model_path)

print('\n ------ Extracting predictions from model ------\n')

scores_list = []
labels_list = []
energy_list = []
vertexX_list = []
vertexZ_list = []

for _, (label_batch, energy_batch, vX, vZ) in test_ds:
    labels_list.append(label_batch.numpy().flatten())
    energy_list.append(energy_batch.numpy().flatten())
    vertexX_list.append(vX.numpy().flatten())
    vertexZ_list.append(vZ.numpy().flatten())

labels_test = np.concatenate(labels_list, axis=0).astype(np.int8)
energy_test = np.concatenate(energy_list, axis=0).astype(np.float32)
vertexX_test = np.concatenate(vertexX_list, axis=0).astype(np.float32)
vertexZ_test = np.concatenate(vertexZ_list, axis=0).astype(np.float32)

ds_images_only = test_ds.map(lambda x, y: x, num_parallel_calls=tf.data.AUTOTUNE)

scores_test = model.predict(ds_images_only, verbose=1).flatten().astype(np.float32)

# save information for later plots

npz_save_path = output_dir / 'final_plots' / f'{script_name}_test_inference.npz'
npz_save_path.parent.mkdir(parents=True, exist_ok=True)

np.savez_compressed(
    npz_save_path,
    scores_test=scores_test,
    labels_test=labels_test,
    energy_test=energy_test,
    vertexX_test = vertexX_test,
    vertexZ_test = vertexZ_test
)

# ============================================================================================
#                                   \\ benchmarking plots & numbers //
# ============================================================================================

# compute relevant numbers from test
n_total_bkg_events = 93636
n_total_MC_events = 10120 
total_MC_POT = 1.0711964425954948e+18 

n_exp_bkg = ( n_total_bkg_events / total_spillOff_time ) * total_spillOn_time
n_exp_neutrinos = ( n_total_MC_events / total_MC_POT ) * total_spillOn_POT


fpr, tpr, thresholds = roc_curve(labels_test, scores_test)
roc_auc = auc(fpr, tpr)

plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 13,
    'font.family': 'serif',  
    'figure.dpi': 300,       
    'savefig.bbox': 'tight'  
})

# PLOT GLOBAL ROC CURVE 
plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, color='#d95f02', lw=2.5, label=f'CNN (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='#7570b3', lw=2, linestyle='--')
plt.xlim([-0.01, 1.0])
plt.ylim([0.0, 1.02])
plt.xlabel('False Positive Rate (Background Efficiency)')
plt.ylabel('True Positive Rate (Signal Efficiency)')
plt.grid(True, linestyle=':', alpha=0.7) 
plt.legend(loc="lower right", frameon=True, edgecolor='black')
plt.tight_layout()
plt.savefig(output_dir / 'roc_curve.png')
plt.clf()


# Initialize final JSON structure
benchmarking_results = {
    "model_name": model_keras_name,
    "scaling_factors": {
        "spill_on_pot": float(total_spillOn_POT),
        "mc_test_set_pot": float(total_MC_POT),
        "expected_pure_neutrinos": float(n_exp_neutrinos),
        "expected_pure_bkg": float(n_exp_bkg)
    },
    "threshold_methods": {}
}


print("\n--- EVALUATING THRESHOLDS ---")

for th_name, threshold in TH_points_dict.items():
    print(f"\nEvaluating: {th_name} (Threshold = {threshold:.4f})")
    
    y_pred_binary = (scores_test > threshold).astype(int)
    
    # CONFUSION MATRIX PLOT
    cm = confusion_matrix(labels_test, y_pred_binary)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Background', 'Signal (MC)'])
    disp.plot(cmap=plt.cm.Blues, ax=ax, values_format='d')
    plt.title(f'Confusion Matrix ({th_name})', pad=20)
    plt.tight_layout()
    plt.savefig(output_dir / f'confusion_matrix_{th_name}.png')
    plt.clf()
    
    # PROBABILITY DISTRIBUTIONS PLOT
    probs_bkg = scores_test[labels_test == 0]
    probs_sig = scores_test[labels_test == 1]

    plt.figure(figsize=(10, 6))

    # Histogrames tipus 'step' per a una visualització més neta a les presentacions
    plt.hist(probs_bkg, bins=80, range=(0, 1), density=False, color='#377eb8', alpha=1, 
             label='Background', histtype='step', linewidth=1.8)
    plt.hist(probs_sig, bins=80, range=(0, 1), density=False, color='#e41a1c', alpha=1, 
             label='Signal (MC)', histtype='step', linewidth=1.8)

    plt.axvline(x=threshold, color='green', linestyle='--', linewidth=2, label=f'Threshold ({threshold:.3f})')
    
    plt.xlabel('CNN Output Score (Probability)')
    plt.ylabel('Number of events (log)')
    plt.yscale('log')
    plt.xlim([-0.02, 1.02])
    plt.ylim(bottom=0.5)
    plt.grid(True, linestyle=':', alpha=0.6, axis='both')
    
    plt.title(f'Score Distribution ({th_name})', pad=15)
    plt.legend(loc='upper center', ncol=2, frameon=True, edgecolor='black')
    
    plt.tight_layout()
    plt.savefig(output_dir / f'prob_distribution_{th_name}.png')
    plt.clf()
    plt.close('all')


    # CALCULATE METRICS
    tn, fp, fn, tp = cm.ravel() # extract numbers from confusion matrix

    n_MC_events = tp + fn # true number of neutrinos in test
    n_BKG_events = tn + fp # true number of bkg in test 

    # calculate efficiencies and error rates

    eff_sig = tp / n_MC_events if n_MC_events > 0 else -1.0 # signal efficiency (how many of the selected signals were really signals)
    eff_bkg = tn / n_BKG_events if n_BKG_events > 0 else -1.0 # bkg efficiency (how many of the selected bkg were really bkg)
    false_positive_rate = fp / n_BKG_events if n_BKG_events > 0 else -1.0 # fp rate (how many of bkg are wrongly classified)
    false_negative_rate = fn / n_MC_events if n_MC_events > 0 else -1.0 # fn rate (how many of signal are wrongly classified)

    # calculate expected predictions of the model based on expected SpillON data
    n_model_will_predict_nu = (n_exp_neutrinos * eff_sig) + (n_exp_bkg * false_positive_rate)
    n_model_will_predict_bkg = (n_exp_bkg * eff_bkg) + (n_exp_neutrinos * false_negative_rate)

    # now just calculate (to compare with Dario's) the number of REAL neutrinos and number of BKG FP
    n_model_predict_real_neutrinos = n_exp_neutrinos * eff_sig
    n_model_predict_fp_bkg = n_exp_bkg * false_positive_rate
    # better to calculate de RATE between them:
    expected_rate_fp_bkg_over_real_neutrinos = n_model_predict_fp_bkg / n_model_predict_real_neutrinos

    # STORE IN DICTIONARY
    benchmarking_results["threshold_methods"][th_name] = {
        "threshold": threshold,
        "test_set_metrics": {
            "true_mc_neutrinos": int(n_MC_events),
            "true_bkg_events": int(n_BKG_events),
            "signal_efficiency": float(eff_sig),
            "background_rejection": float(eff_bkg),
            "false_positive_rate": float(false_positive_rate),
            "false_negative_rate": float(false_negative_rate)
        },
        "model_expectations_on_spill_on": {
            "predicted_neutrinos": float(n_model_will_predict_nu),
            "predicted_background": float(n_model_will_predict_bkg),
            "expected_real_neutrinos": float(n_model_predict_real_neutrinos),
            "expected_fp_bkg": float(n_model_predict_fp_bkg),
            "expected_rate_fp_bkg_over_real_neutrinos": float(expected_rate_fp_bkg_over_real_neutrinos)
        }
    }

# ============================================================================================
#                             PLOT 4: True Positives Vertex Z Distribution
# ============================================================================================
print("\n--- GENERATING TRUE POSITIVES VERTEX PLOTS ---")

# Obtenim la distribució total de vèrtexs Z per al senyal (MC)
mc_mask = (labels_test == 1)
vertexZ_mc_total = vertexZ_test[mc_mask]

# Determinem els límits geomètrics per mantenir el mateix binning a tots els plots
z_min, z_max = -200, 200
bins_z = np.linspace(z_min, z_max, 50)

# Generem i guardem un plot independent per a cada threshold
for th_name, threshold in TH_points_dict.items():
    
    plt.figure(figsize=(8, 6))
    
    # Màscara d'esdeveniments True Positives (És MC i supera el tall de la CNN)
    tp_mask = (labels_test == 1) & (scores_test > threshold)
    vertexZ_tp = vertexZ_test[tp_mask]
    
    clean_wp_name = th_name.replace("WP_", "").replace("_", " ")
    
    # 1. Referència: Distribució Total de Neutrins (Sense cap tall)
    plt.hist(vertexZ_mc_total, bins=bins_z, color='gray', histtype='step', 
             linestyle='--', linewidth=1.5, label=f'Total MC (N={np.sum(mc_mask)})')
            
    # 2. Distribució de True Positives (Seleccionats per la CNN)
    plt.hist(vertexZ_tp, bins=bins_z, color='#e41a1c', histtype='step', 
             linewidth=2, label=f'True Positives (N={np.sum(tp_mask)})')
    
    # Formatació professional de la gràfica
    plt.title(f'Spatial Distribution (Z)\n{clean_wp_name} (th: {threshold:.3f})', pad=15)
    plt.xlabel('True Vertex Z [cm]')
    plt.ylabel('Events')
        
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='best', frameon=True, edgecolor='black', fontsize=11)

    plt.tight_layout()
    plot_path = output_dir / f'true_positives_vertexZ_{th_name}.png'
    plt.savefig(plot_path)
    plt.close()

    print(f"---> Vertex plot saved to: {plot_path}")

# ============================================================================================
#                                   \\ save results //
# ============================================================================================

results_file_path = output_dir / f"{script_name}_results.json"

with open(results_file_path, 'w') as f:
    json.dump(benchmarking_results, f, indent=4)

print(f'\n---> Benchmarking results successfully saved to: {results_file_path}')
print('BENCHMARKING FINISHED')