# ============================================================================================
#                                  \\ changes from last model //
# ============================================================================================
"""
Same but using new datasets.
"""
# ============================================================================================

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

import os
import sys
import argparse
import glob
import random
import json
import time
from pathlib import Path # to extract script file name and num of events from .tfrecord files
import re

tf.random.set_seed(42)

start_time_total = time.time()

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

raw_class_weights_train = config["class_weights_train"]
class_weights_train = {int(k): float(v) for k, v in raw_class_weights_train.items()} # convert to correct format

# extract this script name
script_path = Path(__file__)
script_name = script_path.stem # without the .py extension

match = re.search(r'_(\d+)$', script_name)
model_version = match.group(1) if match else "unknown" # Ex: "01"

# create the folder
output_dir = Path(f'/afs/cern.ch/work/p/ppalomar/neutrino_search_cnn/models_storage/{script_name}')
output_dir.mkdir(parents=True, exist_ok=True)

# ============================================================================================
#                                      \\ data pipeline //
# ============================================================================================

path_mc = glob.glob('/eos/user/p/ppalomar/official_sharded_datasets_def/mc_TAcut/mc_data_125_*.tfrecord')
path_bkg = glob.glob('/eos/user/p/ppalomar/official_sharded_datasets_def/bkg_just_spillOff/bkg_data_125_*.tfrecord')

image_keys = [
    'imageU1', 'imageV1', 'imageZ1', 
    'imageU2', 'imageV2', 'imageZ2', 
    'imageU3', 'imageV3', 'imageZ3', 
    'imageU4', 'imageV4', 'imageZ4'
]

random.Random(42).shuffle(path_mc)
random.Random(42).shuffle(path_bkg)

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

print(f"--------> THERE'S {total_mc_events} MC EVENTS AND {total_bkg_events} BKG SPILL OFF EVENTS")

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

def parse_event(serialized_example):
    """
    This function is applied to every event from the .tfrecord file.
    """

    # Extract the features, TensorFlow should expect to find these in the file
    features = {
        'label': tf.io.FixedLenFeature([], tf.int64),
        'res': tf.io.FixedLenFeature([], tf.int64),
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

train_ds = build_dataset(paths_train_mc, paths_train_bkg, [mc_train_sampling, bkg_train_sampling], True)
val_ds = build_dataset(paths_val_mc, paths_val_bkg, [mc_val_sampling, bkg_val_sampling], False)
test_ds = build_dataset(paths_test_mc, paths_test_bkg, [mc_test_sampling, bkg_test_sampling], False)

# ============================================================================================
#                                   \\ model architecture //
# ============================================================================================


# DEFINE INPUT CONFIG

resolution = 125
channels = 3

# CREATE SINGLE VIEW MODEL

def create_shared_feature_extractor(input_shape):
    input_tensor = tf.keras.Input(shape=input_shape)

    x = tf.keras.layers.Conv2D(16, kernel_size=(2,2), padding = 'same', activation = 'relu')(input_tensor)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(x) # we use max instead of average since we don't want the hits to smooth with the 0 background and lose resolution

    x = tf.keras.layers.Conv2D(32, kernel_size=(3,3), padding = 'same', activation = 'relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size = (2,2))(x)

    x = tf.keras.layers.Conv2D(64, kernel_size=(3,3), padding = 'same', activation = 'relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size = (2,2))(x)

    x = tf.keras.layers.GlobalMaxPooling2D()(x) #Flattening here will yield too many parameters, causing it to go very slow.
    return tf.keras.models.Model(inputs=input_tensor, outputs = x, name='shared_extractor')

def create_dense_layer(merged_input):
    x = tf.keras.layers.Dense(128, activation = 'relu')(merged_input)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(64, activation = 'relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(16, activation = 'relu')(x)

    x = tf.keras.layers.Dense(1, activation = 'sigmoid')(x) #output layer
    return x

def create_model():
    apa_keys = ['APA1', 'APA2', 'APA3', 'APA4']
    inputs = {key: tf.keras.Input(shape=(resolution, resolution, channels), name = key) for key in apa_keys}

    shared_extractor = create_shared_feature_extractor((resolution, resolution, channels))

    encoded_apas = []
    for key in apa_keys:
        encoded_apas.append(shared_extractor(inputs[key]))
    
    merged_apas = tf.keras.layers.concatenate(encoded_apas)
    outputs = create_dense_layer(merged_apas)
    model = tf.keras.models.Model(inputs=inputs, outputs = outputs)

    return model

model = create_model()

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

model.compile(
    optimizer = optimizer,
    loss = 'binary_crossentropy',
    metrics = ['accuracy', tf.keras.metrics.AUC(name='auc')]
)
model.summary()

# ============================================================================================
#                                      \\ training //
# ============================================================================================

# MODEL FITTING 

timestr = time.strftime('%Y%m%d_%H%M')

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor = 'val_auc',
        mode = 'max',
        patience = 5,
        restore_best_weights = True,
        verbose = 1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        filepath = f'/afs/cern.ch/work/p/ppalomar/neutrino_search_cnn/models_storage/{script_name}/cnn_best_model_{timestr}.keras',
        save_best_only = True,
        monitor = 'val_auc',
        mode = 'max',
        verbose = 1
    )
]

print('\n --------- Starting the training ---------')

history = model.fit(
    train_ds,
    validation_data = val_ds,
    epochs = 50,
    callbacks = callbacks,
    class_weight = class_weights_train
)

# ============================================================================================
#                                      \\ testing //
# ============================================================================================

print('\n ----------- EVALUATING WITH TEST DATA -----------')
test_loss, test_acc, test_auc = model.evaluate(test_ds)
print(f'Final accuracy with test is: {test_acc*100:.2f}%')

# ============================================================================================
#                                \\ model description file //
# ============================================================================================

# Take time it took to train model
end_time_total = time.time()

total_seconds = end_time_total - start_time_total

hours, remainder = divmod(total_seconds, 3600)
minutes, seconds = divmod(remainder, 60)
time_formatted = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"

# Take best val accuracy and auc and total epochs
best_val_acc = max(history.history['val_accuracy'])
best_val_auc = max(history.history['val_auc'])
epochs = len(history.epoch)

# Create .txt file
summary_filename = f'/afs/cern.ch/work/p/ppalomar/neutrino_search_cnn/models_storage/{script_name}/model_description_{timestr}.txt'

with open(summary_filename, 'w') as f:
    f.write(f"=========================================\n")
    f.write(f"       MODEL RUN: {timestr}\n")
    f.write(f"=========================================\n\n")

    f.write("--- FINAL RESULTS ---\n")
    f.write(f"Total time of execution:      {time_formatted}\n")
    f.write(f"Number of epochs completed:   {epochs}\n")
    f.write(f"Average time per epoch:       {total_seconds/epochs} s\n")
    f.write(f"Best Validation Acc:          {best_val_acc*100:.2f}%\n")
    f.write(f"Best Validation AUC:          {best_val_auc:.4f}\n")
    f.write(f"Test Accuracy:                {test_acc*100:.2f}%\n")
    f.write(f"Test AUC:                     {test_auc:.4f}\n\n")

    f.write("--- DATA PIPELINE CONFIG ---\n")
    f.write(f"Resolution:                                   {resolution}x{resolution}\n")
    f.write(f"Batch size:                                   {batch_size}\n")
    f.write(f"Total events (MC / BKG):                      {total_mc_events} / {total_bkg_events}\n")
    f.write(f"Splits (Train/Val/Test) constrained by MC:    {train_split} / {val_split} / {test_split}\n")
    f.write(f"Training Sampling (MC / BKG):                 {mc_train_sampling} / {bkg_train_sampling}\n")
    f.write(f"Validation Sampling (MC / BKG):               {mc_val_sampling} / {bkg_val_sampling}\n")
    f.write(f"Test Sampling (MC / BKG):                     {mc_test_sampling} / {bkg_test_sampling}\n")
    f.write(f"Class Train Weights (BKG: 0, MC: 1):          {class_weights_train}\n\n")
    
    f.write("--- MODEL ARCHITECTURE ---\n")
    # Aquesta línia màgica escriu l'arquitectura de Keras a l'arxiu txt
    model.summary(print_fn=lambda x: f.write(x + '\n'))
    
    f.write("\n--- ADDITIONAL NOTES ---\n")
    f.write(" None. \n")


# ============================================================================================
#                                   \\ benchmarking plots //
# ============================================================================================

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

fig, (ax_main, ax_auc) = plt.subplots(1, 2, figsize=(16, 6))

# ---------------------------------------------------------
# SUBPLOT 1: LOSS & ACCURACY (Left)
# ---------------------------------------------------------
color_loss = '#e41a1c' # Red
ax_main.set_xlabel('Epochs')
ax_main.set_ylabel('Loss', color=color_loss)
line1 = ax_main.plot(history.history['loss'], label='Train Loss', color=color_loss, linestyle='-', lw=2)
line2 = ax_main.plot(history.history['val_loss'], label='Val Loss', color=color_loss, linestyle='--', lw=2)
ax_main.tick_params(axis='y', labelcolor=color_loss)
ax_main.grid(True, linestyle=':', alpha=0.7)

# Twin axis for Accuracy
ax_main_2 = ax_main.twinx()  
color_acc = '#377eb8' # Blue
ax_main_2.set_ylabel('Accuracy', color=color_acc)
line3 = ax_main_2.plot(history.history['accuracy'], label='Train Accuracy', color=color_acc, linestyle='-', lw=2)
line4 = ax_main_2.plot(history.history['val_accuracy'], label='Val Accuracy', color=color_acc, linestyle='--', lw=2)
ax_main_2.tick_params(axis='y', labelcolor=color_acc)

# Combine legends
lines = line1 + line2 + line3 + line4
labels = [l.get_label() for l in lines]
ax_main.legend(lines, labels, loc='center right', frameon=True, edgecolor='black')
ax_main.set_title('Loss & Accuracy Evolution')

# ---------------------------------------------------------
# SUBPLOT 2: AUC (Right)
# ---------------------------------------------------------
color_train_auc = '#4daf4a' # Green
color_val_auc = '#ff7f00'   # Orange

ax_auc.set_xlabel('Epochs')
ax_auc.set_ylabel('AUC')
# IMPORTANT: Using 'auc' and 'val_auc' assuming tf.keras.metrics.AUC(name='auc') is set in model.compile
ax_auc.plot(history.history['auc'], label='Train AUC', color=color_train_auc, linestyle='-', lw=2.5)
ax_auc.plot(history.history['val_auc'], label='Val AUC', color=color_val_auc, linestyle='--', lw=2.5)

ax_auc.tick_params(axis='y')
ax_auc.grid(True, linestyle=':', alpha=0.7)
ax_auc.legend(loc='lower right', frameon=True, edgecolor='black')
ax_auc.set_title('AUC Evolution')

# ---------------------------------------------------------
# Save Plot
# ---------------------------------------------------------
plt.tight_layout()
plt.savefig(f'/afs/cern.ch/work/p/ppalomar/neutrino_search_cnn/models_storage/{script_name}/training_metrics_{timestr}.png')
plt.clf()

# ============================================================================================
#                                   \\ update benchmark .json //
# ============================================================================================

bm_modelList_path = '/afs/cern.ch/user/p/ppalomar/private/neutrino_search_cnn/benchmarking/json_settings/model_list.json'

if os.path.exists(bm_modelList_path):
    with open(bm_modelList_path, 'r') as f:
        model_list = json.load(f)
else:
    model_list = {"models_" + model_version[0] + "x": {}, "which_to_benchmark": ""}

if ( "models_" + model_version[0] + "x" ) not in model_list:
    model_list["models_" + model_version[0] + "x"] = {}

model_list["models_" + model_version[0] + "x"][f"{script_name}_name"] = f"cnn_best_model_{timestr}.keras"
model_list["models_" + model_version[0] + "x"][f"{script_name}_output_file"] = f"/afs/cern.ch/work/p/ppalomar/neutrino_search_cnn/models_storage/{script_name}/"
model_list["which_to_benchmark"] = model_version

with open(bm_modelList_path, 'w') as f:
    json.dump(model_list, f, indent=4)