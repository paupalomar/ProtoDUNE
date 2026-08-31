import numpy as np
import tensorflow as tf
import os
import sys
import uproot
import argparse
import json

# ====================================================================================
#                                \\ initial parameters //
# ====================================================================================

# PARSE ARGUMENTS

parser = argparse.ArgumentParser(description='Cooking with some data. Turning ROOT trees into numpy arrays and storing them.')
parser.add_argument('--config', type=str, required=True, help='Path to json file.')
parser.add_argument('--input_file', type=str, required=True, help='Path to the .root file')
parser.add_argument('--start', type=int, required=True, help='Start event to convert')
parser.add_argument('--end', type=int, required=True, help='End event to convert')
parser.add_argument('--chunk_id', type=int, required=True, help='Global ID of the event batch')

args = parser.parse_args()

with open(args.config, 'r') as f:
    config = json.load(f)

# INITIAL PARAMS

# params for processing
start = args.start 
end = args.end
chunk_id = args.chunk_id
resolution = config["output_info"]["resolution"]
is_mc = config["input_info"]["is_mc"]
mc_or_bkg = "mc" if is_mc else "bkg"
step = config["input_info"]["steps_per_batch"]

if is_mc:
    apply_ta_cut = config["mc_info"]["apply_ta_cut"]
else:
    take_spill_on = config["bkg_info"]["take_spill_on"]
    spill_on_inverse_cut = config["bkg_info"]["spill_on_inverse_cut"]

features_aggregate = [
    feature for feature, active in config["features_to_extract_from_trees"]["aggregate"].items() if active
]

features_true = [
    feature for feature, active in config["features_to_extract_from_trees"]["true"].items() if active
]

# paths
path_root_file = args.input_file
if not is_mc:
    if not take_spill_on:
        path_output = config["output_paths"][mc_or_bkg + "_out_folder"] #MAKE SURE THIS HAS THE / AT THE END!!!
        file_output = f"{path_output}{mc_or_bkg}_data_{resolution}_{chunk_id:03d}.tfrecord"
    elif not spill_on_inverse_cut:
        path_output = config["output_paths"]["bkg_out_spillON_noCut"]
        file_output = f"{path_output}{mc_or_bkg}_data_{resolution}_spillOn_NoCut_{chunk_id:03d}.tfrecord"
    elif spill_on_inverse_cut:
        path_output = config["output_paths"]["bkg_out_spillON_inverseCut"]
        file_output = f"{path_output}{mc_or_bkg}_data_{resolution}_spillOn_inverseCut_{chunk_id:03d}.tfrecord"
else:
    path_output = config["output_paths"][mc_or_bkg + "_out_folder"] #MAKE SURE THIS HAS THE / AT THE END!!!
    file_output = f"{path_output}{mc_or_bkg}_data_{resolution}_{chunk_id:03d}.tfrecord"


# ====================================================================================
#                                 \\ file reformating //
# ====================================================================================

print(f"======================================================")
print(f" JOB ID: {chunk_id:03d}")
print(f" INPUT:  {path_root_file}")
print(f" OUTPUT: {file_output}")
print(f" EVENTS: From {start} to {end} (Max to read: {end - start})")
print(f" RESOLUTION: {resolution}x{resolution}")
print(f"======================================================\n")

# Some functions to make it easier to write

def _bytes_feature(value):
    """
    Takes a value in bytes and turns it into a TF Feature.
    """
    return tf.train.Feature(bytes_list = tf.train.BytesList(value = [value]))

def _int64_feature(value):
    """
    Takes a value integer and turns it into a TF Feature.
    """
    return tf.train.Feature(int64_list = tf.train.Int64List(value = [value]))

def _float_feature(value):
    """
    Takes a value float and turns it into a TF Feature
    """
    return tf.train.Feature(float_list = tf.train.FloatList(value = [value]))

# Event to example converter

def event_to_example(image_dict, scalar_dict, is_mc, resolution):
    """
    Takes a diccionary of all 12 planes images for an event and all the data associated with it as another dictionary. 
    It packs everything in one TFRecord Example.
    """

    label = 1 if is_mc else 0

    feature = {
        'label': _int64_feature(label),
        }

    # Assume every event has the same shape
    feature['res'] = _int64_feature(resolution)

    # Save the image as a feature given the key (which is 'imageU1', for example)
    for key, img in image_dict.items():
        feature[key] = _bytes_feature(img.astype(np.float32).tobytes())

    for key, val in scalar_dict.items():
        feature[key] = _float_feature(float(val))

    return tf.train.Example(features = tf.train.Features(feature=feature))

def write_new_file(path_root_file, file_output, options, is_mc, resolution):
    """
    This converts a .root file into a .tfrecord file and stores it.
    """
    with uproot.open(path_root_file) as f, tf.io.TFRecordWriter(file_output, options=options) as writer:

        #   MONTECARLO CONVERTER
        if is_mc:
            print('ROOT file is MC.\n')
            if apply_ta_cut:
                print('Applying TA cut.\n')
            else:
                print('Not applying TA cut\n')

            tree_image=f['ana/tree_image']
            tree_truth=f['ana/tree_truth']
            tree_agg=f['ana/tree_aggregate']

            image_keys = tree_image.keys()

            agg_data = tree_agg.arrays(features_aggregate, entry_start=start, entry_stop=end, library='np') # this is a dictionary of arrays like {'eventID': [1, ..., 500], 'vertexX': [3, -1,3, ...], ...}
            truth_data = tree_truth.arrays(features_true, entry_start=start, entry_stop=end, library='np')

            ev_idx = 0
            ev_counter = 0
            for batch in tree_image.iterate(image_keys, step_size=step, library='np', entry_start=start, entry_stop=end):
                num_events = len(batch[image_keys[0]])
                
                for i in range(num_events):
                    
                    if apply_ta_cut:
                        ta_flag = truth_data['triggerActivityFlag'][ev_idx]
                        if ta_flag !=1:
                            ev_idx += 1
                            continue

                    image_dict = {}
                    scalar_dict = {}

                    for key in image_keys:
                        img = np.array(batch[key][i].tolist(), dtype = np.float32).reshape(500,500)
                        img_reshaped = img.reshape(resolution,int(500/resolution),resolution,int(500/resolution))
                        img_reduced = img_reshaped.sum(axis=(1,3))
                        image_dict[key] = img_reduced

                    for key in features_aggregate:
                        val = agg_data[key][ev_idx]
                        scalar_dict[key] = val

                    for key in features_true:
                        # Change name if they have the same name in agg and true
                        true_val = truth_data[key][ev_idx]
                        if key in scalar_dict:
                            scalar_dict[f"true_{key}"] = true_val
                        else:
                            scalar_dict[key] = true_val

                    example = event_to_example(image_dict, scalar_dict, is_mc=is_mc, resolution=resolution)
                    writer.write(example.SerializeToString())
                    ev_counter += 1
                    ev_idx += 1
            total_read = end - start
            discarded = total_read - ev_counter
            print(f"\n--- SUMMARY ---")
            print(f"Total events read: {total_read}")
            print(f"Events discarded by cuts: {discarded}")
            print(f"Events successfully written: {ev_counter}")

            return ev_counter
        #   BACKGROUND CONVERTER
        else:
            print('ROOT file is BKG.\n')
            if take_spill_on:
                if spill_on_inverse_cut:
                    # Here the code to also take the spill on bkg data to get the bkg's from the beam
                    # should be written. The idea is to take spill on data but with the inverse cuts applied from
                    # Dario's cuts script, so that I mostly eliminate neutrinos and get just beam backgrounds.
                    print('\nCode for BKG spill-on with inverse cuts applied is not ready yet! Turn it off in config.json\n')
                    ev_counter = 0
                    return ev_counter
                else:
                    print('Taking just spill-on data.\n')

                    tree_image=f['ana/tree_image']
                    tree_agg=f['ana/tree_aggregate']

                    image_keys = tree_image.keys()

                    agg_data = tree_agg.arrays(features_aggregate, entry_start=start, entry_stop=end, library='np')

                    ev_idx = 0
                    ev_counter = 0
                    for batch in tree_image.iterate(image_keys, step_size=step, library='np', entry_start=start, entry_stop=end):
                        num_events = len(batch[image_keys[0]])
                        
                        for i in range(num_events):
                            # Make sure we just take data from Spill ON
                            spill_status_flag = agg_data['spillStatusFlag'][ev_idx]
                            if spill_status_flag !=1:
                                ev_idx += 1
                                continue
    
                            image_dict = {}
                            scalar_dict = {}
        
                            for key in image_keys:
                                img = np.array(batch[key][i].tolist(), dtype = np.float32).reshape(500,500)
                                img_reshaped = img.reshape(resolution,int(500/resolution),resolution,int(500/resolution))
                                img_reduced = img_reshaped.sum(axis=(1,3))
                                image_dict[key] = img_reduced
        
                            for key in features_aggregate:
                                val = agg_data[key][ev_idx]
                                scalar_dict[key] = val
    
                            for key in features_true:
                            # In order for both mc and bkg datasets to have the same features, we set the truth features to -99999.0 in the bkg DS.
                                if key in scalar_dict:
                                    scalar_dict[f"true_{key}"] = -99999.0
                                else:
                                    scalar_dict[key] = -99999.0
                            
                            example = event_to_example(image_dict, scalar_dict, is_mc=is_mc, resolution=resolution)
                            writer.write(example.SerializeToString())
        
                            ev_idx += 1
                            ev_counter += 1
                    total_read = end - start
                    discarded = total_read - ev_counter
                    print(f"\n--- SUMMARY ---")
                    print(f"Total events read: {total_read}")
                    print(f"Events discarded by cuts: {discarded}")
                    print(f"Events successfully written: {ev_counter}")

                    return ev_counter
            else:
                print('Taking just spill-off data.\n')
                tree_image=f['ana/tree_image']
                tree_agg=f['ana/tree_aggregate']
    
                image_keys = tree_image.keys()
    
                agg_data = tree_agg.arrays(features_aggregate, entry_start=start, entry_stop=end, library='np') # this is a dictionary of arrays like {'eventID': [1, ..., 500], 'vertexX': [3, -1,3, ...], ...}
    
                ev_idx = 0
                ev_counter = 0
                for batch in tree_image.iterate(image_keys, step_size=step, library='np', entry_start=start, entry_stop=end):
                    num_events = len(batch[image_keys[0]])
                    
                    for i in range(num_events):
                        # Make sure we just take data from Spill OFF
                        spill_status_flag = agg_data['spillStatusFlag'][ev_idx]
                        if spill_status_flag !=0:
                            ev_idx += 1
                            continue

                        image_dict = {}
                        scalar_dict = {}
    
                        for key in image_keys:
                            img = np.array(batch[key][i].tolist(), dtype = np.float32).reshape(500,500)
                            img_reshaped = img.reshape(resolution,int(500/resolution),resolution,int(500/resolution))
                            img_reduced = img_reshaped.sum(axis=(1,3))
                            image_dict[key] = img_reduced
    
                        for key in features_aggregate:
                            val = agg_data[key][ev_idx]
                            scalar_dict[key] = val

                        for key in features_true:
                        # In order for both mc and bkg datasets to have the same features, we set the truth features to -99999.0 in the bkg DS.
                            if key in scalar_dict:
                                scalar_dict[f"true_{key}"] = -99999.0
                            else:
                                scalar_dict[key] = -99999.0

                        example = event_to_example(image_dict, scalar_dict, is_mc=is_mc, resolution=resolution)
                        writer.write(example.SerializeToString())
    
                        ev_idx += 1
                        ev_counter += 1
                total_read = end - start
                discarded = total_read - ev_counter
                print(f"\n--- SUMMARY ---")
                print(f"Total events read: {total_read}")
                print(f"Events discarded by cuts: {discarded}")
                print(f"Events successfully written: {ev_counter}")

                return ev_counter


# We compress the file in gzip
options = tf.io.TFRecordOptions(compression_type="GZIP")

# Make sure the output folder exists
os.makedirs(os.path.dirname(path_output), exist_ok=True)

final_events_count = write_new_file(path_root_file=path_root_file, file_output=file_output, options=options, is_mc=is_mc, resolution=resolution)

if final_events_count != 0:
    new_file_output = file_output.replace('.tfrecord', f'_{final_events_count}.tfrecord')
    os.rename(file_output, new_file_output)
else:
    print('No events present.')

print(f"\n[OK] Job {chunk_id:03d} finished successfully.")
