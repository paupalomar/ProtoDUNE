import uproot
import math
import os
import argparse
import json

parser = argparse.ArgumentParser(description='Submitting the data cooking script jobs.')
parser.add_argument('--config', type=str, required=True, help='Path to json file.')
args = parser.parse_args()

with open(args.config, 'r') as f:
    config = json.load(f)

mc_or_bkg = "mc" if config["input_info"]["is_mc"] else "bkg"

root_files = []
for i in range(config["input_info"]["num_" + mc_or_bkg + "_files"]):
    root_files.append(config["input_files"]["path_" + mc_or_bkg + f"_{i+1}"])

events_per_job = config["input_info"]["events_per_job"]
tree_image = "ana/tree_image" 
submitter_args = f'../condor/submitter_{mc_or_bkg}_args.txt'

chunk_id = 0

with open(submitter_args, 'w') as f:
    for file in root_files:
        with uproot.open(file) as rf:
            tree = rf[tree_image]
            num_events = tree.num_entries
            num_jobs = math.ceil(num_events / events_per_job) 
        for i in range(num_jobs):
            start = i * events_per_job
            end = min((i+1) * events_per_job, num_events)

            f.write(f'{file} {start} {end} {chunk_id}\n')
            chunk_id += 1

os.chdir('/afs/cern.ch/user/p/ppalomar/private/neutrino_search_cnn/data_prep/condor/')
os.system('condor_submit data_root_to_tf.sub')