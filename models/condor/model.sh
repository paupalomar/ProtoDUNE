#!/bin/bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_104cuda/x86_64-el9-gcc11-opt/setup.sh
cd /afs/cern.ch/user/p/ppalomar/private/neutrino_search_cnn/models/python/

python model_22.py --config ../json_settings/config_22.json
