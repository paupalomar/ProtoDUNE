# ProtoDUNE-HD LArTPC: Neutrino Event Classification using CNNs

This repository contains the core Deep Learning and data processing scripts developed during my summer research internship at the CERN Neutrino Group (July-August 2026).

## Overview
The goal of this project was to classify rare neutrino interactions in the ProtoDUNE-HD Liquid Argon Time Projection Chamber (LArTPC) using Convolutional Neural Networks (CNNs).

## Repository Structure
* **`data_prep/data_root_to_tensorflow.py`**: Pipeline to process raw `.root` files from the detector simulation into TensorFlow-ready tensors.
* **`data_prep/submitter.py` & `condor/`**: Scripts used to submit distributed data processing and model training jobs to the CERN HTCondor grid.
* **`models/model_21.py`**: The final CNN architecture (implementing contrastive learning/early fusion techniques) used to evaluate topological features of neutrino events.
* **`benchmarking/`**: Evaluation and plotting scripts to compute signal stability, purity, and compute classical metrics against the model's predictions.

## Technical Stack
* Python, TensorFlow/Keras
* ROOT/PyROOT
* HTCondor (High Performance Computing)
