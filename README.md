# CoNeMOS

In this repository, we CoNeMOS (Conditional Network for Multi-Organ Segmentation) , a framework that leverages a 
label-conditioned network for synergistic learning on partially labelled region-based segmentations. Conditioning is 
achieved by combining convolutions with expressive Feature-wise Linear Modulation (FiLM) layers, whose parameters are 
controlled by an auxiliary network.

In contrast to other conditioning methods, FiLM layers are stable to train and add negligible computation overhead, 
which enables us to condition the entire network.

This repository contains all the code necessary to train your own model. The script used to train our model is 
available [here](scripts/training.py).

----------------

### Installation

1. Clone this repository.

2. Create a virtual environment (i.e., with pip or conda) and install all the required packages. \
These depend on your python version, and we list them [here](requirements.txt) for python 3.10.

3. If you wish to run on the GPU, you will also need to install Cuda. Note that if you used conda, these should have 
already been automatically installed.


----------------

### Citation/Contact

If you have any question regarding the usage of this code, or any suggestions to improve it, please raise an issue 
(preferred) or contact us at: bbillot@mit.edu
