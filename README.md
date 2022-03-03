# Voltage Imaging Data Processing


## Directory Structure

* correct/     : motion/shading correction
* preproc/     : preliminary feature extraction
* segment/     : segmentation of active cell regions
* demix/       : demixing of overlapping cells
* simulate/    : synthetic data generation for training U-Net segmentation
* evaluate/    : accuracy evaluation by comparison with ground truth
* utils/       : C++ utilities used by some of the above modules
* params/      : parameter files for pipeline script
* pipeline.py  : pipeline script


## Environment Setup

$ conda create -n voltage python=3.8  
$ conda activate voltage  
$ conda install tiffile  
$ conda install scikit-image  
$ conda install keras  
$ conda install tensorflow-gpu  
$ conda install ipykernel  
$ conda install nbconvert  
$ conda install pandas  
$ conda install cython  
$ pip install elasticdeform  
$ pip install read-roi


## How To Run

Build and install C++ Cython modules:

$ make

To train the U-Net cell segmentation network, edit params/train.py (path strings in particular) and run:

$ python pipeline.py params/train.py

This will create synthetic data and train the network. The trained U-Net model will be stored in MODEL_DIR.

To run the pipeline for real data sets using the trained model, edit params/run_xxx.py (choose one of the preconfigured parameter files or create your own) and run:

$ python pipeline.py params/run_xxx.py
