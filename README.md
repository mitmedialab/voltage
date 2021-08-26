# Voltage Imaging Data Processing


## Directory Structure

* preproc/     : preprocessing (motion/shading correction and spike extraction)
* segment/     : segmentation of active cells
* demix/       : demixing of individual cells
* simulation/  : synthetic data generation for training U-Net segmentation
* evaluate/    : accuracy evaluation by comparing the results with ground truth
* lib/         : libraries
* pipeline.py  : pipeline script


## Environment Setup

$ conda create -n voltage python=3.8
$ conda activate voltage
$ conda install tiffile
$ conda install scipy
$ conda install scikit-image
$ conda install keras
$ conda install tensorflow-gpu
$ conda install nbformat
$ conda install nbconvert
$ conda install pandas
$ pip install elasticdeform


## Building Libraries

$ cd lib/tiff-4.1.0_mod  
$ ./configure  
$ make  
$ cd ../..
$ make  

Building tiff-4.1.0_mod allows much faster saving of multipage tiffs, but one could also use standard libtiff instead.


## How To Run

Set the parameters at the top of pipeline.py appropriately, the path strings in particular.

To train the U-Net cell segmentation network, set mode = 'train' in pipeline.py and run

$ python pipeline.py

This will create synthetic data and train the network. The trained U-Net model will be stored in MODEL_PATH. 
To run the pipeline for real data sets using the trained model, set mode = 'run' and run the pipeline.py.

