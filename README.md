# Voltage Imaging Data Processing

This is a data processing pipeline for voltage imaging data (called "Voltage" for short).
It takes as input a 2D microscopy video caputuring neurons expressing time-varying fluorescence
depending on their membrane potential, and outputs voltage traces of individual neurons.

The Voltage pipeline consists of three stages.
1. **Motion correction** to cancel 2D translational motion of neurons relative to the microscope.  
1. **Neuron segmentation** to localize and delineate neuron boundaries from the background.
1. **Voltage trace extraction** to reconstruct time-varying membrane potentials of neurons.

The focus of Voltage is to accelerate the first two stages in order to quickly identify
region-of-interest (ROI) masks of firing neurons in the video.
For a video with a moderate resolution (a few hundred pixels in width and height),
Voltage works in real time on a single high-end desktop computer equipped with GPUs and SSDs,
meaning that the processing speed is faster than the image recording rate (e.g., hundreds of frames per second).



## How To Run

### Computational environment

### Set up a Python environment

Run the following commands to set up a Python virtual environment and install necessary packages.
```
conda env create -f environment.yml
conda activate voltage
```
Edit environment.yml if you prefer a different environment name to "voltage."


### Build modules

Voltage uses modules written in C++ and CUDA. To build and install them, run:
```
make
```


### Set paths

Edit params/paths.py so the pipeline knows where to read input data and write output data.
See the comments in paths.py, and download test datasets as necessary.


### Test run 

The following commands run the pipeline on test datasets.

```
python pipeline.py params/run_l1.py
python pipeline.py params/run_teg.py
python pipeline.py params/run_hpc.py
python pipeline.py params/run_hpc2.py
```

### Train a segmentation model

To train the U-Net cell segmentation network, run:
```
python pipeline.py params/train.py
```
This will create synthetic data and train the network. The trained U-Net model will be stored in TRAIN_BASE_PATH/train/model.
