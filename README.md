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
For a video with moderate resolution (a few hundred pixels in width and height),
Voltage runs in real time on a single high-end desktop computer equipped with GPUs and SSDs,
meaning that the processing speed is faster than the image recording rate (e.g., hundreds of frames per second).



## Computational Environment

Voltage has been tested on the computational environment summarized in the following table.

| Component | Specification |
|-----------|---------------|
| CPU       | AMD Ryzen Threadripper 3960X (24 cores)  |
| RAM       | DDR 4 3200 MHz, 192 GB (6 ch. x 32 GB)   |
| GPU       | 2 of NVIDIA GeForce RTX 2080 Ti          |
| SSD       | KIOXIA EXCERIA PRO 2 TB (PCIe Gen 4 x4)  |
| OS        | Ubuntu 20.04.6 LTS, Linux kernel 5.15    |
| Software  | NVIDIA Driver 525.105.17, CUDA 12.0      |
|           | Python 3.8, TensorFlow 2.4.1., g++ 9.4.0 |

In terms of hardware,
this is not necessarily a requirement, but at least one GPU and NVMe SSD will be needed
to gain processing speed Voltage is designed for.

In terms of software,
Voltage requires Linux and other pieces of software in the table.
Versions are for reference only: other versions may also work.



## How To Run


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

Edit `params/paths.py` so the pipeline knows where to read input data and write output data.
See the comments in `paths.py`, and download test datasets as necessary.


### Test-run 

The following commands run the pipeline on test datasets.
```
python pipeline.py params/run_l1.py
python pipeline.py params/run_teg.py
python pipeline.py params/run_hpc.py
python pipeline.py params/run_hpc2.py
```
Each line takes a preset parameter file named `run_<group>.py`,
and runs the pipeline on all of the test datasets in that dataset group.
Results will be written under `OUTPUT_BASE_PATH/results/voltage_<group>/`,
where `OUTPUT_BASE_PATH` is the path specified in `params/paths.py`.
A summary of the results can be found in `all.html`.


### Run on new data

To run the pipeline on new videos, create a parameter file (say, `params/new_param.py`)
by following the examples `run_<group>.py` above, and run:
```
python pipeline.py params/new_param.py
```
See `params/defaults.py` for the explanations of parameters.
The values in `params/defaults.py` will be used by default, and those in a parameter file
passed to the pipeline (`new_param.py` as in the above command) will override them.
The preset parameter files `run_<group>.py` set path variables by referring to the base paths
specified in `params/paths.py`, but that is not a requirement: one can also directly set
variables such as `INPUT_DIR` in a parameter file if that is more convenient.


### Train a segmentation model

To train the U-Net cell segmentation network, run:
```
python pipeline.py params/train.py
```
This will create synthetic data and train the network.
The trained U-Net models will be stored under `TRAIN_BASE_PATH/train/model/`
with names `model_e<epoch_count>_v<validation_loss>.h5`.
In most cases, one should choose the model with the smallest validation loss.
