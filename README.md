# Voltage Imaging Data Processing



## Directory Structure
* lib   : preprocessing (motion correction and U-Net preprocessing)

## Creating conda environment

Follow below steps to create the conda environment required to run the pipeline

```bash
# Create conda environment
conda create -n VI_base python=3.6

# Install sklearn
conda install -y -q scikit-learn

# Install Tensorflow GPU version 1.13
conda install -y -q tensorflow-gpu=1.13.1
# Install Keras
conda install -y -q keras

# Install TIFF library
pip install tifffile  --progress-bar off

# Install Read ROI library
pip install read_roi  --progress-bar off

# Install ray
pip install -U ray --progress-bar off

# Install prettytable
pip install PTable --progress-bar off

# Install OpenCV
pip install opencv-python --progress-bar off

# Install skimage
conda install -y -q scikit-image

# Install numba
conda install -y -q numba

# Install NVGPU
pip install nvgpu --progress-bar off

# Install cython
conda install -y -q -c anaconda cython

# Install Jupyter Notebook
conda install -c conda-forge notebook
```

## How to Run

#### Build the pre-processing library
```bash
bash build.sh
```

#### Running the pipeline

Make sure to update the `file_params.txt` accordingly.

To execute in file mode:
```bash
python pipeline --file <tag>

#Example
python pipeline --file 018
```

To execute in batch mode:

```bash
python pipeline --batch
```
The results will be stored as per the paths set in `file_params.txt`.
