# Voltage Imaging Data Processing



## Directory Structure
* lib   : preprocessing (motion correction and U-Net preprocessing)

## Creating conda environment

Follow below steps to create the conda environment required to run the pipeline

```bash
# Create conda environment
conda create -n VI_base python=3.6

# Activate the conda environment
codna activate VI_base

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
conda install -y -q jupyter
```

## How to Run

#### Build all the required libraries
```bash
bash build.sh
```

#### Running the pipeline

Make sure to update the `settings.txt` (global settings) and `file_params.txt` (file-specific parameters) accordingly.

To execute in file mode:
```bash
python run_pipeline.py --file <tag>

#Example
python run_pipeline.py --file 018
```

To execute in batch mode:

```bash
python run_pipeline.py --batch
```

Optional parameters:
```bash
# Optional Paramters:
# --new-project (To create new execution log files and discard any previous timing information)
# --motion-correct-only (To run only motion correction on all files)
# --evaluate (If ground truth is available, we can evalute the performance using this paramter)
```

The pipeline script requires at least 1 GPU and will map all available GPUs.

The results will be stored as per the paths set in `settings.txt` and `file_params.txt`.
