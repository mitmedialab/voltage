# Voltage Imaging Data Processing



## Directory Structure
* lib   : preprocessing (motion correction and U-Net preprocessing)

## Creating conda environment

Follow below steps to create the conda environment required to run the pipeline

```bash
# Create conda environment
conda create -n voltage python=3.8

# Activate the conda environment
codna activate voltage

# Install sklearn
conda install -y -q scikit-learn

# Install Tensorflow GPU
conda install -y -q tensorflow-gpu

# Install Keras
conda install -y -q keras

# Install ND2 library
pip install nd2reader  --progress-bar off

# Install TIFF library
pip install tifffile  --progress-bar off

# Install Read ROI library
pip install read_roi  --progress-bar off

# Install ray
pip install -U ray --progress-bar off
Note: As of May 11, 2021, the latest ray 1.3 has some problems, so please install ray 1.2 using
pip install -U ray==1.2 --progress-bar off

# Install prettytable
pip install PTable --progress-bar off

# Install OpenCV
pip install opencv-python --progress-bar off

# Install NVGPU as nvidia-smi python interface and psutil
pip install nvgpu --progress-bar off
pip install psutil --progress-bar off

# Install Webcolors package
pip install webcolors --progress-bar off

# Install DASH components
pip install PTable plotly dash dash-core-components dash_html_components dash_table --progress-bar off

# Install elastic deformation library
pip install elasticdeform --progress-bar off

# Install skimage
conda install -y -q scikit-image

# Install numba
conda install -y -q numba

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

To execute in batch mode:

```bash
python run_pipeline.py --batch --(temporal|spatial|mix)

--temporal: Use temporal inference for 40x, 20x and 16x magnification
--spatial: Use spatial inference for 40x, 20x and 16x magnification
--mix: Use spatial inference for 40x and 20x, and, temporal inference for 16x datasets.
```

Optional parameters:
```bash
# Optional Paramters:
# --motion-correct-only (To run only motion correction on all files)
# --evaluate (If ground truth is available, we can evalute the performance using this paramter)
```

The pipeline script requires at least 1 GPU and will map all available GPUs for certain functionalities.

The results will be stored as per the paths set in `settings.txt` and `file_params.txt`.


## How to add newly created datasets

- Update the dataset information in `file_params.txt`
- In `settings.txt` update the following below directory paths
- Update the input and output directory path for the dataset, `input_base_path` and `output_base_path`, the input images is expected to be in uint16 format normalized to 0-65535 
- Update `individual_gt_dir` and `individual_gt_ids` with the names of the sub-directories that will contain the individual ground truth files in .roi/.zip format
- Update `consensus_gt_dir` with the directory of the consensus ground truth directory, the consensus ground truth images will be 3-Dimensional uint8 images with either 0 or 255 values
- Update `summary_image_dir` with the Maximum Intensity Projection of the Motion Corrected images from a previous pipeline run, this is used for the evaluation notebooks and is expected to be a 2-Dimensional float32 image




