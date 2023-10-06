# Comparison with VolPy

## Setup

Clone the CaImAn repository
```
git clone https://github.com/flatironinstitute/CaImAn.git
```
As far as we tested, VolPy (CaImAn/demos/general/demo_pipeline_voltage_imaging.py)
resulted in an error in TensorFlow when Mask R-CNN was used to detect neurons.
Our workaround is to revert CaImAn to the version prior to the VolPy (Mask R-CNN) adaptation to TensorFlow 2.4,
as TensorFlow 1.14.0 (along with Python 3.7.3) was used to train Mask R-CNN according to the VolPy paper.
To this end, we go back to the preceding version to the following merge:
Merge pull request #965 from flatironinstitute/mask_rcnn_tf_2.0 (volpy maskrcnn compatible tensorflow 2.4.1) on Apr 7, 2022.
```
cd CaImAn
git checkout 91aaec809eceaff9e26041a20c1793dfafdf137f
```
Then, create a conda environment, run setup_env.sh to install necessary packages, and install CaImAn locally.
```
conda create -n <env_name> -y python=3.7.3
conda activate <env_name>
sh ../setup_env.sh
pip install -e .
```
The reason for running setup_env.sh instead of using CaImAn/environment.yml is because, in order for this VolPy version to work, some packages must have certain versions not specified in the environment.yml.
Also, in our experimental environment, it was challenging to get TensorFlow to run on GPU by installing everything from conda-forge channel, and therefore setup_env.sh installs only a few packages from conda-forge channel.

The VolPy demo requires a certain directory structure to store data.
This can be created by running CaImAn/caimanmanager.py according to the CaImAn installation instructions.
Alternatively, one may run the following commands.
```
mkdir -p ~/caiman_data/example_movies
mkdir -p ~/caiman_data/model
```
After that, the VolPy demo can be run as follows.
```
python demos/general/demo_pipeline_voltage_imaging.py
```
This will use manually annotated neuron ROIs.
To run automatic neuron segmentation using Mask R-CNN, modify Line 166 of demo_pipeline_voltage_imaging.py as follows (by replacing 0 with 1).
```
method = methods_list[1]
```

## Run

### Run VolPy on VolPy Datasets

The datasets used in the VolPy paper can be downloaded from: https://zenodo.org/record/4515768#.Y3gVI77MLE8

To reproduce the results in the VolPy paper, run the following script after setting the parameters (the variables in capital letters) approapriately.
```
python run_volpy_on_volpy_datasets.py
```
Once done, the accuracy can be evaluated using the voltage pipeline as
```
cd ../..
conda deactivate
conda activate <voltage>
python pipeline.py params/volpy/eval_volpy_{l1,teg,hpc}.py
```

### Run VolPy on HPC2 Datasets

The HPC2 datasets can be downloaded from: TBD

Run the following script after setting the parameters.
```
python run_volpy_on_hpc2_datasets.py
```
Once done, the accuracy can be evaluated using the voltage pipeline as
```
cd ../..
conda deactivate
conda activate <voltage>
python pipeline.py params/volpy/eval_volpy_hpc2.py
```


## Train

Clone the Mask R-CNN repository
```
git clone https://github.com/matterport/Mask_RCNN.git
```
To test, run train.py after setting the parameters (the variables in capital letters) approapriately.
```
python train.py
```
This will create training and validation data under the Mask_RCNN directory by converting summary images and ground truth ROIs.
Once done, run the Mask R-CNN training as
```
cd Mask_RCNN/samples/neurons
python neurons.py train --dataset=../../datasets/neurons --weights=coco
```
One might need to remove ".." preceding "mrcnn" in the following lines in neuron.py.
```
from ..mrcnn.config import Config
from ..mrcnn import model as modellib, utils
```

### Train VolPy on VolPy Datasets

Run VolPy first to generate summary images from VolPy datasets.
```
python run_volpy_on_volpy_datasets.py
```
Then, run the following script after setting the parameters.
```
python train_volpy_on_volpy_datasets.py
```
VALIDATION_RATIO is set to 3 to reproduce the stratified 3-fold cross-validation in the VolPy paper.
Once done, the validation accuracy can be evaluated in the same way as above.
```
cd ../..
conda deactivate
conda activate <voltage>
python pipeline.py params/volpy/eval_volpy_{l1,teg,hpc}.py
```

### Train VolPy on HPC2 Datasets

Run VolPy first to generate summary images from HPC2 datasets.
```
python run_volpy_on_hpc2_datasets.py
```
Then, run the following script after setting the parameters.
```
python train_volpy_on_hpc2_datasets.py
```
VALIDATION_RATIO is set to 13 to perform leave-one-out cross-validation.
Once done, the validation accuracy can be evaluated in the same way as above.
```
cd ../..
conda deactivate
conda activate <voltage>
python pipeline.py params/volpy/eval_volpy_hpc2.py
```
