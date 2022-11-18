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

Run main.py after setting the parameters (the variables in capital letters) approapriately.
```
python main.py
```
The VolPy datasets can be downloaded from: https://zenodo.org/record/4515768#.Y3gVI77MLE8

## Train

Clone the Mask R-CNN repository
```
git clone https://github.com/matterport/Mask_RCNN.git
```
Run train.py after setting the parameters (the variables in capital letters) approapriately.
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
