# Comparison with VolPy

## Setup

Clone the repository
```
git clone https://github.com/flatironinstitute/CaImAn.git
```
As far as we tested, VolPy (CaImAn/demos/general/demo_pipeline_voltage_imaging.py)
resulted in an error in TensorFlow when Mask R-CNN was used to detect neurons.
Our workaround is to revert CaImAn to the version before VolPy (Mask R-CNN) adaptation to TensorFlow 2.4,
as TensorFlow 1.14.0 (along with Python 3.7.3) was used to train Mask R-CNN according to the VolPy paper.
To this end, we go back to the preceding version to the following merge:
Merge pull request #965 from flatironinstitute/mask_rcnn_tf_2.0 (volpy maskrcnn compatible tensorflow 2.4.1) on Apr 7, 2022.
```
cd CaImAn
git checkout 91aaec809eceaff9e26041a20c1793dfafdf137f
```
Then, create a conda environment, install CaImAn locally, and use setup_env.sh to install necessary packages.
```
conda create -n <env_name> -c conda-forge -y python=3.7.3
conda activate <env_name>
pip install -e .
sh setup_env.sh
```
In order for this VolPy version to work, some packages must have certain versions not specified in CaImAn/environment.yml. Some of them are:
* h5py must be of version 2.x instead of 3.x
* If the version string of tifffile contains more than three numbers (e.g., X.Y.Z.W), CaImAn/caiman/base/timeseries.py causes an error.

CaImAn/demos/general/demo_pipeline_voltage_imaging.py requires a certain directory structure to store data.
This can be created by running CaImAn/caimanmanager.py according to the CaImAn installation instructions.
Alternatively, you can run the following commands.
```
mkdir -p ~/caiman_data/example_movies
mkdir -p ~/caiman_data/model
```
After that, one should be able to run
```
python demos/general/demo_pipeline_voltage_imaging.py
```
This uses manually annotated ROIs, and modifying Line 166 as follows (by replacing 0 by 1)
```
method = methods_list[1]
```
uses Mask R-CNN for neuron segmentation.
