# Comparison with Invivo-Imaging (SGPMD-NMF)

## Setup

Create a conda environment
```
conda create -n invivo python=3.6
conda activate invivo
sh ./setup_env.sh
```
There is a line of code in Invivo-Imaging that assumes the environment name to be "invivo." A different name may be used, in which case that line needs to be modified (see below).

Install TreFiDe (Trend Filter Denoising)
```
git clone https://github.com/ikinsella/trefide.git
cp preprocess.py trefide/trefide/
cd trefide
make
pip install .
```
Following [this issue](https://github.com/adamcohenlab/invivo-imaging/issues/4), trefide/trefide/preprocess.py needs to be replaced by [this](https://github.com/m-xie/trefide/blob/master/trefide/preprocess.py), whose copy is included in this directory.

Install Invivo-Imaging (SGPMD-NMF)
```
git clone https://github.com/adamcohenlab/invivo-imaging.git
```
Modify Line 111 of invivo-imageing/denoise/denoise.py as follows (i.e., add dtype):
```
disc_idx = np.array([], dtype=int)
```
This is to keep recent versions of numpy.delete(*arr*, *obj*) from causing an error for float *obj* (even if it is an empty array). 

## Test Run

Denoise and extract cell components (demix) from the test image (invivo-imaging/demo_data/raw_data.tif).

### Denoise
MATLAB is required. Launch MATLAB, navigate to invivo-imaging/denoise, and run main.m. Or in command lines,
```
cd invivo-imaging/denoise
matlab -nodisplay -nosplash -nodesktop -r "run('main.m'); exit;"
```
If the conda environment name set up above is different from "invivo," edit main.m and replace the name in the line containing "source activate invivo."

### Demix

Launch Jupyter Notebook and run all the cells of invivo-imaging/demix/main.ipynb.
```
cd invivo-imaging/demix
jupyter notebook main.ipynb
```
