# Comparison with Invivo-Imaging (SGPMD-NMF)

## Setup

Create a conda environment
```
conda create -n invivo python=3.6
conda activate invivo
sh ./setup_env.sh
```

Install TreFiDe (Trend Filter Denoising)
```
git clone https://github.com/ikinsella/trefide.git
cd trefide
git checkout 14dbb21a0cb8a2276ef1616208ac6f69cbb9fc77
cp ../preprocess.py trefide/
make
pip install .
cd ..
```
The third line above is to match the TreFiDe version to the one we tested. This may not be necessary and the latest version may also work. 
The fourth line is a workaround for [this issue](https://github.com/adamcohenlab/invivo-imaging/issues/4), where trefide/trefide/preprocess.py needs to be replaced by [this](https://github.com/m-xie/trefide/blob/master/trefide/preprocess.py), whose copy is included in this directory.

Install Invivo-Imaging (SGPMD-NMF)
```
git clone https://github.com/adamcohenlab/invivo-imaging.git
cd invivo-imaging
git checkout 985c610d667d5cab1b169e2e38e5b50e46b00801
cd ..
```
Again, the thrid line above may not be necessary.

Line 111 of invivo-imageing/denoise/denoise.py needs to be modified as follows (i.e., add "dtype=int"):
```
disc_idx = np.array([], dtype=int)
```
This is to keep recent versions of numpy.delete(*arr*, *obj*) from causing an error for float *obj* (even if it is an empty array). 

## Test Run

Denoise and extract cell components (demix) from the test image (invivo-imaging/demo_data/raw_data.tif).

### Denoise
MATLAB is required. Launch MATLAB, navigate to invivo-imaging/denoise, and run main.m.
If the conda environment name set up above is different from "invivo," edit main.m and replace the name in the line containing "source activate invivo."

### Demix

Launch Jupyter Notebook and run all the cells of invivo-imaging/demix/main.ipynb.
```
cd invivo-imaging/demix
jupyter notebook main.ipynb
```

### Run the entire pipeline in a single command
For convenience, a Python script is provided that runs everything (except for visualization of results). MATLAB is still required.
```
python main.py
```

## Run on HPC2 Datasets
By running the following script, main.py mentioned above will run on all of the HPC2 datasets. This can take tens of hours.
```
python run_invivo_on_hpc2_datasets.py
```
