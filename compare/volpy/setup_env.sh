#!/bin/sh

MODULES="
tensorflow=1.14.0
cython
future
holoviews
ipyparallel
matplotlib
opencv
peakutils
pynwb
scikit-image
scikit-learn
pims
keras=2.3.1
tifffile=2020.6.3
jupyter_client=7.4.2
h5py=2.10.0
numpy=1.16.5
"

for mm in ${MODULES}
do
    echo ${mm}
    conda install -y ${mm} -c conda-forge
done

