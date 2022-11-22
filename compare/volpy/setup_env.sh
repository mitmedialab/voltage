#!/bin/sh

MODULES_DEFAULTS="
tensorflow-gpu=1.14.0
numpy=1.16.5
scikit-image=0.16.2
scikit-learn
cython
future
holoviews
ipyparallel
pims
"

MODULES_CONDA_FORGE="
opencv=4.1.1
pynwb
keras=2.2.5
imgaug=0.3.0
"

for mm in ${MODULES_DEFAULTS}
do
    echo ${mm}
    conda install -y ${mm} -c defaults
done

for mm in ${MODULES_CONDA_FORGE}
do
    echo ${mm}
    conda install -y ${mm} -c conda-forge
done

pip install scikit-cuda

