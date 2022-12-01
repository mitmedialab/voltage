#!/bin/sh

conda install -y mkl-devel
conda install -y cvxopt
conda install -y -c conda-forge cvxpy
conda install -y -c menpo opencv3
conda install -y jupyter matplotlib
conda install -y scikit-image scikit-learn
conda install -y cython
conda install -y -c pytorch pytorch

