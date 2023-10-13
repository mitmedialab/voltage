# Path to a directory containing copies of the datasets used in the VolPy paper,
# which can be downloaded from: https://zenodo.org/record/4515768#.Y3gVI77MLE8
# Download voltage_HPC.zip, voltage_L1.zip, and voltage_TEG.zip.
# Unzipped files should be placed as follows:
#     <Directory specified by VOLPY_DATASETS>/
#           |
#           +-- voltage_HPC/
#           |       |
#           |       +-- HPC.29.04/
#           |       +-- HPC.29.06/
#           |       ...
#           |
#           +-- voltage_L1/
#           +-- voltage_TEG/
#
VOLPY_DATASETS = ''

# Path to a directory containing copies of the datasets introduced in this work,
# which can be downloaded from: XXX.
# Download voltage_HPC2.zip. Unzipped files should be placed as follows:
#     <Directory specified by HPC2_DATASETS>/
#           |
#           +-- HPC2/
#           |     |
#           |     +-- 00_02.tif
#           |     +-- 00_03.tif
#           |     ...
#           |
#           +-- HPC2_GT/
#                 |
#                 +-- AnnotatorA/
#                 +-- AnnotatorB/
#                 +-- AnnotatorC/
#                 +-- 00_02.tif
#                 +-- 00_03.tif
#                 ...
#
HPC2_DATASETS = ''

# Path to a directory under which output data will be stored.
# The directory structure will be as follows:
#     <Directory specified by OUTPUT_BASE_PATH>/
#           |
#           +-- compare/
#                  |
#                  + invivo/
#                  + volpy/
#
#           +-- results/
#                  |
#                  + voltage_HPC/
#                  + voltage_HPC2/
#                  + voltage_L1/
#                  + voltage_TEG/
#
# The total data size will be around 300 GB.
OUTPUT_BASE_PATH = ''

# Path to a file storing a U-Net segmentation network model. A pre-trained
# model can be downloaded from: XXX.
# One can also train a model using the provided pipeline and speficy it here.
MODEL_FILE = ''

# Path to a directory under which training data and results will be stored.
# The total data size will be around 150 GB.
TRAIN_BASE_PATH = ''
