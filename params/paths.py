# Path to a directory containing copies of the datasets used in the VolPy paper,
# which can be downloaded from: https://zenodo.org/records/4515768
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
# which can be downloaded from: https://zenodo.org/records/10020273
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

# Path to a directory for storing output data.
# After running all the scripts, the directory structure will be as follows:
#     <Directory specified by OUTPUT_BASE_PATH>/
#           |
#           +-- compare/
#           |      |
#           |      + invivo/
#           |      + volpy/
#           |
#           +-- results/
#                  |
#                  + voltage_HPC/
#                  + voltage_HPC2/
#                  + voltage_L1/
#                  + voltage_TEG/
#
# The total data size will be around 300 GB.
OUTPUT_BASE_PATH = ''

# Path to a file storing a U-Net segmentation network model.
# One can either specify the pre-trained model (voltage_u-net_model.h5)
# which can be downloaded from: https://zenodo.org/records/10020273
# or train a model using the provided pipeline and speficy it here.
MODEL_FILE = ''

# Path to a directory for storing training data and results.
# After training, the total data size will be around 150 GB.
TRAIN_BASE_PATH = ''
