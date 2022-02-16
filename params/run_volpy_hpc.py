# common parameters (must be consistent with training)
TIME_SEGMENT_SIZE = 50

# preprocessing parameters
FIRST_FRAME = 10
MOTION_SEARCH_LEVEL = 2
MOTION_SEARCH_SIZE = 5
MOTION_PATCH_SIZE = 15
MOTION_PATCH_OFFSET = 7
SIGNAL_SCALE = 3.0 # must be consistent with training


# runtime parameters
RUN_MODE = 'run' # run the pipeline for neuron detection
RUN_PREPROC = True
RUN_SEGMENT = True
RUN_DEMIX = True
RUN_EVALUATE = True


# performance parameters (optimal values depend on the computer environment)
NUM_THREADS_DEMIXING = 16


# real data parameters
TILE_SHAPE = (64, 64)
TILE_STRIDES = (8, 8)
BATCH_SIZE = 128


# file paths
import pathlib
INPUT_DIR = '/media/bandy/nvme_data/VolPy_Data/Extracted/voltage_HPC'
INPUT_FILES = sorted(pathlib.Path(INPUT_DIR).glob('*/*.tif'))
GT_FILES = [f.with_name(f.stem + '_ROI.zip') for f in INPUT_FILES]
MODEL_FILE = '/media/bandy/nvme_work/voltage/models/model20220106_dendrites/model.h5'
OUTPUT_DIR = '/media/bandy/nvme_work/voltage/volpyHPC'
