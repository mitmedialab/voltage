# common parameters (must be consistent with training)
TIME_SEGMENT_SIZE = 50

# motion/shading correction parameters
MOTION_SEARCH_LEVEL = 2
MOTION_SEARCH_SIZE = 3
MOTION_PATCH_SIZE = 10
MOTION_PATCH_OFFSET = 7
MOTION_Y_RANGE = 0.5

# preprocessing parameters
SIGNAL_METHOD = 'med-min'

# demixing parameters
PROBABILITY_THRESHOLD = 0.2
AREA_THRESHOLD_MIN = 20
AREA_THRESHOLD_MAX = 200
CONCAVITY_THRESHOLD = 1.3
INTENSITY_THRESHOLD = 0.02
ACTIVITY_THRESHOLD = 0
BACKGROUND_EDGE = 2.0
BACKGROUND_THRESHOLD = 0.003
MASK_DILATION = 1


# runtime parameters
RUN_MODE = 'run' # run the pipeline for neuron detection
RUN_CORRECT = True
RUN_PREPROC = True
RUN_SEGMENT = True
RUN_DEMIX = True
RUN_EVALUATE = True


# performance parameters (optimal values depend on the computer environment)
NUM_THREADS_CORRECT = 0  # 0 uses all the available logical cores
NUM_THREADS_PREPROC = 0  # 0 uses all the available logical cores
GPU_MEM_SIZE = 5 # GB


# real data parameters
TILE_SHAPE = (64, 64)
TILE_STRIDES = (8, 8)
BATCH_SIZE = 128
NORM_CHANNEL = 1
NORM_SHIFTS = [False, True]


# file paths
from pathlib import Path
INPUT_DIR = '/media/bandy/nvme_data/VolPy_Data/Extracted/voltage_L1'
INPUT_FILES = sorted(Path(INPUT_DIR).glob('*/*.tif'))
GT_FILES = [f.with_name(f.stem + '_ROI.zip') for f in INPUT_FILES]
MODEL_FILE = '/media/bandy/nvme_work/voltage/models/model20220620_patchnorm/model_e11_v0.0456.h5'
OUTPUT_DIR = '/media/bandy/nvme_work/voltage/volpyL1'
