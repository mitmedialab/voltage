# common parameters (must be consistent with training)
TIME_SEGMENT_SIZE = 50

# motion/shading correction parameters
MOTION_SEARCH_LEVEL = 2
MOTION_SEARCH_SIZE = 3
MOTION_PATCH_SIZE = 10
MOTION_PATCH_OFFSET = 7
MOTION_X_RANGE = 0.5

# demixing parameters
PROBABILITY_THRESHOLD = 0.2
AREA_THRESHOLD_MIN = 70
AREA_THRESHOLD_MAX = 1000
CONCAVITY_THRESHOLD = 1.2
INTENSITY_THRESHOLD = 0.02
ACTIVITY_THRESHOLD = 0.003
BACKGROUND_EDGE = 1.0
BACKGROUND_THRESHOLD = 0.02
MASK_DILATION = 0


# runtime parameters
RUN_MODE = 'run' # run the pipeline for neuron detection
RUN_CORRECT = False
RUN_PREPROC = False
RUN_SEGMENT = False
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
NORM_SHIFTS = [True, True]


# file paths
from pathlib import Path
INPUT_DIR = '/media/bandy/nvme_data/voltage/datasets_v0.5/highmag'
INPUT_FILES = sorted(Path(INPUT_DIR).glob('*.tif'))
#INPUT_FILES = [INPUT_FILES[i] for i in [4, 20]]
GT_DIR = '/media/bandy/nvme_data/voltage/datasets_v0.5/highmag_GT'
GT_FILES = [Path(GT_DIR).joinpath(f.name) for f in INPUT_FILES]
MODEL_FILE = '/media/bandy/nvme_work/voltage/models/model20220620_patchnorm/model_e11_v0.0456.h5'
OUTPUT_DIR = '/media/bandy/nvme_work/voltage/highmag_v0.5'
