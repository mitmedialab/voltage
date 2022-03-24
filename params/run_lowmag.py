# common parameters (must be consistent with training)
TIME_SEGMENT_SIZE = 50

# motion/shading correction parameters
MOTION_SEARCH_LEVEL = 2
MOTION_SEARCH_SIZE = 5
MOTION_PATCH_SIZE = 15
MOTION_PATCH_OFFSET = 7

# preprocessing parameters
SIGNAL_SCALE = 3.0

# demixing parameters
AREA_THRESHOLD = 55
ACTIVITY_LEVEL_THRESHOLD_RELATIVE = 1/9
ACTIVITY_LEVEL_THRESHOLD_ABSOLUTE = 0.0001


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
NUM_THREADS_DEMIXING = 16
GPU_MEM_SIZE = 5 # GB


# real data parameters
TILE_SHAPE = (64, 64)
TILE_STRIDES = (8, 8)
BATCH_SIZE = 128


# file paths
import pathlib
INPUT_DIR = '/media/bandy/nvme_data/voltage/datasets_v0.3/lowmag'
INPUT_FILES = sorted(pathlib.Path(INPUT_DIR).glob('*.tif'))
GT_DIR = '/media/bandy/nvme_data/voltage/datasets_v0.3/lowmag_GT_v20201027'
GT_FILES = [pathlib.Path(GT_DIR).joinpath(f.name) for f in INPUT_FILES]
MODEL_FILE = '/media/bandy/nvme_work/voltage/models/model20220106_dendrites/model.h5'
OUTPUT_DIR = '/media/bandy/nvme_work/voltage/lowmag_v0.3'
