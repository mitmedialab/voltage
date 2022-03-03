# common parameters (must be consistent with training)
TIME_SEGMENT_SIZE = 50

# motion/shading correction parameters
MOTION_SEARCH_LEVEL = 0
MOTION_SEARCH_SIZE = 0
MOTION_PATCH_SIZE = 1
MOTION_PATCH_OFFSET = 1000

# preprocessing parameters
SIGNAL_METHOD = 'med-min'
SIGNAL_SCALE = 3.0
SIGNAL_BINNING = 2

# demixing parameters
AREA_THRESHOLD = 100
ACTIVITY_LEVEL_THRESHOLD_RELATIVE = 0
ACTIVITY_LEVEL_THRESHOLD_ABSOLUTE = 0.0001


# runtime parameters
RUN_MODE = 'run' # run the pipeline for neuron detection
RUN_CORRECT = True
RUN_PREPROC = True
RUN_SEGMENT = True
RUN_DEMIX = True
RUN_EVALUATE = True


# performance parameters (optimal values depend on the computer environment)
NUM_THREADS_DEMIXING = 16
GPU_MEM_SIZE = 5 # GB


# real data parameters
TILE_SHAPE = (128, 128)
TILE_STRIDES = (32, 32)
BATCH_SIZE = 128


# file paths
import pathlib
INPUT_DIR = '/media/bandy/nvme_data/VolPy_Data/Extracted/voltage_TEG'
INPUT_FILES = sorted(pathlib.Path(INPUT_DIR).glob('*/*.tif'))
GT_FILES = [f.with_name(f.stem + '_ROI.zip') for f in INPUT_FILES]
MODEL_FILE = '/media/bandy/nvme_work/voltage/models/model20220106_dendrites/model.h5'
OUTPUT_DIR = '/media/bandy/nvme_work/voltage/volpyTEG'
