# common parameters (must be consistent with inference)
TIME_SEGMENT_SIZE = 50
MODEL_IO_SHAPE = (64, 64)

# motion/shading correction parameters
MOTION_SEARCH_LEVEL = 2
MOTION_SEARCH_SIZE = 3
MOTION_PATCH_SIZE = 10
MOTION_PATCH_OFFSET = 7

# demixing parameters
PROBABILITY_THRESHOLD = 0.5
AREA_THRESHOLD_MIN = 0
AREA_THRESHOLD_MAX = 1000
CONCAVITY_THRESHOLD = 2
INTENSITY_THRESHOLD = 0
ACTIVITY_THRESHOLD = 0
BACKGROUND_EDGE = 1.0
BACKGROUND_THRESHOLD = 0.003


# runtime parameters
RUN_MODE = 'train' # run the pipeline for training
RUN_SIMULATE = True
RUN_CORRECT = True
RUN_PREPROC = True
RUN_TRAIN = True
RUN_DEMIX = True
RUN_EVALUATE = True

FILENAME = '' # if non-empty, only the specified file will be processed


# simulation parameters
IMAGE_SHAPE = (128, 128)
TIME_FRAMES = 1000
NUM_VIDEOS = 1000
NUM_CELLS_MIN = 5
NUM_CELLS_MAX = 30


# training parameters
NUM_DARTS = 10
BATCH_SIZE = 128
EPOCHS = 20
TILE_STRIDES = (16, 16)
NORM_CHANNEL = 1
NORM_SHIFTS = [False, True]

DATA_DIR = '/media/bandy/nvme_data/voltage/train/synthetic'
PREPROC_DIR = '/media/bandy/nvme_data/voltage/train/preproc'
MODEL_DIR = '/media/bandy/nvme_data/voltage/train/model'
OUTPUT_DIR = '/media/bandy/nvme_data/voltage/train'
