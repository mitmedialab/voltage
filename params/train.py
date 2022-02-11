# common parameters (must be consistent with inference)
TIME_SEGMENT_SIZE = 50
PATCH_SHAPE = (64, 64)

# preprocessing parameters
MOTION_SEARCH_LEVEL = 2
MOTION_SEARCH_SIZE = 5
MOTION_PATCH_SIZE = 10
MOTION_PATCH_OFFSET = 7
SIGNAL_SCALE = 3.0 # must be consistent with inference


# runtime parameters
RUN_MODE = 'train' # run the pipeline for training
RUN_SIMULATE = True
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
NUM_CELLS_MAX = 15


# training parameters
NUM_DARTS = 10
BATCH_SIZE = 128
EPOCHS = 10
# WARNING: too small tile strides can lead to many samples to be fed into
# the U-Net for prediction, which can cause GPU out-of-memory error.
# For some reason, GPU memory consumption seems to pile up as more samples
# are input, no matter how small the batch size is set to.
# To avoid this, we might need to split a single input video into multiple
# time segments or even perform prediction on a frame-by-frame basis.
VALIDATION_TILE_STRIDES = (16, 16)

DATA_DIR = '/media/bandy/nvme_work/voltage/train/synthetic'
PREPROC_DIR = '/media/bandy/nvme_work/voltage/train/preproc'
MODEL_DIR = '/media/bandy/nvme_work/voltage/train/model'
OUTPUT_DIR = '/media/bandy/nvme_work/voltage/train'
