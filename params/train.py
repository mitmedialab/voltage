# runtime parameters
RUN_MODE = 'train'
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
MODEL_IO_SHAPE = (64, 64)
NUM_DARTS = 10
BATCH_SIZE = 128
EPOCHS = 20
TILE_STRIDES = (16, 16)


# demixing parameters
PROBABILITY_THRESHOLD = 0.5
AREA_THRESHOLD_MIN = 50
AREA_THRESHOLD_MAX = 1000
CONCAVITY_THRESHOLD = 1.5
INTENSITY_THRESHOLD = 0
ACTIVITY_THRESHOLD = 0
BACKGROUND_EDGE = 0
BACKGROUND_THRESHOLD = 0


# data directories
DATA_DIR = '/media/bandy/nvme_data/voltage/train/synthetic'
PREPROC_DIR = '/media/bandy/nvme_data/voltage/train/preproc'
MODEL_DIR = '/media/bandy/nvme_data/voltage/train/model'
OUTPUT_DIR = '/media/bandy/nvme_data/voltage/train'
