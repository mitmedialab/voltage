# directory under which to save training-related files
# (the data size will be about 150 GB with the default settings)
BASE_DIR = ''


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
BACKGROUND_EDGE = 0
BACKGROUND_THRESHOLD = 0

# runtime parameters
RUN_MODE = 'train'
RUN_SIMULATE = True
RUN_CORRECT = True
RUN_PREPROC = True
RUN_TRAIN = True
RUN_DEMIX = True
RUN_EVALUATE = True
