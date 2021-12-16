# common parameters
TIME_SEGMENT_SIZE = 50
PATCH_SHAPE = (64, 64)
MODEL_PATH = '/media/bandy/nvme_work/voltage/test/model'
PREPROC_COMMAND = 'preproc/main -db -ms 5 -sm 1 -sc 0 -ss 3 -sw %d' % TIME_SEGMENT_SIZE


# runtime parameters
RUN_MODE = 'train'
RUN_SIMULATE = True
RUN_PREPROC = True
RUN_TRAIN = True
RUN_DEMIX = False
RUN_EVALUATE = False

FILENAME = ''
BASE_PATH = '/media/bandy/nvme_work/voltage/test'


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
