# common parameters
TIME_SEGMENT_SIZE = 50
PATCH_SHAPE = (64, 64)
MODEL_PATH = '/media/bandy/nvme_work/voltage/test/model'

# preprocessing parameters
MOTION_SEARCH_LEVEL = 2
MOTION_SEARCH_SIZE = 5
MOTION_PATCH_SIZE = 10
MOTION_PATCH_OFFSET = 7
SIGNAL_SCALE = 3.0


# runtime parameters
RUN_MODE = 'run'
RUN_PREPROC = False
RUN_SEGMENT = True
RUN_DEMIX = False
RUN_EVALUATE = False

FILENAME = ''
BASE_PATH = '/media/bandy/nvme_work/voltage/real'


# real data parameters
INFERENCE_TILE_STRIDES = (8, 8)
INPUT_PATH = '/media/bandy/nvme_data/ramdas/VI/SelectedData_v0.2/WholeTifs'
GT_PATH = '/media/bandy/nvme_data/ramdas/VI/SelectedData_v0.2/GT_comparison/GTs_rev20201027/consensus'
PREPROC_PATH = '/media/bandy/nvme_work/voltage/preproc-db-ms5-sm1-sc0-ss3-sw50'
BATCH_SIZE = 128
