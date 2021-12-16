# common parameters
TIME_SEGMENT_SIZE = 50
PATCH_SHAPE = (64, 64)
MODEL_PATH = '/media/bandy/nvme_work/voltage/test/model'
PREPROC_COMMAND = 'preproc/main -db -ms 5 -sm 1 -sc 0 -ss 3 -sw %d' % TIME_SEGMENT_SIZE


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
