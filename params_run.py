# common parameters (must be consistent with training)
TIME_SEGMENT_SIZE = 50
PATCH_SHAPE = (64, 64)

# preprocessing parameters
MOTION_SEARCH_LEVEL = 2
MOTION_SEARCH_SIZE = 5
MOTION_PATCH_SIZE = 10
MOTION_PATCH_OFFSET = 7
SIGNAL_SCALE = 3.0 # must be consistent with training


# runtime parameters
RUN_MODE = 'run' # run the pipeline for neuron detection
RUN_PREPROC = True
RUN_SEGMENT = True
RUN_DEMIX = True
RUN_EVALUATE = True

FILENAME = '' # if non-empty, only the specified file will be processed


# real data parameters
INFERENCE_TILE_STRIDES = (8, 8)
BATCH_SIZE = 128

INPUT_PATH = '/media/bandy/nvme_data/ramdas/VI/SelectedData_v0.2/WholeTifs'
GT_PATH = '/media/bandy/nvme_data/ramdas/VI/SelectedData_v0.2/GT_comparison/GTs_rev20201027/consensus'
PREPROC_PATH = '/media/bandy/nvme_work/voltage/run' #'preproc-db-ms5-sm1-sc0-ss3-sw50'
MODEL_PATH = '/media/bandy/nvme_work/voltage/train/model'
OUTPUT_PATH = '/media/bandy/nvme_work/voltage/run'