# motion/shading correction parameters
MOTION_X_RANGE = 0.7

# segmentation parameters
TILE_STRIDES = (8, 16)
NORM_SHIFTS = [True, True]

# demixing parameters
PROBABILITY_THRESHOLD = 0.95
AREA_THRESHOLD_MIN = 110
AREA_THRESHOLD_MAX = 1000
CONCAVITY_THRESHOLD = 1.5
INTENSITY_THRESHOLD = 0.01
ACTIVITY_THRESHOLD = 0.001
BACKGROUND_EDGE = 1.0
BACKGROUND_THRESHOLD = 0.003
MASK_DILATION = 1

# runtime parameters
RUN_CORRECT = True
RUN_PREPROC = True
RUN_SEGMENT = True
RUN_DEMIX = True
RUN_SPIKE = True
RUN_EVALUATE = True

# file paths
from pathlib import Path
INPUT_DIR = '/media/bandy/nvme_data/voltage/datasets_v0.5/lowmag'
INPUT_FILES = sorted(Path(INPUT_DIR).glob('*.tif'))
GT_DIR = '/media/bandy/nvme_data/voltage/datasets_v0.5/lowmag_GT'
GT_FILES = [Path(GT_DIR).joinpath(f.name) for f in INPUT_FILES]
MODEL_FILE = '/media/bandy/nvme_data/voltage/models/model20220620_patchnorm/model_e11_v0.0456.h5'
OUTPUT_DIR = '/media/bandy/nvme_work/voltage/lowmag_v0.5'
