# motion/shading correction parameters
MOTION_X_RANGE = 0.7

# segmentation parameters
TILE_STRIDES = (8, 16)
TILE_MARGIN = (0, 0.2)
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

# spike detection parameters
SPIKE_THRESHOLD = 2.5
REMOVE_INACTIVE = True

# runtime parameters
RUN_CORRECT  = True
RUN_PREPROC  = True
RUN_SEGMENT  = True
RUN_DEMIX    = True
RUN_SPIKE    = True
RUN_EVALUATE = True

# file paths
import runpy
from pathlib import Path
paths_file = Path(__file__).parent.joinpath('paths.py')
paths = runpy.run_path(paths_file)

INPUT_DIR = Path(paths['HPC2_DATASETS'], 'HPC2')
INPUT_FILES = sorted(INPUT_DIR.glob('*.tif'))
GT_DIR = Path(paths['HPC2_DATASETS'], 'HPC2_GT')
GT_FILES = [Path(GT_DIR).joinpath(f.name) for f in INPUT_FILES]
MODEL_FILE = paths['MODEL_FILE']
OUTPUT_DIR = Path(paths['OUTPUT_BASE_PATH'], 'results', 'voltage_HPC2')
