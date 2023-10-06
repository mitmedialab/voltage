# motion/shading correction parameters
MOTION_Y_RANGE = 0.5

# preprocessing parameters
SIGNAL_METHOD = 'med-min'

# segmentation parameters
TILE_STRIDES = (8, 8)

# demixing parameters
PROBABILITY_THRESHOLD = 0.2
AREA_THRESHOLD_MIN = 20
AREA_THRESHOLD_MAX = 200
CONCAVITY_THRESHOLD = 1.3
INTENSITY_THRESHOLD = 0.02
ACTIVITY_THRESHOLD = 0
BACKGROUND_EDGE = 2.0
BACKGROUND_THRESHOLD = 0.003
MASK_DILATION = 1

# spike detection parameters
POLARITY = -1

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

INPUT_DIR = Path(paths['VOLPY_DATASETS'], 'voltage_L1')
INPUT_FILES = sorted(INPUT_DIR.glob('*/*.tif'))
GT_FILES = [f.with_name(f.stem + '_ROI.zip') for f in INPUT_FILES]
MODEL_FILE = paths['MODEL_FILE']
OUTPUT_DIR = Path(paths['OUTPUT_BASE_PATH'], 'results', 'voltage_L1')
