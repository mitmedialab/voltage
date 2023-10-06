# runtime parameters
RUN_CORRECT  = False
RUN_PREPROC  = False
RUN_SEGMENT  = False
RUN_DEMIX    = False
RUN_SPIKE    = False
RUN_EVALUATE = True

# file paths
import runpy
from pathlib import Path
paths_file = Path(__file__).absolute().parents[1].joinpath('paths.py')
paths = runpy.run_path(paths_file)

INPUT_DIR = Path(paths['VOLPY_DATASETS'], 'voltage_TEG')
INPUT_FILES = sorted(INPUT_DIR.glob('*/*.tif'))
GT_FILES = [f.with_name(f.stem + '_ROI.zip') for f in INPUT_FILES]
OUTPUT_DIR = Path(paths['OUTPUT_BASE_PATH'], 'compare', 'volpy', 'voltage_TEG')
