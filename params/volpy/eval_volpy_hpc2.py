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

INPUT_DIR = Path(paths['HPC2_DATASETS'], 'HPC2')
INPUT_FILES = sorted(INPUT_DIR.glob('*.tif'))
GT_DIR = Path(paths['HPC2_DATASETS'], 'HPC2_GT')
GT_FILES = [Path(GT_DIR).joinpath(f.name) for f in INPUT_FILES]
OUTPUT_DIR = Path(paths['OUTPUT_BASE_PATH'], 'compare', 'volpy', 'voltage_HPC2')
