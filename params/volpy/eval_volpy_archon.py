# runtime parameters
RUN_MODE = 'run' # run the pipeline for neuron detection
RUN_CORRECT = False
RUN_PREPROC = False
RUN_SEGMENT = False
RUN_DEMIX = False
RUN_EVALUATE = True

# file paths
from pathlib import Path
INPUT_DIR = '/media/bandy/nvme_data/voltage/datasets_v0.5/lowmag'
INPUT_FILES = sorted(Path(INPUT_DIR).glob('*.tif'))
GT_DIR = '/media/bandy/nvme_data/voltage/datasets_v0.5/lowmag_GT'
GT_FILES = [Path(GT_DIR).joinpath(f.name) for f in INPUT_FILES]
OUTPUT_DIR = '/media/bandy/nvme_work/voltage/compare/volpy/lowmag'
