# runtime parameters
RUN_MODE = 'run' # run the pipeline for neuron detection
RUN_CORRECT = False
RUN_PREPROC = False
RUN_SEGMENT = False
RUN_DEMIX = False
RUN_EVALUATE = True

# file paths
import pathlib
INPUT_DIR = '/media/bandy/nvme_data/VolPy_Data/Extracted/voltage_TEG'
INPUT_FILES = sorted(pathlib.Path(INPUT_DIR).glob('*/*.tif'))
GT_FILES = [f.with_name(f.stem + '_ROI.zip') for f in INPUT_FILES]
OUTPUT_DIR = '/media/bandy/nvme_work/voltage/compare/volpy/voltage_TEG'
