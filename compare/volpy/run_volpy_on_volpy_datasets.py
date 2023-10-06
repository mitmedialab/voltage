import os
import runpy
import shutil
from pathlib import Path


# Path and dataset specific parameters
paths_file = Path(__file__).absolute().parents[2].joinpath('params', 'paths.py')
paths = runpy.run_path(paths_file)

INPUT_PATH = paths['VOLPY_DATASETS']
OUTPUT_PATH = Path(paths['OUTPUT_BASE_PATH'], 'compare', 'volpy')
DATASET_GROUPS = [
    # (group_name, frame_rate, min_size, max_size)
    # Note that the sizes are lengths and will be squared to specify an area range
    # The values are taken from the VolPy paper
    ('voltage_L1',   400,  0, 1000), # no size constraint
    ('voltage_TEG',  300, 10, 1000), # remove masks with <100 pixels
    ('voltage_HPC', 1000, 20, 1000), # remove masks with <400 pixels
]

# Computational resource parameters
NUM_PROCESSES = 4           # Number of processes to be used for motion correction and
                            # summary image creation (it seems that all the available
                            # threads will be used anyway though)
USE_CUDA = False            # Whether to use GPU for motion correction

# Motion correction parameters
DO_MOTION_CORRECTION = True # If False, previously saved result (mmap) will be used
MAX_SHIFT = 5               # Search range for motion correction
SAVE_MOVIE = True           # Set False to measure motion correction computation time

# Summary image creation parameters
DO_SUMMARY_CREATION = True  # If False, previously saved result (tiff) will be used
GAUSSIAN_BLUR = False       # Set True when the input video is noisy

# Segmentation parameters
WEIGHTS_PATH = ''           # If blank, the default weights will be downloaded and used


for group in DATASET_GROUPS:
    group_name, frame_rate, min_size, max_size = group
    input_dir = Path(INPUT_PATH).joinpath(group_name)
    input_files = sorted(input_dir.glob('*/*.tif'))
    output_dir = Path(OUTPUT_PATH).joinpath(group_name)
    output_dir.mkdir(exist_ok=True, parents=True)
    for input_file in input_files:
        dataset_name = input_file.stem
        output_subdir = output_dir.joinpath(dataset_name)
        args = (input_file, output_subdir,
                NUM_PROCESSES, USE_CUDA,
                DO_MOTION_CORRECTION, MAX_SHIFT, SAVE_MOVIE,
                DO_SUMMARY_CREATION, frame_rate, GAUSSIAN_BLUR,
                min_size, max_size, WEIGHTS_PATH)

        command = 'python main.py %s %s %d %d %d %d %d %d %d %d %d %d %s' % args
        print(command)
        os.system(command)
        shutil.copy('time_motion_correct.txt', output_subdir)
        shutil.copy('time_summary_images.txt', output_subdir)
        shutil.copy('time_segmentation.txt', output_subdir)
