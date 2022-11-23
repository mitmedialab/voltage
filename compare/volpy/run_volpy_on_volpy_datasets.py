import os
import shutil
from pathlib import Path


INPUT_PATH = '/media/bandy/nvme_data/VolPy_Data/Extracted'
OUTPUT_PATH = '/media/bandy/nvme_work/voltage/compare/volpy'
WEIGHTS_PATH = ''  # if blank, the default weights will be downloaded and used
DATASET_GROUPS = [
    # (group_name, frame_rate, min_size, max_size)
    # Note that the sizes are lengths and will be squared to specify an area range
    # The values are taken from the VolPy paper
    ('voltage_L1',   400,  0, 1000), # no size constraint
    ('voltage_TEG',  300, 10, 1000), # remove masks with <100 pixels
    ('voltage_HPC', 1000, 20, 1000), # remove masks with <400 pixels
]
MAX_SHIFT = 5               # search range for motion correction
USE_CUDA = False            # motion correction on GPU
GAUSSIAN_BLUR = False       # use when the input video is noisy
DO_MOTION_CORRECTION = True # if False, previously saved result (mmap) will be used
DO_SUMMARY_CREATION = True  # if False, previously saved result (tiff) will be used


for group in DATASET_GROUPS:
    group_name, frame_rate, min_size, max_size = group
    input_dir = Path(INPUT_PATH).joinpath(group_name)
    input_files = sorted(input_dir.glob('*/*.tif'))
    output_dir = Path(OUTPUT_PATH).joinpath(group_name)
    output_dir.mkdir(exist_ok=True)
    for input_file in input_files:
        dataset_name = input_file.stem
        output_subdir = output_dir.joinpath(dataset_name)
        args = (input_file, output_subdir, frame_rate, min_size, max_size,
                MAX_SHIFT, USE_CUDA, GAUSSIAN_BLUR,
                DO_MOTION_CORRECTION, DO_SUMMARY_CREATION, WEIGHTS_PATH)
        command = 'python main.py %s %s %d %d %d %d %d %d %d %d %s' % args
        print(command)
        os.system(command)
        shutil.copy('time_motion_correct.txt', output_subdir)
        shutil.copy('time_summary_images.txt', output_subdir)
        shutil.copy('time_segmentation.txt', output_subdir)
