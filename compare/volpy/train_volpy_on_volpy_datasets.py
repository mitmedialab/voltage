import os
import shutil
from pathlib import Path
from datetime import datetime
from train import split_training_data


MODE = 0              # VolPy datasets
VALIDATION_RATIO = 3  # N-fold cross validation

INPUT_PATH = '/media/bandy/nvme_data/VolPy_Data/Extracted'
OUTPUT_PATH = '/media/bandy/nvme_work/voltage/compare/volpy'
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

MRCNN_PATH = './Mask_RCNN'
MRCNN_SCRIPT_PATH = Path(MRCNN_PATH, 'samples', 'neurons')
MRCNN_TRAIN_CMD = 'python neurons.py train --dataset=../../datasets/neurons --weights=coco'
MRCNN_LOG_PATH = Path(MRCNN_PATH, 'logs')
MRCNN_VALIDATION_PATH = Path(MRCNN_PATH, 'datasets', 'neurons', 'val')


def run_command(command):
    print(command)
    os.system(command)


now = datetime.now()
record_name = now.strftime('volpy_%Y%m%dT%H%M%S')
record_dir = Path(OUTPUT_PATH, record_name)
record_dir.mkdir()
logfile = open(record_dir.joinpath('log.txt'), 'w')

for index in range(VALIDATION_RATIO):
    _, val_files = split_training_data(INPUT_PATH, OUTPUT_PATH, INPUT_PATH,
                                       MODE, [g[0] for g in DATASET_GROUPS],
                                       VALIDATION_RATIO, index)
    command = '(cd %s; %s)' % (MRCNN_SCRIPT_PATH, MRCNN_TRAIN_CMD)
    run_command(command)

    weights_dir = sorted(MRCNN_LOG_PATH.iterdir())[-1]
    weights_path = Path(weights_dir, 'mask_rcnn_neurons_0040.h5')
    shutil.copyfile(weights_path, record_dir.joinpath('index%2.2d.h5' % index))
    logfile.write('weights: %s\n' % weights_path.absolute())

    val_files = sorted(MRCNN_VALIDATION_PATH.glob('*_mask.npz'))
    val_names = [str(f.name).replace('_mask.npz', '') for f in val_files]
    for name in val_names:
        logfile.write('%s\n' % name)

    for group in DATASET_GROUPS:
        group_name, frame_rate, min_size, max_size = group
        input_dir = Path(INPUT_PATH).joinpath(group_name)
        input_files = sorted(input_dir.glob('*/*.tif'))
        output_dir = Path(OUTPUT_PATH).joinpath(group_name)
        for input_file in input_files:
            dataset_name = input_file.stem
            if(dataset_name in val_names):
                output_subdir = output_dir.joinpath(dataset_name)
                args = (input_file, output_subdir, frame_rate, min_size, max_size,
                        MAX_SHIFT, USE_CUDA, GAUSSIAN_BLUR,
                        False, False, weights_path)
                command = 'python main.py %s %s %d %d %d %d %d %d %d %d %s' % args
                run_command(command)

logfile.close()
