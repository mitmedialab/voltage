import os
import runpy
import shutil
import keras
from pathlib import Path
from datetime import datetime
from train import split_training_data


MODE = 0              # VolPy datasets
VALIDATION_RATIO = 3  # N-fold cross validation

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
    keras.backend.clear_session()
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
                # most parameters can be False or dummy numbers
                args = (input_file, output_subdir, 1, False,
                        False, 1, False, False, 1, False,
                        min_size, max_size, weights_path)
                command = 'python main.py %s %s %d %d %d %d %d %d %d %d %d %d %s' % args
                run_command(command)

logfile.close()
