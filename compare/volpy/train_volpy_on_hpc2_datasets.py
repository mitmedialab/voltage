import os
import runpy
import shutil
import keras
from pathlib import Path
from datetime import datetime
from train import split_training_data


MODE = 1               # HPC2 datasets
VALIDATION_RATIO = 13  # N-fold cross validation

paths_file = Path(__file__).absolute().parents[2].joinpath('params', 'paths.py')
paths = runpy.run_path(paths_file)

INPUT_PATH = paths['HPC2_DATASETS']
GT_PATH = paths['HPC2_DATASETS'] + '_GT'
OUTPUT_PATH = Path(paths['OUTPUT_BASE_PATH'], 'compare', 'volpy', 'voltage_HPC2')

MIN_SIZE = 10               # Minimum neuron size to be detected
MAX_SIZE = 28               # Maximum neuron size to be detected


MRCNN_PATH = './Mask_RCNN'
MRCNN_SCRIPT_PATH = Path(MRCNN_PATH, 'samples', 'neurons')
MRCNN_TRAIN_CMD = 'python neurons.py train --dataset=../../datasets/neurons --weights=coco'
MRCNN_LOG_PATH = Path(MRCNN_PATH, 'logs')
MRCNN_VALIDATION_PATH = Path(MRCNN_PATH, 'datasets', 'neurons', 'val')


def run_command(command):
    print(command)
    os.system(command)


now = datetime.now()
record_name = now.strftime('hpc2_%Y%m%dT%H%M%S')
record_dir = Path(OUTPUT_PATH, record_name)
record_dir.mkdir()
logfile = open(record_dir.joinpath('log.txt'), 'w')

input_files = sorted(Path(INPUT_PATH).glob('*.tif'))
for index in range(VALIDATION_RATIO):
    _, val_files = split_training_data(INPUT_PATH, OUTPUT_PATH, GT_PATH,
                                       MODE, [''],
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

    for input_file in input_files:
        dataset_name = input_file.stem
        if(dataset_name in val_names):
            output_dir = Path(OUTPUT_PATH).joinpath(dataset_name)
            # most parameters can be False or dummy numbers
            args = (input_file, output_dir, 1, False,
                    False, 1, False, False, 1, False,
                    MIN_SIZE, MAX_SIZE, weights_path)
            command = 'python main.py %s %s %d %d %d %d %d %d %d %d %d %d %s' % args
            run_command(command)

logfile.close()
