import os
import shutil
from pathlib import Path
from datetime import datetime
from train import split_training_data


MODE = 1              # Archon dataset
VALIDATION_RATIO = 3  # N-fold cross validation

INPUT_PATH = '/media/bandy/nvme_data/voltage/datasets_v0.5/lowmag'
GT_PATH = '/media/bandy/nvme_data/voltage/datasets_v0.5/lowmag_GT'
OUTPUT_PATH = '/media/bandy/nvme_work/voltage/compare/volpy/lowmag'
MIN_SIZE = 10
MAX_SIZE = 28
MAX_SHIFT = 20              # search range for motion correction (20 is equivalent to our 5)
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
record_name = now.strftime('archon_%Y%m%dT%H%M%S')
record_dir = Path(OUTPUT_PATH, record_name)
record_dir.mkdir()
logfile = open(record_dir.joinpath('log.txt'), 'w')

input_files = sorted(Path(INPUT_PATH).glob('*.tif'))
for index in range(VALIDATION_RATIO):
    _, val_files = split_training_data(INPUT_PATH, OUTPUT_PATH, GT_PATH,
                                       MODE, [''],
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

    for input_file in input_files:
        dataset_name = input_file.stem
        if(dataset_name in val_names):
            output_dir = Path(OUTPUT_PATH).joinpath(dataset_name)
            frame_rate = 1 # this will not be used so dummy value is fine
            args = (input_file, output_dir, frame_rate, MIN_SIZE, MAX_SIZE,
                    MAX_SHIFT, USE_CUDA, GAUSSIAN_BLUR,
                    False, False, weights_path)
            command = 'python main.py %s %s %d %d %d %d %d %d %d %d %s' % args
            run_command(command)

logfile.close()
