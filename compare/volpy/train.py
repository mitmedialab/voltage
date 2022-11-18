import random
import shutil
import zipfile
import numpy as np
import tifffile as tiff
from pathlib import Path
from caiman.base.rois import nf_read_roi


MRCNN_PATH = Path('./Mask_RCNN')
INPUT_PATH = Path('/media/bandy/nvme_data/VolPy_Data/Extracted')
OUTPUT_PATH = Path('/media/bandy/nvme_work/voltage/compare/volpy')
DATASET_GROUPS = ['voltage_L1', 'voltage_TEG', 'voltage_HPC']
VALIDATION_RATIO = 3
seed = 100


script_dir = MRCNN_PATH.joinpath('samples', 'neurons')
script_dir.mkdir(exist_ok=True)
if(not script_dir.joinpath('neurons.py').exists()):
    shutil.copy('CaImAn/caiman/source_extraction/volpy/mrcnn/neurons.py', script_dir)

train_dir = MRCNN_PATH.joinpath('datasets', 'neurons', 'train')
valid_dir = MRCNN_PATH.joinpath('datasets', 'neurons', 'val')
train_dir.mkdir(parents=True, exist_ok=True)
valid_dir.mkdir(exist_ok=True)


def convert_files(files, save_dir, group):
    for f in files:
        summary_images = tiff.imread(f)
        dataset_name = f.parent.name
        np.savez(save_dir.joinpath(dataset_name + '.npz'),
                 summary_images.transpose([1, 2, 0]))

        zip_file = INPUT_PATH.joinpath(group, dataset_name, dataset_name + '_ROI.zip')
        with zipfile.ZipFile(zip_file) as zf:
            polygons = []
            for roi_name in zf.namelist():
                coords = nf_read_roi(zf.open(roi_name))
                polygon = {
                    'name': 'polygon',
                    'all_points_x': coords[:, 1],
                    'all_points_y': coords[:, 0]
                    }
                polygons.append(polygon)
            np.savez(save_dir.joinpath(dataset_name + '_mask.npz'), polygons)


for group in DATASET_GROUPS:
    files = sorted(OUTPUT_PATH.joinpath(group).glob('*/*_summary_images.tif'))
    random.Random(seed).shuffle(files)
    num_validation = len(files) // VALIDATION_RATIO
    train_files = files[:-num_validation]
    valid_files = files[-num_validation:]

    convert_files(train_files, train_dir, group)
    convert_files(valid_files, valid_dir, group)


