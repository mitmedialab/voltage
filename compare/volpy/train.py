import random
import shutil
import zipfile
import numpy as np
import tifffile as tiff
from pathlib import Path
from skimage.measure import find_contours
from caiman.base.rois import nf_read_roi



def read_zipped_roi(filename):
    coords_list = []
    with zipfile.ZipFile(filename) as zf:
        for roi_name in zf.namelist():
            coords = nf_read_roi(zf.open(roi_name))
            coords_list.append(coords)
    return coords_list

def read_masks(filename):
    masks = tiff.imread(filename)
    coords_list = []
    for mask in masks:
        contours = find_contours(mask, 0.5)
        assert len(contours) == 1
        coords_list.append(contours[0])
    return coords_list

def convert_files(files, save_dir, gt_dir, mode):
    for f in files:
        summary_images = tiff.imread(f)
        dataset_name = f.parent.name
        np.savez(save_dir.joinpath(dataset_name + '.npz'),
                 summary_images.transpose([1, 2, 0]))

        if(mode == 0): # VolPy datasets
            roi_file = gt_dir.joinpath(dataset_name, dataset_name + '_ROI.zip')
            coords_list = read_zipped_roi(roi_file)
        else: # Archon dataset
            roi_file = gt_dir.joinpath(dataset_name + '.tif')
            coords_list = read_masks(roi_file)

        polygons = []
        for coords in coords_list:
            polygon = {
                'name': 'polygon',
                'all_points_x': coords[:, 1],
                'all_points_y': coords[:, 0]
                }
            polygons.append(polygon)
        np.savez(save_dir.joinpath(dataset_name + '_mask.npz'), polygons)


def split_training_data(input_path, output_path, gt_path, mode, dataset_groups,
                        validation_ratio, validation_index, seed=100,
                        mrcnn_path = './Mask_RCNN'):

    input_path = Path(input_path)
    output_path = Path(output_path)
    gt_path = Path(gt_path)
    mrcnn_path = Path(mrcnn_path)

    #%% Initialize directories
    script_dir = mrcnn_path.joinpath('samples', 'neurons')
    script_dir.mkdir(exist_ok=True)
    if(not script_dir.joinpath('neurons.py').exists()):
        shutil.copy('CaImAn/caiman/source_extraction/volpy/mrcnn/neurons.py',
                    script_dir)

    datasets_dir = mrcnn_path.joinpath('datasets')
    datasets_dir.mkdir(exist_ok=True)
    neurons_dir = datasets_dir.joinpath('neurons')
    if(neurons_dir.exists()):
        shutil.rmtree(neurons_dir)
    neurons_dir.mkdir()
    train_dir = neurons_dir.joinpath('train')
    valid_dir = neurons_dir.joinpath('val')
    train_dir.mkdir()
    valid_dir.mkdir()

    #%% Split data
    train_files_all = []
    valid_files_all = []
    for group in dataset_groups:
        files = sorted(output_path.joinpath(group).glob('*/*_summary_images.tif'))
        random.Random(seed).shuffle(files)
        valid_files = np.array_split(files, validation_ratio)[validation_index]
        train_files = [f for f in files if f not in valid_files]
        if(mode == 0):
            gt_dir = gt_path.joinpath(group)
        else:
            gt_dir = gt_path
        convert_files(train_files, train_dir, gt_dir, mode)
        convert_files(valid_files, valid_dir, gt_dir, mode)
        train_files_all.extend(train_files)
        valid_files_all.extend(valid_files)

    return train_files_all, valid_files_all


#%% Test
INPUT_PATH = '/media/bandy/nvme_data/VolPy_Data/Extracted'
OUTPUT_PATH = '/media/bandy/nvme_work/voltage/compare/volpy'
DATASET_GROUPS = ['voltage_L1', 'voltage_TEG', 'voltage_HPC']
split_training_data(INPUT_PATH, OUTPUT_PATH, INPUT_PATH, 0, DATASET_GROUPS, 3, 0)
