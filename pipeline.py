import os
import pathlib
import multiprocessing as mp

from simulate import create_synthetic_data, decimate_video
from segment import train_model, validate_model, apply_model
from demix import compute_masks
from evaluate import run_ipynb_evaluate_each, run_ipynb_evaluate_all


# common parameters
TIME_SEGMENT_SIZE = 50
PATCH_SHAPE = (64, 64)
MODEL_PATH = '/media/bandy/nvme_work/voltage/test/model'

# simulation parameters
IMAGE_SHAPE = (128, 128)
TIME_FRAMES = 1000
NUM_VIDEOS = 1000
NUM_CELLS_MIN = 5
NUM_CELLS_MAX = 15
SIM_PATH = '/media/bandy/nvme_work/voltage/test'

# training parameters
NUM_DARTS = 10
BATCH_SIZE = 128
EPOCHS = 10
# WARNING: too small tile strides can lead to many samples to be fed into
# the U-Net for prediction, which can cause GPU out-of-memory error.
# For some reason, GPU memory consumption seems to pile up as more samples
# are input, no matter how small the batch size is set to.
# To avoid this, we might need to split a single input video into multiple
# time segments or even perform prediction on a frame-by-frame basis.
VALIDATION_TILE_STRIDES = (16, 16)

# real data parameters
INFERENCE_TILE_STRIDES = (8, 8)
REAL_PATH = '/media/bandy/nvme_work/voltage/real'
DATA_PATH = '/media/bandy/nvme_data/ramdas/VI/SelectedData_v0.2/WholeTifs'
GT_PATH = '/media/bandy/nvme_data/ramdas/VI/SelectedData_v0.2/GT_comparison/GTs_rev20201027/consensus'


def set_dir(base_path, dirname):
    p = pathlib.Path(base_path, dirname)
    if not p.exists():
        p.mkdir()
    return p
    

def simulate(num_videos, data_dir, temporal_gt_dir, spatial_gt_dir):
    num_neurons_list = list(range(NUM_CELLS_MIN, NUM_CELLS_MAX))
    args = []
    for i in range(num_videos):
        name = '%4.4d' % i
        num_neurons = num_neurons_list[i % len(num_neurons_list)]
        args.append((IMAGE_SHAPE, TIME_FRAMES, num_neurons,
                     data_dir, temporal_gt_dir, spatial_gt_dir, name))
    
    pool = mp.Pool(mp.cpu_count())
    pool.starmap(create_synthetic_data, args)
    pool.close()


def decimate(in_dir, out_dir, mode, size, filename):
    if(filename):
        filenames = [in_dir.joinpath(filename + '.tif')]
    else:
        filenames = sorted(in_dir.glob('*.tif'))
    args = []
    for in_file in filenames:
        out_file = out_dir.joinpath(in_file.name)
        args.append((in_file, out_file, mode, size))
    
    pool = mp.Pool(mp.cpu_count())
    pool.starmap(decimate_video, args)
    pool.close()


def preprocess(in_dir, out_dir, correction_dir, filename):
    if(filename):
        filenames = [in_dir.joinpath(filename + '.tif')]
    else:
        filenames = sorted(in_dir.glob('*.tif'))
    for in_file in filenames:
        command = 'preproc/main -db -ms 5 -sm 1 -sc 0 -ss 3 -sw %d' % TIME_SEGMENT_SIZE
        command += ' %s %s' % (in_file, out_dir)
        print(command)
        os.system(command)
        
        sig_file = out_dir.joinpath('signal.tif')
        out_file = out_dir.joinpath(in_file.name)
        command = 'mv %s %s' % (sig_file, out_file)
        os.system(command)
        
        cor_file = out_dir.joinpath('corrected.tif')
        out_file = correction_dir.joinpath(in_file.name)
        command = 'mv %s %s' % (cor_file, out_file)
        os.system(command)
        

def train(in_dirs, target_dir, model_dir, out_dir, ref_dir):
    seed = 0
    validation_ratio = 5
    train_model(in_dirs, target_dir, model_dir,
                seed, validation_ratio,
                PATCH_SHAPE, NUM_DARTS, BATCH_SIZE, EPOCHS)
    validate_model(in_dirs, target_dir, model_dir, out_dir, ref_dir,
                   seed, validation_ratio,
                   PATCH_SHAPE, VALIDATION_TILE_STRIDES, BATCH_SIZE)


def segment(in_dirs, model_dir, out_dir, ref_dir, filename):
    apply_model(in_dirs, model_dir, out_dir, ref_dir, filename,
                PATCH_SHAPE, INFERENCE_TILE_STRIDES, BATCH_SIZE)


def demix(in_dir, out_dir, correction_dir, filename):
    if(filename):
        filenames = [in_dir.joinpath(filename + '.tif')]
    else:
        filenames = sorted(in_dir.glob('*.tif'))
    for in_file in filenames:
        print('demixing ' + in_file.stem)
        out_file = out_dir.joinpath(in_file.name)
        corr_file = correction_dir.joinpath(in_file.name)
        compute_masks(in_file, corr_file, out_file)


def evaluate(in_dir, gt_dir, img_dir, out_dir, filename):
    if(filename):
        filenames = [in_dir.joinpath(filename + '.tif')]
    else:
        filenames = sorted(in_dir.glob('*.tif'))
    for in_file in filenames:
        print('evaluating ' + in_file.stem)
        gt_file = gt_dir.joinpath(in_file.name)
        img_file = img_dir.joinpath(in_file.name)
        run_ipynb_evaluate_each(in_file, gt_file, img_file, out_dir)
    run_ipynb_evaluate_all(out_dir)




mode = 'run'
filename = ''

if(mode == 'toy'):
    data_dir = set_dir(SIM_PATH, 'data')
    temporal_gt_dir = set_dir(SIM_PATH, 'temporal_label')
    spatial_gt_dir = set_dir(SIM_PATH, 'spatial_label')
    simulate(NUM_VIDEOS, data_dir, temporal_gt_dir, spatial_gt_dir)

    decimated_gt_dir = set_dir(SIM_PATH, 'temporal_label_%d' % TIME_SEGMENT_SIZE)
    decimate(temporal_gt_dir, decimated_gt_dir, 'logical_or', TIME_SEGMENT_SIZE)

    demix_dir = set_dir(SIM_PATH, 'demixed')
    demix(decimated_gt_dir, demix_dir, filename)

    eval_dir = set_dir(SIM_PATH, 'evaluated')
    evaluate(demix_dir, spatial_gt_dir, data_dir, eval_dir, filename)

elif(mode == 'train'):
    data_dir = set_dir(SIM_PATH, 'data')
    temporal_gt_dir = set_dir(SIM_PATH, 'temporal_label')
    spatial_gt_dir = set_dir(SIM_PATH, 'spatial_label')
    simulate(NUM_VIDEOS, data_dir, temporal_gt_dir, spatial_gt_dir)

    decimated_gt_dir = set_dir(SIM_PATH, 'temporal_label_%d' % TIME_SEGMENT_SIZE)
    decimate(temporal_gt_dir, decimated_gt_dir, 'logical_or', TIME_SEGMENT_SIZE, filename)

    preprocess_dir = set_dir(SIM_PATH, 'preprocessed')
    correction_dir = set_dir(SIM_PATH, 'corrected')
    preprocess(data_dir, preprocess_dir, correction_dir, filename)

    average_dir = set_dir(SIM_PATH, 'average_%d' % TIME_SEGMENT_SIZE)
    decimate(correction_dir, average_dir, 'mean', TIME_SEGMENT_SIZE, filename)
    model_dir = pathlib.Path(MODEL_PATH)
    segment_dir = set_dir(SIM_PATH, 'segmented')
    validate_dir = set_dir(SIM_PATH, 'validate')
    train([preprocess_dir, average_dir], decimated_gt_dir, model_dir,
          segment_dir, validate_dir)

    demix_dir = set_dir(SIM_PATH, 'demixed')
    demix(segment_dir, demix_dir, correction_dir, filename)

    eval_dir = set_dir(SIM_PATH, 'evaluated')
    evaluate(demix_dir, spatial_gt_dir, average_dir, eval_dir, filename)
    
elif(mode == 'run'):
    data_dir = pathlib.Path(DATA_PATH)
    preprocess_dir = set_dir(REAL_PATH, 'preprocessed')
    correction_dir = set_dir(REAL_PATH, 'corrected')
    preprocess(data_dir, preprocess_dir, correction_dir, filename)
    
    average_dir = set_dir(REAL_PATH, 'average_%d' % TIME_SEGMENT_SIZE)
    decimate(correction_dir, average_dir, 'mean', TIME_SEGMENT_SIZE, filename)
    model_dir = pathlib.Path(MODEL_PATH)
    segment_dir = set_dir(REAL_PATH, 'segmented')
    reference_dir = set_dir(REAL_PATH, 'reference')
    segment([preprocess_dir, average_dir], model_dir,
            segment_dir, reference_dir, filename)
    
    demix_dir = set_dir(REAL_PATH, 'demixed')
    demix(segment_dir, demix_dir, correction_dir, filename)
    
    gt_dir = pathlib.Path(GT_PATH)
    eval_dir = set_dir(REAL_PATH, 'evaluated')
    evaluate(demix_dir, gt_dir, average_dir, eval_dir, filename)
