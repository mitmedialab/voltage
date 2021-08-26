import os
import glob
import ntpath
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
REAL_PATH = '/media/bandy/nvme_work/voltage/tmp'
DATA_PATH = '/media/bandy/nvme_data/ramdas/VI/SelectedData_v0.2/WholeTifs'
GT_PATH = '/media/bandy/nvme_data/ramdas/VI/SelectedData_v0.2/GT_comparison/GTs_rev20201027/consensus'


def set_dir(base_path, dirname):
    p = pathlib.Path(base_path, dirname)
    if not p.exists():
        p.mkdir()
    return str(p) + '/'
    

def simulate(num_videos, data_dir, temporal_gt_dir, spatial_gt_dir):
    args = []
    for i in range(num_videos):
        name = '%4.4d' % i
        num_neurons = i % 10 + 5
        args.append((IMAGE_SHAPE, TIME_FRAMES, num_neurons,
                     data_dir, temporal_gt_dir, spatial_gt_dir, name))
    
    pool = mp.Pool(mp.cpu_count())
    pool.starmap(create_synthetic_data, args)
    pool.close()


def decimate(in_dir, out_dir, mode, size):
    filenames = glob.glob(in_dir + '/*.tif')
    filenames.sort()
    args = []
    for in_file in filenames:
        out_file = out_dir + ntpath.basename(in_file)
        args.append((in_file, out_file, mode, size))
    
    pool = mp.Pool(mp.cpu_count())
    pool.starmap(decimate_video, args)
    pool.close()


def preprocess(in_dir, out_dir, correction_dir):
    filenames = glob.glob(in_dir + '/*.tif')
    filenames.sort()
    for in_file in filenames:
        command = 'preproc/main -db -ms 5 -sm 1 -sc 0 -ss 2 -sw %d ' % TIME_SEGMENT_SIZE + in_file + ' ' + out_dir
        print(command)
        os.system(command)
        
        out_file = ntpath.basename(in_file)
        command = 'mv ' + out_dir + '/signal.tif ' + out_dir + out_file
        os.system(command)
        command = 'mv ' + out_dir + '/corrected.tif ' + correction_dir + out_file
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


def demix(in_dir, out_dir, filename):
    if(filename):
        filenames = [os.path.join(in_dir, filename + '.tif')]
    else:
        filenames = glob.glob(in_dir + '/*.tif')
        filenames.sort()
    for in_file in filenames:
        out_file = out_dir + ntpath.basename(in_file)
        compute_masks(in_file, out_file)


def evaluate(in_dir, gt_dir, img_dir, out_dir, filename):
    if(filename):
        filenames = [os.path.join(in_dir, filename + '.tif')]
    else:
        filenames = glob.glob(in_dir + '*.tif')
        filenames.sort()
    for in_file in filenames:
        gt_file = os.path.join(gt_dir, ntpath.basename(in_file))
        img_file = os.path.join(img_dir, ntpath.basename(in_file))
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
    decimate(temporal_gt_dir, decimated_gt_dir, 'logical_or', TIME_SEGMENT_SIZE)

    preprocess_dir = set_dir(SIM_PATH, 'preprocessed')
    correction_dir = set_dir(SIM_PATH, 'corrected')
    preprocess(data_dir, preprocess_dir, correction_dir)

    average_dir = set_dir(SIM_PATH, 'average_%d' % TIME_SEGMENT_SIZE)
    decimate(correction_dir, average_dir, 'mean', TIME_SEGMENT_SIZE)
    segment_dir = set_dir(SIM_PATH, 'segmented')
    validate_dir = set_dir(SIM_PATH, 'validate')
    train([preprocess_dir, average_dir], decimated_gt_dir, MODEL_PATH,
          segment_dir, validate_dir)

    demix_dir = set_dir(SIM_PATH, 'demixed')
    demix(segment_dir, demix_dir, filename)

    eval_dir = set_dir(SIM_PATH, 'evaluated')
    evaluate(demix_dir, spatial_gt_dir, average_dir, eval_dir, filename)
    
elif(mode == 'run'):
    preprocess_dir = set_dir(REAL_PATH, 'preprocessed')
    correction_dir = set_dir(REAL_PATH, 'corrected')
    preprocess(DATA_PATH, preprocess_dir, correction_dir)
    
    average_dir = set_dir(REAL_PATH, 'average_%d' % TIME_SEGMENT_SIZE)
    decimate(correction_dir, average_dir, 'mean', TIME_SEGMENT_SIZE)
    segment_dir = set_dir(REAL_PATH, 'segmented')
    reference_dir = set_dir(REAL_PATH, 'reference')
    segment([preprocess_dir, average_dir], MODEL_PATH,
            segment_dir, reference_dir, filename)
    
    demix_dir = set_dir(REAL_PATH, 'demixed')
    demix(segment_dir, demix_dir, filename)
    
    eval_dir = set_dir(REAL_PATH, 'evaluated')
    evaluate(demix_dir, GT_PATH, average_dir, eval_dir, filename)
