import os
import sys
import importlib
import pathlib
import multiprocessing as mp

from simulate import create_synthetic_data, decimate_video
from preproc import run_preprocessing
from segment import train_model, validate_model, apply_model
from demix import compute_masks
from evaluate import run_ipynb_evaluate_each, run_ipynb_evaluate_all


if(len(sys.argv) != 2):
    print(sys.argv[0] + ' params(.py)')
    sys.exit(0)

params_name = pathlib.Path(sys.argv[1]).stem
params = importlib.import_module(params_name)



def set_dir(base_path, dirname):
    p = pathlib.Path(base_path, dirname)
    if not p.exists():
        p.mkdir()
    return p
    

def simulate(num_videos, data_dir, temporal_gt_dir, spatial_gt_dir):
    num_neurons_list = list(range(params.NUM_CELLS_MIN, params.NUM_CELLS_MAX))
    args = []
    for i in range(num_videos):
        name = '%4.4d' % i
        num_neurons = num_neurons_list[i % len(num_neurons_list)]
        args.append((params.IMAGE_SHAPE, params.TIME_FRAMES,
                     params.TIME_SEGMENT_SIZE, num_neurons,
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
        out_file = out_dir.joinpath(in_file.name)
        correction_file = correction_dir.joinpath(in_file.name)
        run_preprocessing(in_file, out_file, correction_file)


def train(in_dirs, target_dir, model_dir, out_dir, ref_dir):
    seed = 0
    validation_ratio = 5
    train_model(in_dirs, target_dir, model_dir,
                seed, validation_ratio,
                params.PATCH_SHAPE, params.NUM_DARTS,
                params.BATCH_SIZE, params.EPOCHS)
    validate_model(in_dirs, target_dir, model_dir, out_dir, ref_dir,
                   seed, validation_ratio,
                   params.PATCH_SHAPE, params.VALIDATION_TILE_STRIDES,
                   params.BATCH_SIZE)


def segment(in_dirs, model_dir, out_dir, ref_dir, filename):
    apply_model(in_dirs, model_dir, out_dir, ref_dir, filename,
                params.PATCH_SHAPE, params.INFERENCE_TILE_STRIDES,
                params.BATCH_SIZE)


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




if(params.RUN_MODE == 'train'):
    # simulate and create training data sets
    data_dir = set_dir(params.BASE_PATH, 'data')
    temporal_gt_dir = set_dir(params.BASE_PATH, 'temporal_label')
    spatial_gt_dir = set_dir(params.BASE_PATH, 'spatial_label')
    if(params.RUN_SIMULATE):
        simulate(params.NUM_VIDEOS, data_dir, temporal_gt_dir, spatial_gt_dir)

    # decimate temporal labels
    decimated_gt_dir = set_dir(params.BASE_PATH,
                               'temporal_label_%d' % params.TIME_SEGMENT_SIZE)
    decimate(temporal_gt_dir, decimated_gt_dir, 'logical_or',
             params.TIME_SEGMENT_SIZE, params.FILENAME)

    # preprocess images
    preprocess_dir = set_dir(params.BASE_PATH, 'preprocessed')
    correction_dir = set_dir(params.BASE_PATH, 'corrected')
    if(params.RUN_PREPROC):
        preprocess(data_dir, preprocess_dir, correction_dir, params.FILENAME)

    # decimate corrected images to produce average images
    average_dir = set_dir(params.BASE_PATH,
                          'average_%d' % params.TIME_SEGMENT_SIZE)
    decimate(correction_dir, average_dir, 'mean',
             params.TIME_SEGMENT_SIZE, params.FILENAME)
    
    # train the U-Net
    model_dir = pathlib.Path(params.MODEL_PATH)
    segment_dir = set_dir(params.BASE_PATH, 'segmented')
    validate_dir = set_dir(params.BASE_PATH, 'validate')
    if(params.RUN_TRAIN):
        train([preprocess_dir, average_dir], decimated_gt_dir, model_dir,
              segment_dir, validate_dir)

    # demix cells from U-Net outputs
    demix_dir = set_dir(params.BASE_PATH, 'demixed')
    if(params.RUN_DEMIX):
        demix(segment_dir, demix_dir, correction_dir, params.FILENAME)

    # evaluate the accuracy of detections
    eval_dir = set_dir(params.BASE_PATH, 'evaluated')
    if(params.RUN_EVALUATE):
        evaluate(demix_dir, spatial_gt_dir, average_dir, eval_dir,
                 params.FILENAME)


elif(params.RUN_MODE == 'run'):
    # preprocess images
    data_dir = pathlib.Path(params.INPUT_PATH)
    preprocess_dir = set_dir(params.PREPROC_PATH, 'preprocessed')
    correction_dir = set_dir(params.PREPROC_PATH, 'corrected')
    if(params.RUN_PREPROC):
        preprocess(data_dir, preprocess_dir, correction_dir, params.FILENAME)
    
    # decimate corrected images to produce average images
    average_dir = set_dir(params.BASE_PATH,
                          'average_%d' % params.TIME_SEGMENT_SIZE)
    decimate(correction_dir, average_dir, 'mean',
             params.TIME_SEGMENT_SIZE, params.FILENAME)

    # segment neurons
    model_dir = pathlib.Path(params.MODEL_PATH)
    segment_dir = set_dir(params.BASE_PATH, 'segmented')
    reference_dir = set_dir(params.BASE_PATH, 'reference')
    if(params.RUN_SEGMENT):
        segment([preprocess_dir, average_dir], model_dir,
                segment_dir, reference_dir, params.FILENAME)
    
    # demix cells from U-Net outputs
    demix_dir = set_dir(params.BASE_PATH, 'demixed')
    if(params.RUN_DEMIX):
        demix(segment_dir, demix_dir, correction_dir, params.FILENAME)
    
    # evaluate the accuracy of detections
    gt_dir = pathlib.Path(params.GT_PATH)
    eval_dir = set_dir(params.BASE_PATH, 'evaluated')
    if(params.RUN_EVALUATE):
        evaluate(demix_dir, gt_dir, average_dir, eval_dir, params.FILENAME)


else:
    print('Unexpected RUN_MODE: ' + params.RUN_MODE)
    sys.exit(1)
