import os
import sys
import time
import runpy
import tifffile as tiff
import multiprocessing as mp
from pathlib import Path

from simulate import create_synthetic_data, decimate_video
from correct import correct_video
from preproc import preprocess_video
from segment import VI_Segment
from demix import compute_masks
from evaluate import read_roi, run_ipynb_evaluate_each, run_ipynb_evaluate_all, Timer
from spike import detect_spikes


if(len(sys.argv) != 2):
    print(sys.argv[0] + ' params.py')
    sys.exit(0)

default_param_file = Path(sys.argv[0]).parent.joinpath('params/defaults.py')
params = runpy.run_path(default_param_file)
user_params = runpy.run_path(sys.argv[1])
params.update(user_params) # overwrite default values with user parameters


# set # GPUs if specified in the parameters, otherwise use all available GPUs
if('NUM_GPUS' in params):
    n = params['NUM_GPUS']
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in range(n)])
    if(n == 0):
        params['USE_GPU_CORRECT'] = False



def set_dir(base_path, dirname):
    p = Path(base_path, dirname)
    if not p.exists():
        p.mkdir()
    return p


def simulate(num_videos, data_dir, temporal_gt_dir, spatial_gt_dir):
    tic = time.perf_counter()

    num_neurons_list = list(range(params['NUM_CELLS_MIN'], params['NUM_CELLS_MAX']))
    args = []
    for i in range(num_videos):
        name = '%4.4d' % i
        num_neurons = num_neurons_list[i % len(num_neurons_list)]
        args.append((params['IMAGE_SHAPE'], params['TIME_FRAMES'],
                     params['TIME_SEGMENT_SIZE'], num_neurons,
                     data_dir, temporal_gt_dir, spatial_gt_dir, name))
    
    pool = mp.Pool(mp.cpu_count())
    pool.starmap(create_synthetic_data, args)
    pool.close()
    
    toc = time.perf_counter()
    print('%.1f seconds to simulate' % (toc - tic))


def decimate(in_dir, out_dir, mode, size):
    tic = time.perf_counter()

    filenames = sorted(in_dir.glob('*.tif'))
    args = []
    for in_file in filenames:
        out_file = out_dir.joinpath(in_file.name)
        args.append((in_file, out_file, mode, size))

    pool = mp.Pool(mp.cpu_count())
    pool.starmap(decimate_video, args)
    pool.close()

    toc = time.perf_counter()
    print('%.1f seconds to decimate' % (toc - tic))


def correct(in_dir, correction_dir):
    tic = time.perf_counter()

    filenames = sorted(in_dir.glob('*.tif'))
    args = []
    for in_file in filenames:
        correction_file = correction_dir.joinpath(in_file.name)
        motion_file = correction_dir.joinpath(in_file.stem + '_motion.hdf5')
        args.append((in_file, motion_file, correction_file,
                     0, True,
                     params['MOTION_SEARCH_LEVEL'],
                     params['MOTION_SEARCH_SIZE'],
                     params['MOTION_PATCH_SIZE'],
                     params['MOTION_PATCH_OFFSET'],
                     1.0, 1.0,
                     params['TIME_SEGMENT_SIZE'],
                     False, 0, 1)) # use one core of CPU

    pool = mp.Pool(mp.cpu_count())
    pool.starmap(correct_video, args)
    pool.close()

    toc = time.perf_counter()
    print('%.1f seconds to correct' % (toc - tic))


def preprocess(in_dir, temporal_dir, spatial_dir):
    tic = time.perf_counter()

    filenames = sorted(in_dir.glob('*.tif'))
    args = []
    for in_file in filenames:
        temporal_file = temporal_dir.joinpath(in_file.name)
        spatial_file = spatial_dir.joinpath(in_file.name)
        args.append((in_file, None,
                     temporal_file, spatial_file,
                     params['SIGNAL_METHOD'],
                     params['TIME_SEGMENT_SIZE'],
                     params['SIGNAL_SCALE'],
                     1.0, 1)) # use one core of CPU

    pool = mp.Pool(mp.cpu_count())
    pool.starmap(preprocess_video, args)
    pool.close()

    toc = time.perf_counter()
    print('%.1f seconds to preprocess' % (toc - tic))


def train(in_dirs, target_dir, model_dir, log_file, out_dir, ref_dir):
    tic = time.perf_counter()

    seed = 0
    validation_ratio = 5
    segmenter = VI_Segment()
    segmenter.set_training(in_dirs, target_dir, seed, validation_ratio,
                           params['MODEL_IO_SHAPE'],
                           params['NORM_CHANNEL'], params['NORM_SHIFTS'])
    segmenter.train(model_dir, log_file, params['NUM_DARTS'],
                    params['BATCH_SIZE'], params['EPOCHS'])
    segmenter.validate(model_dir, out_dir, ref_dir,
                       params['TILE_STRIDES'],
                       params['BATCH_SIZE'])

    toc = time.perf_counter()
    print('%.1f seconds to train' % (toc - tic))


def demix(in_dir, out_dir, spatial_dir):
    tic = time.perf_counter()

    filenames = sorted(in_dir.glob('*.tif'))
    args = []
    for in_file in filenames:
        out_file = out_dir.joinpath(in_file.name)
        img_file = spatial_dir.joinpath(in_file.name)
        args.append((None, None, in_file, img_file, out_file,
                     params['PROBABILITY_THRESHOLD'],
                     params['AREA_THRESHOLD_MIN'],
                     params['AREA_THRESHOLD_MAX'],
                     params['CONCAVITY_THRESHOLD'],
                     params['INTENSITY_THRESHOLD'],
                     params['ACTIVITY_THRESHOLD'],
                     params['BACKGROUND_SIGMA'],
                     params['BACKGROUND_EDGE'],
                     params['BACKGROUND_THRESHOLD'],
                     params['MASK_DILATION'],
                     None))

    pool = mp.Pool(mp.cpu_count())
    pool.starmap(compute_masks, args)
    pool.close()

    toc = time.perf_counter()
    print('%.1f seconds to demix' % (toc - tic))


def evaluate(in_dir, gt_dir, img_dir, out_dir):
    tic = time.perf_counter()

    filenames = sorted(in_dir.glob('*.tif'))
    # notebook can't do multiprocessing, run one file at a time
    for in_file in filenames:
        print('evaluating ' + in_file.stem)
        gt_file = gt_dir.joinpath(in_file.name)
        img_file = img_dir.joinpath(in_file.name)
        out_subdir = set_dir(out_dir, in_file.stem)
        run_ipynb_evaluate_each(in_file, gt_file, img_file, None, out_subdir,
                                params['REPRESENTATIVE_IOU'], in_file.stem)
    run_ipynb_evaluate_all(out_dir, params['REPRESENTATIVE_IOU'])

    toc = time.perf_counter()
    print('%.1f seconds to evaluate' % (toc - tic))



if(params['RUN_MODE'] == 'train'):
    if(not params['BASE_DIR']):
        print('Please specify BASE_DIR')
        sys.exit(0)

    # simulate and create training data sets
    synth_dir = set_dir(params['BASE_DIR'], 'synthetic')
    data_dir = set_dir(synth_dir, 'data')
    temporal_gt_dir = set_dir(synth_dir, 'temporal_label')
    spatial_gt_dir = set_dir(synth_dir, 'spatial_label')
    decimated_gt_dir = set_dir(synth_dir,
                               'temporal_label_%d' % params['TIME_SEGMENT_SIZE'])
    if(params['RUN_SIMULATE']):
        simulate(params['NUM_VIDEOS'], data_dir, temporal_gt_dir, spatial_gt_dir)
        decimate(temporal_gt_dir, decimated_gt_dir, 'logical_or',
                 params['TIME_SEGMENT_SIZE'])

    # correct images
    preproc_dir = set_dir(params['BASE_DIR'], 'preproc')
    correction_dir = set_dir(preproc_dir, 'corrected')
    if(params['RUN_CORRECT']):
        correct(data_dir, correction_dir)

    # preprocess images
    temporal_dir = set_dir(preproc_dir, 'temporal')
    spatial_dir = set_dir(preproc_dir, 'spatial')
    if(params['RUN_PREPROC']):
        preprocess(correction_dir, temporal_dir, spatial_dir)

    # train the U-Net
    model_dir = set_dir(params['BASE_DIR'], 'model')
    log_file = model_dir.joinpath('log.csv')
    segment_dir = set_dir(params['BASE_DIR'], 'segmented')
    validate_dir = set_dir(params['BASE_DIR'], 'validate')
    if(params['RUN_TRAIN']):
        train([temporal_dir, spatial_dir], decimated_gt_dir,
              model_dir, log_file, segment_dir, validate_dir)

    # demix cells from U-Net outputs
    demix_dir = set_dir(params['BASE_DIR'], 'demixed')
    if(params['RUN_DEMIX']):
        demix(segment_dir, demix_dir, spatial_dir)

    # evaluate the accuracy of detections
    eval_dir = set_dir(params['BASE_DIR'], 'evaluated')
    if(params['RUN_EVALUATE']):
        evaluate(demix_dir, spatial_gt_dir, spatial_dir, eval_dir)


elif(params['RUN_MODE'] == 'run'):

    if(params['RUN_SEGMENT']):
        segmenter = VI_Segment()
        segmenter.set_inference(params['MODEL_FILE'])

    for i, filename in enumerate(params['INPUT_FILES']):
        tag = filename.stem
        print('')
        print('Processing ' + tag)
        out_dir = set_dir(params['OUTPUT_DIR'], tag)
        timer = Timer(out_dir.joinpath(tag + '_times.csv'))

        # correct images
        correction_file = out_dir.joinpath(tag + '_corrected.tif')
        motion_file = out_dir.joinpath(tag + '_motion.hdf5')
        if(params['RUN_CORRECT']):
            timer.start()
            c = correct_video(filename, motion_file, None,
                              params['FIRST_FRAME'], params['NORMALIZE'],
                              params['MOTION_SEARCH_LEVEL'],
                              params['MOTION_SEARCH_SIZE'],
                              params['MOTION_PATCH_SIZE'],
                              params['MOTION_PATCH_OFFSET'],
                              params['MOTION_X_RANGE'],
                              params['MOTION_Y_RANGE'],
                              params['SHADING_PERIOD'],
                              params['USE_GPU_CORRECT'],
                              params['BATCH_SIZE_CORRECT'],
                              params['NUM_THREADS_CORRECT'])
            timer.stop('Correct')
        else:
            c = tiff.imread(correction_file).astype('float32')
            timer.skip('Correct')

        # extract signal
        temporal_file = out_dir.joinpath(tag + '_temporal.tif')
        spatial_file = out_dir.joinpath(tag + '_spatial.tif')
        if(params['RUN_PREPROC']):
            timer.start()
            preprocess_video(None, c, temporal_file, spatial_file,
                             params['SIGNAL_METHOD'],
                             params['TIME_SEGMENT_SIZE'],
                             params['SIGNAL_SCALE'],
                             params['SIGNAL_DOWNSAMPLING'],
                             params['NUM_THREADS_PREPROC'])
            timer.stop('Preproc')
        else:
            timer.skip('Preproc')

        # segment neurons
        segment_file = out_dir.joinpath(tag + '_segmented.tif')
        reference_file = out_dir.joinpath(tag + '_reference.tif')
        if(params['RUN_SEGMENT']):
            timer.start()
            segmenter.predict([temporal_file, spatial_file],
                              segment_file, None, # pass reference_file if needed
                              params['NORM_CHANNEL'], params['NORM_SHIFTS'],
                              params['TILE_SHAPE'], params['TILE_STRIDES'],
                              params['TILE_MARGIN'],
                              params['BATCH_SIZE'], params['GPU_MEM_SIZE'])
            timer.stop('Segment')
        else:
            timer.skip('Segment')

        # demix cells from U-Net outputs
        demix_file = out_dir.joinpath(tag + '_demix.tif')
        if(params['RUN_DEMIX']):
            timer.start()
            compute_masks(None, None,
                          segment_file, spatial_file, demix_file,
                          params['PROBABILITY_THRESHOLD'],
                          params['AREA_THRESHOLD_MIN'],
                          params['AREA_THRESHOLD_MAX'],
                          params['CONCAVITY_THRESHOLD'],
                          params['INTENSITY_THRESHOLD'],
                          params['ACTIVITY_THRESHOLD'],
                          params['BACKGROUND_SIGMA'],
                          params['BACKGROUND_EDGE'],
                          params['BACKGROUND_THRESHOLD'],
                          params['MASK_DILATION'],
                          c.shape[1:])
            timer.stop('Mask')
        else:
            timer.skip('Mask')

        # extract voltage traces, detect spikes, and remove inactive neurons
        spike_file = out_dir.joinpath(tag + '_spikes.hdf5')
        mask_file = out_dir.joinpath(tag + '_masks.tif')
        if(params['RUN_SPIKE']):
            timer.start()
            masks = read_roi(demix_file, None)
            detect_spikes(c, masks, spike_file, mask_file,
                          params['POLARITY'], params['SPIKE_THRESHOLD'],
                          params['REMOVE_INACTIVE'])
            timer.stop('Spike')
        else:
            timer.skip('Spike')

        # save speed stats
        timer.save(filename)

        # save large intermediate file later
        if(params['RUN_CORRECT']):
            tiff.imwrite(correction_file, c, photometric='minisblack')

        # evaluate the accuracy of detections
        if(params['RUN_EVALUATE']):
            gt_file = params['GT_FILES'][i]
            run_ipynb_evaluate_each(mask_file, gt_file, correction_file, #spatial_file,
                                    spike_file, out_dir,
                                    params['REPRESENTATIVE_IOU'], tag)

    if(params['RUN_EVALUATE']):
        run_ipynb_evaluate_all(params['OUTPUT_DIR'],
                               params['REPRESENTATIVE_IOU'])

else:
    print('Unexpected RUN_MODE: ' + params['RUN_MODE'])
    sys.exit(1)
