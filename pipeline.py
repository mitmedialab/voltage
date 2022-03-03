import sys
import time
import runpy
import pathlib
import multiprocessing as mp

from simulate import create_synthetic_data, decimate_video
from correct import correct_video
from preproc import preprocess_video
from segment import train_model, validate_model, apply_model
from demix import compute_masks
from evaluate import run_ipynb_evaluate_each, run_ipynb_evaluate_all


if(len(sys.argv) != 2):
    print(sys.argv[0] + ' params.py')
    sys.exit(0)

params = runpy.run_path(sys.argv[1])
params.setdefault('FIRST_FRAME', 0)
params.setdefault('SIGNAL_METHOD', 'max-med')
params.setdefault('SIGNAL_BINNING', 1)


def set_dir(base_path, dirname):
    p = pathlib.Path(base_path, dirname)
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


def decimate(in_dir, out_dir, mode, size, filename):
    tic = time.perf_counter()
    
    if(filename):
        in_file = in_dir.joinpath(filename + '.tif')
        out_file = out_dir.joinpath(in_file.name)
        decimate_video(in_file, out_file, mode, size)
    else:
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


def correct(in_dir, correction_dir, filename):
    tic = time.perf_counter()

    if(filename): # file mode, multi-threaded job for a single file
        in_file = in_dir.joinpath(filename + '.tif')
        correction_file = correction_dir.joinpath(in_file.name)
        motion_file = correction_dir.joinpath(in_file.stem + '_motion.hdf5')
        correct_video(in_file, correction_file, motion_file,
                      motion_search_level=params['MOTION_SEARCH_LEVEL'],
                      motion_search_size=params['MOTION_SEARCH_SIZE'],
                      motion_patch_size=params['MOTION_PATCH_SIZE'],
                      motion_patch_offset=params['MOTION_PATCH_OFFSET'])
        
    else: # batch mode, single-threaded jobs for multiple files
        filenames = sorted(in_dir.glob('*.tif'))
        args = []
        for in_file in filenames:
            correction_file = correction_dir.joinpath(in_file.name)
            motion_file = correction_dir.joinpath(in_file.stem + '_motion.hdf5')
            args.append((in_file, correction_file, motion_file,
                         0,
                         params['MOTION_SEARCH_LEVEL'],
                         params['MOTION_SEARCH_SIZE'],
                         params['MOTION_PATCH_SIZE'],
                         params['MOTION_PATCH_OFFSET'], 1000, 0))

        pool = mp.Pool(mp.cpu_count())
        pool.starmap(correct_video, args)
        pool.close()

    toc = time.perf_counter()
    print('%.1f seconds to correct' % (toc - tic))


def preprocess(in_dir, temporal_dir, spatial_dir, filename):
    tic = time.perf_counter()

    if(filename): # file mode, multi-threaded job for a single file
        in_file = in_dir.joinpath(filename + '.tif')
        temporal_file = temporal_dir.joinpath(in_file.name)
        spatial_file = spatial_dir.joinpath(in_file.name)
        preprocess_video(in_file,
                         temporal_file, spatial_file,
                         params['SIGNAL_METHOD'],
                         params['TIME_SEGMENT_SIZE'],
                         params['SIGNAL_SCALE'])
        
    else: # batch mode, single-threaded jobs for multiple files
        filenames = sorted(in_dir.glob('*.tif'))
        args = []
        for in_file in filenames:
            temporal_file = temporal_dir.joinpath(in_file.name)
            spatial_file = spatial_dir.joinpath(in_file.name)
            args.append((in_file,
                         temporal_file, spatial_file,
                         params['SIGNAL_METHOD'],
                         params['TIME_SEGMENT_SIZE'],
                         params['SIGNAL_SCALE'],
                         1, 1))

        pool = mp.Pool(mp.cpu_count())
        pool.starmap(preprocess_video, args)
        pool.close()

    toc = time.perf_counter()
    print('%.1f seconds to preprocess' % (toc - tic))


def train(in_dirs, target_dir, model_file, log_file, out_dir, ref_dir):
    tic = time.perf_counter()

    seed = 0
    validation_ratio = 5
    train_model(in_dirs, target_dir, model_file, log_file,
                seed, validation_ratio,
                params['MODEL_IO_SHAPE'], params['NUM_DARTS'],
                params['BATCH_SIZE'], params['EPOCHS'])
    validate_model(in_dirs, target_dir, model_file, out_dir, ref_dir,
                   seed, validation_ratio,
                   params['MODEL_IO_SHAPE'], params['TILE_STRIDES'],
                   params['BATCH_SIZE'])

    toc = time.perf_counter()
    print('%.1f seconds to train' % (toc - tic))


def segment(in_dirs, model_dir, out_dir, ref_dir, filename):
    tic = time.perf_counter()
    
    apply_model(in_dirs, model_dir, out_dir, ref_dir, filename,
                params['TILE_SHAPE'], params['TILE_STRIDES'],
                params['BATCH_SIZE'])
    
    toc = time.perf_counter()
    print('%.1f seconds to segment' % (toc - tic))


def demix(in_dir, out_dir, correction_dir, filename):
    tic = time.perf_counter()
    
    if(filename):
        in_file = in_dir.joinpath(filename + '.tif')
        out_file = out_dir.joinpath(in_file.name)
        corr_file = correction_dir.joinpath(in_file.name)
        compute_masks(in_file, corr_file, out_file,
                      num_threads=params['NUM_THREADS_DEMIXING'])
    else:
        filenames = sorted(in_dir.glob('*.tif'))
        args = []
        for in_file in filenames:
            out_file = out_dir.joinpath(in_file.name)
            corr_file = correction_dir.joinpath(in_file.name)
            args.append((in_file, corr_file, out_file))

        pool = mp.Pool(mp.cpu_count())
        pool.starmap(compute_masks, args)
        pool.close()
    
    toc = time.perf_counter()
    print('%.1f seconds to demix' % (toc - tic))


def evaluate(in_dir, gt_dir, img_dir, out_dir, filename):
    tic = time.perf_counter()

    if(filename):
        filenames = [in_dir.joinpath(filename + '.tif')]
    else:
        filenames = sorted(in_dir.glob('*.tif'))
    # notebook can't do multiprocessing, run one file at a time
    for in_file in filenames:
        print('evaluating ' + in_file.stem)
        gt_file = gt_dir.joinpath(in_file.name)
        img_file = img_dir.joinpath(in_file.name)
        run_ipynb_evaluate_each(in_file, gt_file, img_file, out_dir, in_file.stem)
    run_ipynb_evaluate_all(out_dir)

    toc = time.perf_counter()
    print('%.1f seconds to evaluate' % (toc - tic))



if(params['RUN_MODE'] == 'train'):
    # simulate and create training data sets
    data_dir = set_dir(params['DATA_DIR'], 'data')
    temporal_gt_dir = set_dir(params['DATA_DIR'], 'temporal_label')
    spatial_gt_dir = set_dir(params['DATA_DIR'], 'spatial_label')
    decimated_gt_dir = set_dir(params['DATA_DIR'],
                               'temporal_label_%d' % params['TIME_SEGMENT_SIZE'])
    if(params['RUN_SIMULATE']):
        simulate(params['NUM_VIDEOS'], data_dir, temporal_gt_dir, spatial_gt_dir)
        decimate(temporal_gt_dir, decimated_gt_dir, 'logical_or',
                 params['TIME_SEGMENT_SIZE'], params['FILENAME'])

    # correct images
    correction_dir = set_dir(params['PREPROC_DIR'], 'corrected')
    if(params['RUN_CORRECT']):
        correct(data_dir, correction_dir, params['FILENAME'])

    # preprocess images
    temporal_dir = set_dir(params['PREPROC_DIR'], 'temporal')
    spatial_dir = set_dir(params['PREPROC_DIR'], 'spatial')
    if(params['RUN_PREPROC']):
        preprocess(correction_dir, temporal_dir, spatial_dir,
                   params['FILENAME'])

    # train the U-Net
    model_dir = pathlib.Path(params['MODEL_DIR'])
    model_file = model_dir.joinpath('model.h5')
    log_file = model_dir.joinpath('log.csv')
    segment_dir = set_dir(params['OUTPUT_DIR'], 'segmented')
    validate_dir = set_dir(params['OUTPUT_DIR'], 'validate')
    if(params['RUN_TRAIN']):
        train([temporal_dir, spatial_dir], decimated_gt_dir,
              model_file, log_file, segment_dir, validate_dir)

    # demix cells from U-Net outputs
    demix_dir = set_dir(params['OUTPUT_DIR'], 'demixed')
    if(params['RUN_DEMIX']):
        demix(segment_dir, demix_dir, correction_dir, params['FILENAME'])

    # evaluate the accuracy of detections
    eval_dir = set_dir(params['OUTPUT_DIR'], 'evaluated')
    if(params['RUN_EVALUATE']):
        evaluate(demix_dir, spatial_gt_dir, spatial_dir, eval_dir,
                 params['FILENAME'])


elif(params['RUN_MODE'] == 'run'):
    for i, filename in enumerate(params['INPUT_FILES']):
        tag = filename.stem
        out_dir = set_dir(params['OUTPUT_DIR'], tag)

        # correct images
        correction_file = out_dir.joinpath(tag + '_corrected.tif')
        motion_file = out_dir.joinpath(tag + '_motion.hdf5')
        if(params['RUN_CORRECT']):
            tic = time.perf_counter()
            correct_video(filename, correction_file, motion_file,
                          first_frame=params['FIRST_FRAME'],
                          motion_search_level=params['MOTION_SEARCH_LEVEL'],
                          motion_search_size=params['MOTION_SEARCH_SIZE'],
                          motion_patch_size=params['MOTION_PATCH_SIZE'],
                          motion_patch_offset=params['MOTION_PATCH_OFFSET'])
            toc = time.perf_counter()
            print('%.1f seconds to correct' % (toc - tic))

        # extract signal
        temporal_file = out_dir.joinpath(tag + '_temporal.tif')
        spatial_file = out_dir.joinpath(tag + '_spatial.tif')
        if(params['RUN_PREPROC']):
            tic = time.perf_counter()
            preprocess_video(correction_file,
                             temporal_file, spatial_file,
                             params['SIGNAL_METHOD'],
                             params['TIME_SEGMENT_SIZE'],
                             params['SIGNAL_SCALE'],
                             params['SIGNAL_BINNING'])
            toc = time.perf_counter()
            print('%.1f seconds to preprocess' % (toc - tic))

        # segment neurons
        segment_file = out_dir.joinpath(tag + '_segmented.tif')
        reference_file = out_dir.joinpath(tag + '_reference.tif')
        if(params['RUN_SEGMENT']):
            tic = time.perf_counter()
            apply_model([temporal_file, spatial_file], params['MODEL_FILE'],
                        segment_file, reference_file,
                        params['TILE_SHAPE'], params['TILE_STRIDES'],
                        params['BATCH_SIZE'], params['GPU_MEM_SIZE'])
            toc = time.perf_counter()
            print('%.1f seconds to segment' % (toc - tic))

        # demix cells from U-Net outputs
        demix_file = out_dir.joinpath(tag + '_masks.tif')
        if(params['RUN_DEMIX']):
            tic = time.perf_counter()
            compute_masks(segment_file, correction_file, demix_file,
                          num_threads=params['NUM_THREADS_DEMIXING'],
                          **params)
            toc = time.perf_counter()
            print('%.1f seconds to compute masks' % (toc - tic))

        # evaluate the accuracy of detections
        if(params['RUN_EVALUATE']):
            gt_file = params['GT_FILES'][i]
            run_ipynb_evaluate_each(demix_file, gt_file, spatial_file,
                                    out_dir, tag)

    if(params['RUN_EVALUATE']):
        run_ipynb_evaluate_all(params['OUTPUT_DIR'])


else:
    print('Unexpected RUN_MODE: ' + params['RUN_MODE'])
    sys.exit(1)
