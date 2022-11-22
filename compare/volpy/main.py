#!/usr/bin/env python
"""
A function to run VolPy motion correction and cell segmentation.
Adapted from CaImAn/demos/general/demo_pipeline_voltage_imaging.py.
The original comments to follow.

Demo pipeline for processing voltage imaging data. The processing pipeline
includes motion correction, memory mapping, segmentation, denoising and source
extraction. The demo shows how to construct the params, MotionCorrect and VOLPY 
objects and call the relevant functions. See inside for detail.
Dataset courtesy of Karel Svoboda Lab (Janelia Research Campus).
author: @caichangjia
"""
import cv2
import glob
import logging
import numpy as np
import tifffile as tiff
import os
import sys
import time
from pathlib import Path

try:
    cv2.setNumThreads(0)
except:
    pass

try:
    if __IPYTHON__:
        # this is used for debugging purposes only. allows to reload classes
        # when changed
        get_ipython().magic('load_ext autoreload')
        get_ipython().magic('autoreload 2')
except NameError:
    pass

import caiman as cm
from caiman.motion_correction import MotionCorrect
from caiman.source_extraction.volpy import utils
from caiman.source_extraction.volpy.volparams import volparams
from caiman.summary_images import local_correlations_movie_offline
from caiman.summary_images import mean_image
from caiman.utils.utils import download_model

# %%
# Set up the logger (optional); change this if you like.
# You can log to a file using the filename parameter, or make the output more
# or less verbose by setting level to logging.DEBUG, logging.INFO,
# logging.WARNING, or logging.ERROR
logging.basicConfig(format=
                    "%(relativeCreated)12d [%(filename)s:%(funcName)20s():%(lineno)s]" \
                    "[%(process)d] %(message)s",
                    level=logging.INFO)

# %%
def run_volpy_segmentation(input_file, output_dir,
                           frame_rate, min_size, max_size,
                           do_motion_correction, do_summary_creation,
                           weights_path=''):
    pass  # For compatibility between running under Spyder and the CLI

    # %%  Load demo movie and ROIs
    #fnames = download_demo('demo_voltage_imaging.hdf5', 'volpy')  # file path to movie file (will download if not present)
    fnames = str(input_file)
    file_dir = os.path.split(fnames)[0]

    input_file = Path(input_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
#%% dataset dependent parameters
    # dataset dependent parameters
    #fr = 400                                        # sample rate of the movie
    fr = frame_rate

    # motion correction parameters
    pw_rigid = False                                # flag for pw-rigid motion correction
    gSig_filt = (3, 3)                              # size of filter, in general gSig (see below),
                                                    # change this one if algorithm does not work
    max_shifts = (5, 5)                             # maximum allowed rigid shift
    strides = (48, 48)                              # start a new patch for pw-rigid motion correction every x pixels
    overlaps = (24, 24)                             # overlap between pathes (size of patch strides+overlaps)
    max_deviation_rigid = 3                         # maximum deviation allowed for patch with respect to rigid shifts
    border_nan = 'copy'

    opts_dict = {
        'fnames': fnames,
        'fr': fr,
        'pw_rigid': pw_rigid,
        'max_shifts': max_shifts,
        'gSig_filt': gSig_filt,
        'strides': strides,
        'overlaps': overlaps,
        'max_deviation_rigid': max_deviation_rigid,
        'border_nan': border_nan,
        'use_cuda': False,
    }

    opts = volparams(params_dict=opts_dict)

# %% start a cluster for parallel processing
    c, dview, n_processes = cm.cluster.setup_cluster(
        backend='local', n_processes=None, single_thread=False)

# %%% MOTION CORRECTION
    # first we create a motion correction object with the specified parameters
    mc = MotionCorrect(fnames, dview=dview, **opts.get_group('motion'))
    # Run correction
    if do_motion_correction:
        tic = time.perf_counter()
        mc.motion_correct(save_movie=True)
        toc = time.perf_counter()
        with open('time_motion_correct.txt', 'w') as f:
            f.write('%f\n' % (toc - tic))
        mv = cm.load(mc.mmap_file[0])
        tiff.imwrite(output_dir.joinpath(input_file.stem + '_corrected.tif'),
                     mv, photometric='minisblack')
    else: 
        mc_list = [file for file in os.listdir(file_dir) if 
                   (os.path.splitext(os.path.split(fnames)[-1])[0] in file and '.mmap' in file)]
        mc.mmap_file = [os.path.join(file_dir, mc_list[0])]
        print(f'reuse previously saved motion corrected file:{mc.mmap_file}')

# %% SEGMENTATION
    # create summary images
    if do_summary_creation:
        tic = time.perf_counter()
        img = mean_image(mc.mmap_file[0], window=1000, dview=dview)
        img = (img-np.mean(img))/np.std(img)

        gaussian_blur = False # Use gaussian blur when there is too much noise in the video
        Cn = local_correlations_movie_offline(mc.mmap_file[0], fr=fr, window=fr*4,
                                              stride=fr*4, winSize_baseline=fr,
                                              remove_baseline=True,
                                              gaussian_blur=gaussian_blur,
                                              dview=dview).max(axis=0)
        img_corr = (Cn-np.mean(Cn))/np.std(Cn)
        summary_images = np.stack([img, img, img_corr], axis=0).astype(np.float32)
        toc = time.perf_counter()
        with open('time_summary_images.txt', 'w') as f:
            f.write('%f\n' % (toc - tic))
        tiff.imwrite(output_dir.joinpath(input_file.stem + '_summary_images.tif'),
                     summary_images, photometric='minisblack')
        # below is for compatibility with our evaluation framework
        tiff.imwrite(output_dir.joinpath(input_file.stem + '_spatial.tif'),
                     summary_images[0], photometric='minisblack')
    else:
        summary_images = tiff.imread(output_dir.joinpath(input_file.stem + '_summary_images.tif'))

    if(not weights_path):
        weights_path = download_model('mask_rcnn')
    tic = time.perf_counter()
    ROIs = utils.mrcnn_inference(img=summary_images.transpose([1, 2, 0]),
                                 size_range=[min_size, max_size], # was [5, 22]
                                 weights_path=weights_path, display_result=False)
    toc = time.perf_counter()
    with open('time_segmentation.txt', 'w') as f:
        f.write('%f\n' % (toc - tic))
    tiff.imwrite(output_dir.joinpath(input_file.stem + '_masks.tif'),
                 ROIs, photometric='minisblack')

# %% STOP CLUSTER and clean up log files
    cm.stop_server(dview=dview)
    log_files = glob.glob('*_LOG_*')
    for log_file in log_files:
        os.remove(log_file)


input_file = sys.argv[1]
output_dir = sys.argv[2]
frame_rate = int(sys.argv[3])
min_size = int(sys.argv[4])
max_size = int(sys.argv[5])
do_motion_correction = bool(sys.argv[6])
do_summary_creation = bool(sys.argv[7])
weights_path = sys.argv[8] if len(sys.argv) > 8 else ''
run_volpy_segmentation(input_file, output_dir, frame_rate, min_size, max_size,
                       do_motion_correction, do_summary_creation, weights_path)
