#!/usr/bin/env python
"""
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
def run_volpy_segmentation(input_file, output_dir):
    pass  # For compatibility between running under Spyder and the CLI

    # %%  Load demo movie and ROIs
    #fnames = download_demo('demo_voltage_imaging.hdf5', 'volpy')  # file path to movie file (will download if not present)
    fnames = str(input_file)
    input_file = Path(input_file)
    output_dir = Path(output_dir)
    
#%% dataset dependent parameters
    # dataset dependent parameters
    fr = 400                                        # sample rate of the movie

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
        'use_cuda': True,
    }

    opts = volparams(params_dict=opts_dict)

# %% start a cluster for parallel processing
    c, dview, n_processes = cm.cluster.setup_cluster(
        backend='local', n_processes=None, single_thread=False)

# %%% MOTION CORRECTION
    # first we create a motion correction object with the specified parameters
    mc = MotionCorrect(fnames, dview=dview, **opts.get_group('motion'))
    # Run correction
    mc.motion_correct(save_movie=True)
    mv = cm.load(mc.mmap_file[0])
    tiff.imwrite(output_dir.joinpath(input_file.stem + '_corrected.tif'), mv, photometric='minisblack')

# %% SEGMENTATION
    # create summary images
    img = mean_image(mc.mmap_file[0], window = 1000, dview=dview)
    img = (img-np.mean(img))/np.std(img)
    
    gaussian_blur = False        # Use gaussian blur when there is too much noise in the video
    Cn = local_correlations_movie_offline(mc.mmap_file[0], fr=fr, window=fr*4, 
                                          stride=fr*4, winSize_baseline=fr, 
                                          remove_baseline=True, gaussian_blur=gaussian_blur,
                                          dview=dview).max(axis=0)
    img_corr = (Cn-np.mean(Cn))/np.std(Cn)
    summary_images = np.stack([img, img, img_corr], axis=0).astype(np.float32)
    # save summary images which are used in the VolPy GUI
    #cm.movie(summary_images).save(fnames[:-5] + '_summary_images.tif')
    tiff.imwrite(output_dir.joinpath(input_file.stem + '_summary_images.tif'), summary_images, photometric='minisblack')
    #fig, axs = plt.subplots(1, 2)
    #axs[0].imshow(summary_images[0]); axs[1].imshow(summary_images[2])
    #axs[0].set_title('mean image'); axs[1].set_title('corr image')

    #elif method == 'maskrcnn':                 # Important!! Make sure install keras before using mask rcnn. 
    weights_path = download_model('mask_rcnn')    # also make sure you have downloaded the new weight. The weight was updated on Dec 1st 2020.
    ROIs = utils.mrcnn_inference(img=summary_images.transpose([1, 2, 0]), size_range=[5, 22],
                                 weights_path=weights_path, display_result=False) # size parameter decides size range of masks to be selected
    #cm.movie(ROIs).save(fnames[:-5] + 'mrcnn_ROIs.hdf5')
    tiff.imwrite(output_dir.joinpath(input_file.stem + '_masks.tif'), ROIs, photometric='minisblack')

# %% STOP CLUSTER and clean up log files
    cm.stop_server(dview=dview)
    log_files = glob.glob('*_LOG_*')
    for log_file in log_files:
        os.remove(log_file)


input_dir = '/media/bandy/nvme_data/VolPy_Data/Extracted/voltage_TEG'
input_files = sorted(Path(input_dir).glob('*/*.tif'))
output_dir = '/media/bandy/nvme_work/voltage/volpyTEG_volpy'
for input_file in input_files:
    run_volpy_segmentation(input_file, output_dir)
