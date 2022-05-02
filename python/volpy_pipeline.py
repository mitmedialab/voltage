#!/usr/bin/env python
# coding: utf-8

# (Started on 2022-04-01)
# 
# Our goal is to reuse the dataset used in VolPy to train our own segmentation algorithm.
# 
# Problem: volpy's segmentation tries to identify all the neurons present in an image, and therefore their ground truths
# reflect that. Our segmentation only identifies neurons that actually activate during a movie sequence. Therefore, in
# order to reuse VolPy datasets, we need to remove from their groundtruths the neurons it detects but that are not
# firing.
# 
# This file

"""
Demo pipeline for processing voltage imaging data. The processing pipeline
includes motion correction, memory mapping, segmentation, denoising and source
extraction. The demo shows how to construct the params, MotionCorrect and VOLPY 
objects and call the relevant functions. See inside for detail.
Dataset courtesy of Karel Svoboda Lab (Janelia Research Campus).
author: @caichangjia
"""
import cv2
import logging
import numpy as np
import os
import caiman as cm
from caiman.motion_correction import MotionCorrect
from caiman.source_extraction.volpy.volparams import volparams
from caiman.source_extraction.volpy.volpy import VOLPY
from caiman.base.rois import nf_read_roi_zip

try:
    cv2.setNumThreads(0)
except:
    pass


# # %%  Load demo movie and ROIs
# fnames = download_demo('demo_voltage_imaging.hdf5', 'volpy')  # file path to movie file (will download if not present)
# path_ROIs = download_demo('demo_voltage_imaging_ROIs.hdf5', 'volpy')  # file path to ROIs file (will download if not present)
# file_dir = os.path.split(fnames)[0]


basedir = "/home/yves/Projects/active/Fixstars/datasets/VolPy_dataset/"

# [filename, framerate, Voltron]


datasets = [
                ["TEG.01.02", 300, True],
                ["TEG.02.01", 300, True],
                ["TEG.03.01", 300, True],
                ["L1.00.00", 400, True],
                ["L1.01.00", 400, True],
                ["L1.01.35", 400, True],
                ["L1.02.00", 400, True],
                ["L1.02.80", 400, True],
                ["L1.03.00", 400, True],
                ["L1.03.35", 400, True],
                ["L1.04.00", 400, True],
                ["L1.04.50", 400, True],
                ["HPC.29.04", 1000, False],
                ["HPC.29.06", 1000, False],
                ["HPC.32.01", 1000, False],
                ["HPC.38.03", 1000, False],
                ["HPC.38.05", 1000, False],
                ["HPC.39.03", 1000, False],
                ["HPC.39.04", 1000, False],
                ["HPC.39.07", 1000, False],
                ["HPC.48.01", 1000, False],
                ["HPC.48.05", 1000, False],
                ["HPC.48.07", 1000, False],
                ["HPC.48.08", 1000, False]
]

logging.basicConfig(format=
                    "%(relativeCreated)12d [%(filename)s:%(funcName)20s():%(lineno)s]" \
                    "[%(process)d] %(message)s",
                    level=logging.INFO)

for directory, framerate, flip_signal in datasets:
    fnames = basedir+"/"+directory+"/"+directory+".tif"
    path_ROIs = basedir+"/"+directory+"/"+directory+"_ROI.zip"
    file_dir = os.path.split(fnames)[0]

    print('memmap_' + os.path.splitext(os.path.split(fnames)[-1])[0])

    # m_orig = cm.load(fnames)

    do_motion_correction = True
    do_memory_mapping = True

    if len([file for file in os.listdir(file_dir) if file.startswith("memmap_")])>0:
        do_motion_correction = False
        do_memory_mapping = False

    if len([file for file in os.listdir(file_dir) if file.endswith(".mmap") and "order_F" in file]) > 0:
        do_motion_correction = False

    print(do_motion_correction, do_memory_mapping)
    # dataset dependent parameters
    fr = framerate                                        # sample rate of the movie

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
        'border_nan': border_nan
    }

    opts = volparams(params_dict=opts_dict)



    # %% start a cluster for parallel processing
    c, dview, n_processes = cm.cluster.setup_cluster(
        backend='local', n_processes=2, single_thread=True)


    # %%% MOTION CORRECTION
    # first we create a motion correction object with the specified parameters
    mc = MotionCorrect(fnames, dview=dview, **opts.get_group('motion'))
    # Run correction
    if do_motion_correction:
        mc.motion_correct(save_movie=True)
    else:
        mc_list = [file for file in os.listdir(file_dir) if
                   (os.path.splitext(os.path.split(fnames)[-1])[0] in file and '.mmap' in file)]
        mc.mmap_file = [os.path.join(file_dir, mc_list[0])]
        print(f'reuse previously saved motion corrected file:{mc.mmap_file}')

    if do_memory_mapping:
        border_to_0 = 0 if mc.border_nan == 'copy' else mc.border_to_0
        # you can include the boundaries of the FOV if you used the 'copy' option
        # during motion correction, although be careful about the components near
        # the boundaries
        # memory map the file in order 'C'
        print(file_dir + '/memmap_' + directory)
        print("mc.mmap_file=" + str(mc.mmap_file))
        fname_new = cm.save_memmap_join(mc.mmap_file, base_name=file_dir + '/memmap_' + directory,
                                        add_to_mov=border_to_0, dview=dview)  # exclude border
    else:
        mmap_list = [file for file in os.listdir(file_dir) if
                     ('memmap_' + os.path.splitext(os.path.split(fnames)[-1])[0]) in file and file.endswith(".mmap")]
        fname_new = os.path.join(file_dir, mmap_list[0])
        print(f'reuse previously saved memory mapping file:{fname_new}')


    import imagesize
    width, height = imagesize.get(fnames)
    ROIs = nf_read_roi_zip(path_ROIs, (height, width)) # (364, 320))

    # %% restart cluster to clean up memory
    cm.stop_server(dview=dview)
    c, dview, n_processes = cm.cluster.setup_cluster(
        backend='local', n_processes=2, single_thread=True, maxtasksperchild=1)


    index = list(range(len(ROIs)))                # index of neurons
    weights = None                                # if None, use ROIs for initialization; to reuse weights check reuse weights block

    context_size = 35                             # number of pixels surrounding the ROI to censor from the background PCA
    visualize_ROI = False                         # whether to visualize the region of interest inside the context region
    flip_signal = True                            # Important!! Flip signal or not, True for Voltron indicator, False for others
    hp_freq_pb = 1 / 3                            # parameter for high-pass filter to remove photobleaching
    clip = 100                                    # maximum number of spikes to form spike template
    threshold_method = 'adaptive_threshold'       # adaptive_threshold or simple
    min_spikes= 10                                # minimal spikes to be found
    pnorm = 0.5                                   # a variable deciding the amount of spikes chosen for adaptive threshold method
    threshold = 3                                 # threshold for finding spikes only used in simple threshold method, Increase the threshold to find less spikes
    do_plot = False                               # plot detail of spikes, template for the last iteration
    ridge_bg= 0.01                                # ridge regression regularizer strength for background removement, larger value specifies stronger regularization
    sub_freq = 20                                 # frequency for subthreshold extraction
    weight_update = 'ridge'                       # ridge or NMF for weight update
    n_iter = 2                                    # number of iterations alternating between estimating spike times and spatial filters

    opts_dict={'fnames': fname_new,
               'ROIs': ROIs,
               'index': index,
               'weights': weights,
               'context_size': context_size,
               'visualize_ROI': visualize_ROI,
               'flip_signal': flip_signal,
               'hp_freq_pb': hp_freq_pb,
               'clip': clip,
               'threshold_method': threshold_method,
               'min_spikes':min_spikes,
               'pnorm': pnorm,
               'threshold': threshold,
               'do_plot':do_plot,
               'ridge_bg':ridge_bg,
               'sub_freq': sub_freq,
               'weight_update': weight_update,
               'n_iter': n_iter}

    opts.change_params(params_dict=opts_dict);


    #%% TRACE DENOISING AND SPIKE DETECTION
    vpy = VOLPY(n_processes=n_processes, dview=dview, params=opts)
    vpy.fit(n_processes=n_processes, dview=dview)


    vpy.estimates["locality"]
    vpy.estimates["num_spikes"]

    #%% save the result in .npy format
    save_result = True
    if save_result:
        vpy.estimates['ROIs'] = ROIs
        vpy.estimates['params'] = opts
        save_name = f'volpy_{os.path.split(fnames)[1][:-4]}_{threshold_method}'
        np.save(os.path.join(file_dir, save_name), vpy.estimates)

    # %% STOP CLUSTER and clean up log files
    cm.stop_server(dview=dview)

