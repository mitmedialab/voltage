"""
Script for running Invivo-Imaging (SGPMD-NMF) in one shot.
It combines invivo-imaging/denoise/main.m and invivo-imaging/demix/main.ipynb
in one script. For the former (main.m), it is rewritten in such a way that
the MATLAB scripts are called from Python rather than the other way around.
The relevant parts of invivo-imaging/denoise/main.m are extracted and put in
./run_normcorre.m and ./correct_motion.m. For the latter (main.ipynb), all the
notebook cells except for visualization are copied to this script.
"""
import os
import time
import math
import numpy as np
from pathlib import Path
from skimage import io
from sklearn.decomposition import TruncatedSVD
from scipy.ndimage import center_of_mass
import torch

import sys
sys.path.append('invivo-imaging/demix')
import superpixel_analysis as sup


MATLAB = 'matlab -nodisplay -nosplash -nodesktop -r '


def run_command(command):
    print(command)
    os.system(command)


def run_invivo_segmentation(input_file, output_dir):

    input_file = Path(input_file).absolute()
    output_dir = Path(output_dir).absolute()
    output_dir.mkdir(exist_ok=True)

    #%% NoRMCorre
    command = MATLAB + f'"run_normcorre(\'{input_file}\', \'{output_dir}\'); exit;"'
    run_command(command)

    #%% denoising parameters
    mov_in = 'movReg.tif'
    detr_spacing = 5000; # in number of frames
    #row_blocks = 4;
    #col_blocks = 2;
    trunc_start = 1; # frame to start denoising
    #trunc_length = 5000; # length of movie segment to denoise on

    # modified parameter settings as follows
    raw_mov = io.imread(str(input_file))
    trunc_length = raw_mov.shape[0] - 100 # denoising removes first 100 frames
    max_block_size = 30
    row_blocks = 1
    while(raw_mov.shape[1] // row_blocks >= max_block_size):
        row_blocks += 1
    col_blocks = 1
    while(raw_mov.shape[2] // col_blocks >= max_block_size):
        col_blocks += 1

    #%% denoising
    args = (output_dir, mov_in, output_dir, detr_spacing,
            row_blocks, col_blocks, trunc_start-1, trunc_length)
    command = 'python invivo-imaging/denoise/denoise.py %s %s %s %d %d %d %d %d' % args
    run_command(command)

    #%% motion correction
    command = MATLAB + f'"correct_motion(\'{output_dir}\');exit;"'
    run_command(command)

    #%% Read in movie

    # input movie path
    path = str(output_dir)
    
    # read in motion corrected movie
    noise = np.squeeze(io.imread(path + '/Sn_image.tif'))
    [nrows, ncols] = noise.shape
    
    if os.path.isfile(path + '/motion_corrected.tif'):
        mov = io.imread(path + '/motion_corrected.tif').transpose(1, 2, 0)
    elif os.path.isfile(path + '/denoised.tif'):
        mov = io.imread(path + '/denoised.tif')
    else:
        raise ValueError('No valid input file')
    
    # read in the mask for blood
    if os.path.isfile(path + '/bloodmask.tif'):
        bloodmask = np.squeeze(io.imread(path + '/bloodmask.tif'))
        mov = mov * np.repeat(np.expand_dims(noise * bloodmask, 2), mov.shape[2], axis=2)
    else:
        mov = mov * np.repeat(np.expand_dims(noise,2), mov.shape[2], axis=2)

    #%% Spatial 2x2 Binning
    movB = mov.reshape(int(mov.shape[0] / 2), 2, int(mov.shape[1] / 2), 2, mov.shape[2])
    movB = np.mean(np.mean(movB, axis=1), axis=2)

    movB = mov # this seems to cancel the binning

    #%% Load Manually Initialized Background
    bg_flag = os.path.isfile(path + '/ff.tif')

    if bg_flag:
        # import manually initialized background components
        ff_ini = io.imread(path + '/ff.tif')
        fb_ini = io.imread(path + '/fb.tif')
    
        # bin the spatial components
        fb_ini = fb_ini.reshape(mov.shape[1], mov.shape[0], -1).transpose(1, 0, 2)
    #     fb_ini = fb_ini.reshape(int(fb_ini.shape[0] / 2), 2, int(fb_ini.shape[1] / 2), 2, fb_ini.shape[2])
    #     fb_ini = np.mean(np.mean(fb_ini, axis=1), axis=2)

    if bg_flag:
        # select which background components to use for initialization
        bkg_components = range(3)
    
        fb_ini = fb_ini[:, :, bkg_components].reshape(movB.shape[0] * movB.shape[1], len(bkg_components))
        ff_ini = ff_ini[:, bkg_components]

    #%% Get Cell Spatial Supports from High Pass Filtered Movie
    start = time.time()

    # select which window to demix on
    first_frame = 1
    #last_frame = 5000
    last_frame = mov.shape[2]

    movHP = sup.hp_filt_data(movB, spacing=10)
    
    rlt = sup.axon_pipeline_Y(movHP[:, :, first_frame:last_frame].copy(),
                              fb_ini = np.zeros(1),
                              ff_ini = np.zeros(1),

                              ##### Superpixel parameters
                              # thresholding level
                              th = [4],

                              # correlation threshold for finding superpixels
                              # (range around 0.8-0.99)
                              cut_off_point = [0.95],

                              # minimum pixel count of a superpixel
                              # don't need to change these unless cell sizes change
                              length_cut = [10],

                              # maximum pixel count of a superpixel
                              # don't need to change these unless cell sizes change
                              length_max = [1000],

                              patch_size = [30, 30],

                              # correlation threshold between superpixels for merging
                              # likely don't need to change this
                              residual_cut = [np.sqrt(1 - (0.8)**2)],

                              pass_num = 1, bg = False,

                              ##### Cell-finding, NMF parameters
                              # correlation threshold of pixel with superpixel trace to include pixel in cell
                              # (range 0.3-0.6)
                              corr_th_fix = 0.4,

                              # correlation threshold for merging two cells
                              # (default 0.8, but likely don't need to change)
                              merge_corr_thr = 0.8,

                              ##### Other options
                              # if True, only superpixel analysis run; if False, NMF is also run to find cells
                              sup_only = False,

                              # the number of superpixels to remove (starting from the dimmest)
                              remove = 0
                              )
    
    print('Demixing took: ' + str(time.time() - start) + ' sec')

    #%% Get Background Components from Unfiltered Movie

    # rank of background to model, if none selected
    bg_rank = 3

    #final_cells = [0,1,3,5,7]
    #nCells = len(final_cells)
    #a = rlt["fin_rlt"]["a"][:,final_cells].copy()
    #c = rlt["fin_rlt"]["c"][:,final_cells].copy()
    b = rlt["fin_rlt"]["b"].copy()
    # As the final_cells above looks arbitrary, use everything instead
    a = rlt["fin_rlt"]["a"].copy()
    c = rlt["fin_rlt"]["c"].copy()
    nCells = c.shape[1]

    dims = movB.shape[:2]
    T = last_frame - first_frame
    
    movVec = movB.reshape(np.prod(dims), -1, order='F')
    mov_min = movVec.min()
    if mov_min < 0:
        mov_min_pw = movVec.min(axis=1, keepdims=True)
        movVec -= mov_min_pw
    
    normalize_factor = np.std(movVec, axis=1, keepdims=True) * T
        
    if bg_flag:
        fb = fb_ini
        ff = ff_ini[first_frame:last_frame, :]
        bg_rank = fb.shape[1]
    else:
        bg_comp_pos = np.where(a.sum(axis=1) == 0)[0]
        y_temp = movVec[bg_comp_pos, first_frame:last_frame]
        fb = np.zeros([movVec.shape[0], bg_rank])
        y_temp = y_temp - y_temp.mean(axis=1, keepdims=True)
        svd = TruncatedSVD(n_components=bg_rank, n_iter=7, random_state=0)
        fb[bg_comp_pos, :] = svd.fit_transform(y_temp)
        ff = svd.components_.T
        ff = ff - ff.mean(axis=0, keepdims=True)
    
    a, c, b, fb, ff, res, corr_img_all_r, num_list = \
        sup.update_AC_bg_l2_Y(movVec[:, first_frame:last_frame].copy(),
                              normalize_factor, a, c, b, ff, fb, dims,
                              corr_th_fix=0.35,
                              maxiter=35, tol=1e-8,
                              merge_corr_thr=0.8,
                              merge_overlap_thr=0.8, keep_shape=True)

    #%% Choose Cells and Recover Temporal Correlation Structures
    def tv_norm(image):
        return np.sum(np.abs(image[:, :-1] - image[:, 1:])) + np.sum(np.abs(image[:-1, :] - image[1:, :]))
    
    Y = movB.transpose(1, 0, 2).reshape(movB.shape[0] * movB.shape[1], movB.shape[2])
    X = np.hstack((a, fb))
    X = X / np.ptp(X, axis=0)
    X2 = np.zeros((X.shape[0], nCells + bg_rank))
    X2[:, :nCells] = X[:, :nCells]

    lr = 0.001
    maxIters = 1000

    for b in range(bg_rank):
        bg_im = X[:, -(b+1)].reshape(movB.shape[-2::-1])

        weights = torch.zeros((nCells, 1), requires_grad=True, dtype=torch.double)

        image = torch.from_numpy(bg_im)

        for idx in range(maxIters):
            test_im = image - torch.reshape(torch.from_numpy(X[:, :nCells]) @ weights, movB.shape[-2::-1])
            tv = torch.sum(torch.abs(test_im[:, :-1] - test_im[:, 1:])) + torch.sum(torch.abs(test_im[:-1, :] - test_im[1:, :]))

            tv.backward()

            with torch.no_grad():
                weights -= lr * weights.grad
                
            weights.grad.zero_()

        opt_weights = weights.data.numpy()
        
        X2[:, -(b+1)] = np.maximum(X[:, -(b+1)] - np.squeeze(X[:, :nCells] @ opt_weights), 0)

    #%% Get Final Traces
    beta_hat2 = np.linalg.lstsq(X2, Y)[0]
    res = np.mean(np.square(Y - X2 @ beta_hat2), axis = 0)

    #%% Save Results
    suffix = ''

    io.imsave(path + '/spatial_footprints' + suffix + '.tif', X2)
    io.imsave(path + '/cell_spatial_footprints' + suffix + '.tif', X2[:, :nCells])
    io.imsave(path + '/temporal_traces' + suffix + '.tif', beta_hat2)
    io.imsave(path + '/cell_traces' + suffix + '.tif', beta_hat2[:nCells, :])
    io.imsave(path + '/residual_var' + suffix + '.tif', res)
    
    cell_locations = center_of_mass(X2[:, 0].reshape(movB.shape[1::-1]).transpose(1, 0))
    for idx in range(nCells - 1):
        cell_locations = np.vstack((cell_locations, 
                                    center_of_mass(X2[:, idx + 1].reshape(movB.shape[1::-1]).transpose(1, 0))))
    io.imsave(path + '/cell_locations' + suffix + '.tif', np.array(cell_locations))
    
    if nCells > 1:
        io.imsave(path + '/cell_demixing_matrix' + suffix + '.tif', 
                  np.linalg.inv(np.array(X2[:, :nCells].T @ X2[:, :nCells])) @ X2[:, :nCells].T)

    #%% Reshape footprints and generate segmentation masks
    # What follows is newly added to what's copied from main.m and main.ipynb above
    row_cut_lower = math.floor((raw_mov.shape[1] % (2 * row_blocks)) / 2)
    row_cut_upper = raw_mov.shape[1] - math.ceil((raw_mov.shape[1] % (2 * row_blocks)) / 2)
    col_cut_lower = math.floor((raw_mov.shape[2] % (2 * col_blocks)) / 2)
    col_cut_upper = raw_mov.shape[2] - math.ceil((raw_mov.shape[2] % (2 * col_blocks)) / 2)

    footprints = X2[:, :nCells].reshape(ncols, nrows, -1).transpose(2, 1, 0)
    #from skimage.filters import threshold_otsu
    footprints = np.zeros((nCells,) + raw_mov.shape[1:], dtype='float32')
    masks = np.zeros((nCells,) + raw_mov.shape[1:], dtype=bool)
    for i in range(nCells):
        fp = X2[:, i].reshape(ncols, nrows).transpose(1, 0)
        footprints[i, row_cut_lower:row_cut_upper,col_cut_lower:col_cut_upper] = fp
        #th = threshold_otsu(fp)
        th = 1e-8
        masks[i, row_cut_lower:row_cut_upper,col_cut_lower:col_cut_upper] = fp > th

    io.imsave(path + '/cell_spatial_footprints_reshaped' + suffix + '.tif', footprints)
    io.imsave(str(output_dir.joinpath(input_file.stem + '_mask.tif')), masks)


#%% Test
if __name__ == '__main__':
    INPUT_FILE = 'invivo-imaging/demo_data/raw_data.tif'
    OUTPUT_DIR = 'test'
    run_invivo_segmentation(INPUT_FILE, OUTPUT_DIR)

"""
INPUT_PATH = '/media/bandy/nvme_data/voltage/datasets_v0.5/lowmag'
OUTPUT_PATH = '/media/bandy/nvme_work/voltage/compare/invivo/lowmag'
input_files = sorted(Path(INPUT_PATH).glob('*.tif'))[9:]
Path(OUTPUT_PATH).mkdir(exist_ok=True)
for input_file in input_files:
    dataset_name = input_file.stem
    print('Processing ', dataset_name)
    output_dir = Path(OUTPUT_PATH).joinpath(dataset_name)
    run_invivo_segmentation(input_file, output_dir)
"""
