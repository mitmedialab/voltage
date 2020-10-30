#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
from read_roi import read_roi_zip, read_roi_file
import tifffile as tiff
from skimage import draw as skdraw
from skimage.segmentation import find_boundaries
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from scipy.linalg import block_diag
import os
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert import HTMLExporter



def roi2masks(roi_dict, image_shape):
    num_masks = len(roi_dict.keys())
    if(num_masks == 0):
        num_masks = 1
    mask_images = np.zeros((num_masks, ) + image_shape, dtype=bool)
    contour_image = np.zeros(image_shape, dtype=bool)
    for i, key in enumerate(roi_dict):
        roi = roi_dict[key]
        r = np.array(roi['y'])
        c = np.array(roi['x'])

        rr, cc = skdraw.polygon(r, c, image_shape)
        mask_images[i][rr, cc] = True

        rr, cc = skdraw.polygon_perimeter(r, c, image_shape)
        contour_image[rr, cc] = True

    return mask_images, contour_image


def roi2masks_from_demix_result(roi_dict, image_shape):
    num_masks = len(roi_dict.keys())
    if(num_masks == 0):
        num_masks = 1
    mask_images = np.zeros((num_masks, ) + image_shape, dtype=bool)
    contour_image = np.zeros(image_shape, dtype=bool)
    for i, key in enumerate(roi_dict):
        roi = roi_dict[key]
        c = np.array(roi['idxy'].strip('][').split(', '), dtype=int) # convert text string
        r = np.array(roi['idxx'].strip('][').split(', '), dtype=int) # to a list of int
        mask_images[i][r, c] = True

        contour = find_boundaries(mask_images[i], mode='outer')
        contour_image = np.logical_or(contour_image, contour)

    return mask_images, contour_image


def load_roi(roi_path, roi_base, image_shape):
    roi_file = roi_path + '/' + roi_base + '.roi'
    roi_zip  = roi_path + '/' + roi_base + '.zip'
    roi_tiff = roi_path + '/' + roi_base + '.tif'

    if(os.path.isfile(roi_file)):
        roi_dict = read_roi_file(roi_file)
    elif(os.path.isfile(roi_zip)):
        roi_dict = read_roi_zip(roi_zip)
    elif(os.path.isfile(roi_tiff)):
        mask_images = tiff.imread(roi_tiff).astype(bool)
        contour_image = np.zeros(image_shape, dtype=bool)
        for mask in mask_images:
            contour = find_boundaries(mask, mode='outer')
            contour_image = np.logical_or(contour_image, contour)
        return mask_images, contour_image
    else:
        print('file not found: ' + roi_path + '/' + roi_base + '.{roi,zip,tif}')
        roi_dict = {}

    return roi2masks(roi_dict, image_shape)



def remove_duplicate_matches(matches):
    """
    Remove duplicate True's in a boolean matrix representing matches
    such that no row or column has more than one True.

    Parameters
    ----------
    matches : 2D numpy.ndarray of bool
        Boolean matrix representing matches between two sets.

    Returns
    -------
    result : 2D numpy.ndarray of bool
        Boolean matrix where duplicate True's are removed.
    """
    graph = csr_matrix(matches)
    cols = maximum_bipartite_matching(graph, perm_type='column')
    result = np.zeros(matches.shape, dtype=bool)
    for i, row in enumerate(result):
        c = cols[i]
        if(c >= 0): # c = -1 means no match
            row[c] = True

    return result


def count_matches(eval_masks, gt_masks, thresholds, remove_duplicates=True):
    """
    Count the numbers of (mis)matches between predicted masks to be evaluated
    and ground truth masks for a set of intersection-over-union (IoU) thresholds.

    Parameters
    ----------
    eval_masks : 3D numpy.ndarray of bool
        Array (first axis) of 2D mask images (second and third axes)
        representing predicted masks to be evaluated.
    gt_masks : 3D numpy.ndarray of bool
        Array of 2D mask images representing ground truth masks
        against which eval_masks will be evaluated.
    thresholds : list of float
        List of IoU thresholds. Matches will be counted for each threshold.
        The value range of thresholds is [0, 1].
    remove_duplicates : bool, optional
        Whether or not to remove duplicate matches before counting them.
        The default is True.

    Returns
    -------
    counts : list of 3-tuples
        Each element of the list is a tuple (true_pos, false_pos, false_neg)
        representing the numbers of true positives, false positives,
        and false negatives for each IoU threshold.
    IoU: 2D numpy.ndarray
        Matrix representing IoU between eval_masks and gt_masks
    """
    IoU = np.zeros((len(eval_masks), len(gt_masks)))
    for i, e_mask in enumerate(eval_masks):
        for j, g_mask in enumerate(gt_masks):
            I = np.logical_and(e_mask, g_mask)
            U = np.logical_or(e_mask, g_mask)
            IoU[i, j] = np.sum(I) / np.sum(U)

    counts = []
    for th in thresholds:
        matches = np.array(IoU > th)
        if(remove_duplicates):
            matches = remove_duplicate_matches(matches)
        gt_has_eval = np.any(matches, axis=0)
        eval_has_gt = np.any(matches, axis=1)
        true_pos = np.sum(eval_has_gt == True)
        # true_pos = np.sum(gt_has_eval == True) should yield the same value
        false_pos = np.sum(eval_has_gt == False)
        false_neg = np.sum(gt_has_eval == False)
        counts.append((true_pos, false_pos, false_neg))

    return counts, IoU


def calc_f1_scores(counts):
    true_pos  = np.array(counts)[:, 0]
    false_pos = np.array(counts)[:, 1]
    false_neg = np.array(counts)[:, 2]
    precision = true_pos / (true_pos + false_pos)
    recall = true_pos / (true_pos + false_neg)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(recall), where=(true_pos > 0))
    return f1, precision, recall


def evaluate_each(outdir_demix, outdir_eval, basename):
    with open('evaluate_each.ipynb') as f:
        data = f.read()
    data = data.replace('@@@DEMIX_DIR', outdir_demix)
    data = data.replace('@@@EVAL_DIR',  outdir_eval)
    data = data.replace('@@@BASENAME',  basename)
    nb = nbformat.reads(data, nbformat.NO_CONVERT)
    
    ep = ExecutePreprocessor(timeout=None)
    ep.preprocess(nb)
    with open(outdir_eval + '/' + basename + '.ipynb', 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

    he = HTMLExporter()
    he.template_name = 'classic'
    (body, resources) = he.from_notebook_node(nb)
    with open(outdir_eval + '/' + basename + '.html', 'w', encoding='utf-8') as f:
        f.write(body)


def evaluate_all(outdir):
    with open('evaluate_all.ipynb') as f:
        nb = nbformat.read(f, nbformat.NO_CONVERT)
    
    ep = ExecutePreprocessor(timeout=None)
    ep.preprocess(nb)
    with open(outdir + '/all.ipynb', 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

    he = HTMLExporter()
    he.template_name = 'classic'
    (body, resources) = he.from_notebook_node(nb)
    with open(outdir + '/all.html', 'w', encoding='utf-8') as f:
        f.write(body)

