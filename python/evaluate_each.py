import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import tifffile as tiff
from scipy.ndimage import center_of_mass
try:
    from python.evaluation_helper import load_roi, roi2masks_from_demix_result, count_matches, calc_f1_scores
    from python.file_helper import fread, fwrite
except:
    from evaluation_helper import load_roi, roi2masks_from_demix_result, count_matches, calc_f1_scores
    from file_helper import fread, fwrite
from webcolors import name_to_rgb
import pathlib
import cv2

def create_contour_overlay(contour_image, color):
    color = name_to_rgb(color)
    overlay = np.zeros(contour_image.shape + (4,), dtype='uint8')
    overlay[:, :, 0] = color[0]
    overlay[:, :, 1] = color[1]
    overlay[:, :, 2] = color[2]
    overlay[:, :, 3] = contour_image * 255
    return overlay

def blend_images(bg, fg):

    bg = cv2.cvtColor(bg, cv2.COLOR_RGB2RGBA)
    
    bg = bg.astype('float32')    
    bg = bg - bg[:,:,:3].min()
    bg = bg / bg[:,:,:3].max()
    bg[:,:,3][bg[:,:,3] > 0] = 1
    
    fg = fg.astype('float32')
    fg = fg - fg[:,:,:3].min()
    fg = fg / fg[:,:,:3].max()
    fg[:,:,3][fg[:,:,3] > 0] = 1
    
    
    mask = fg[:,:,3]    
    bg[np.where(mask == 1)] = fg[np.where(mask == 1)]
    
    return bg[:,:,:3]

def get_f1_score(settings, fname):

    eval_data = {}

    SUMMARY_IMAGE_DIR = settings['summary_image_dir']
    CONSENSUS_GT_DIR = settings['consensus_gt_dir']
    DEMIX_DIR = settings['output_base_path'] + '/' + settings['cell_demixing_result_path'] + '/'
    BASENAME = pathlib.Path(fname).stem

    summary_image = fread(SUMMARY_IMAGE_DIR + '/' + fname)
    eval_data['summary_image'] = summary_image

    with open(DEMIX_DIR + '/' + BASENAME + '.json') as f:
        demix_result = json.load(f)

    eval_mask_images, eval_contour_image,_ = roi2masks_from_demix_result(demix_result, summary_image.shape)
    gt_mask_images, gt_contour_image,_ = load_roi(CONSENSUS_GT_DIR, BASENAME, summary_image.shape)

    thresholds = np.array(range(0, 100, 1)) / 100
    counts, IoU = count_matches(eval_mask_images, gt_mask_images, thresholds)
    f1, precision, recall = calc_f1_scores(counts)

    eval_data['consensus_thresholds'] = thresholds
    eval_data['consensus_f1'] = f1
    eval_data['consensus_precision'] = precision
    eval_data['consensus_recall'] = recall

    return eval_data


def evaluate_file(settings, fname, save=True):

    eval_data = {}

    SUMMARY_IMAGE_DIR = settings['summary_image_dir']
    CONSENSUS_GT_DIR = settings['consensus_gt_dir']
    INDIVIDUAL_GT_DIR = settings['individual_gt_dir']
    GT_IDS = settings['individual_gt_ids']
    REPRESENTATIVE_IOU = settings['representative_iou']
    eval_data['representative_iou'] = REPRESENTATIVE_IOU
    eval_data['gt_ids'] = GT_IDS

    DEMIX_DIR = settings['output_base_path'] + '/' + settings['cell_demixing_result_path'] + '/'
    EVAL_DIR = settings['output_base_path'] + '/' + settings['evaluation_result_path'] + '/'

    BASENAME = pathlib.Path(fname).stem

    summary_image = fread(SUMMARY_IMAGE_DIR + '/' + fname)
    eval_data['summary_image'] = summary_image


    with open(DEMIX_DIR + '/' + BASENAME + '.json') as f:
        demix_result = json.load(f)

    eval_mask_images, eval_contour_image, eval_roi_idxs = roi2masks_from_demix_result(demix_result, summary_image.shape)
    if(save == True):
        fwrite(EVAL_DIR + '/' + fname, eval_mask_images)
    annotations = []
    for j in range(len(eval_roi_idxs)):
        roi_idxs = eval_roi_idxs[j]
        for i in range(1):
            annotations.append((roi_idxs[0][-1], roi_idxs[1][-1], 'Pred_' + str(j)))

    eval_data['num_eval_masks'] = len(eval_mask_images)
    overlay = create_contour_overlay(eval_contour_image, 'yellow')
    eval_data['output_rois'] = blend_images(summary_image, overlay)
    eval_data['output_roi_ann'] = list(annotations)

    gt_mask_images, gt_contour_image, gt_roi_idxs = load_roi(CONSENSUS_GT_DIR, BASENAME, summary_image.shape)
    annotations = []
    for j in range(len(gt_roi_idxs)):
        roi_idxs = gt_roi_idxs[j]
        for i in range(1):
            annotations.append((roi_idxs[0][i], roi_idxs[1][i], 'GT_' + str(j)))

    eval_data['num_gt_masks'] = len(gt_mask_images)
    overlay = create_contour_overlay(gt_contour_image, 'cyan')
    eval_data['gt_rois'] = blend_images(summary_image, overlay)
    eval_data['gt_roi_ann'] = list(annotations)

    thresholds = np.array(range(0, 100, 1)) / 100
    counts, IoU = count_matches(eval_mask_images, gt_mask_images, thresholds)
    f1, precision, recall = calc_f1_scores(counts)
    eval_data['consensus_thresholds'] = thresholds
    eval_data['consensus_f1'] = f1
    eval_data['consensus_precision'] = precision
    eval_data['consensus_recall'] = recall
    eval_data['consensus_IoU'] = IoU
    overlay = create_contour_overlay(gt_contour_image, 'cyan')
    eval_data['output_vs_gt'] = blend_images(eval_data['output_rois'], overlay)
    eval_data['output_vs_gt_ann'] = eval_data['output_roi_ann'] + eval_data['gt_roi_ann']

    index = ['Prediction_%2.2d' % i for i in range(eval_data['num_eval_masks'])]
    columns = ['GT_%2.2d' % i for i in range(eval_data['num_gt_masks'])]
    df = pd.DataFrame(IoU, index=index, columns=columns)
    eval_data['prediction_df'] = df
    if(save == True):
        df.to_csv(EVAL_DIR + '/' + BASENAME + '_IoU.csv')

    df = pd.DataFrame(counts, columns=['TruePos', 'FalsePos', 'FalseNeg'])
    df.insert(0, 'IoU_Thresh', thresholds)
    df['Precision'] = precision
    df['Recall'] = recall
    df['F1'] = f1
    eval_data['stats_df'] = df
    if(save == True):    
        df.to_csv(EVAL_DIR + '/' + BASENAME + '_stats.csv', index=False)

    scores = {}
    for gt_id in GT_IDS:
        eval_data[gt_id] = {}
        gt_mask_images, gt_contour_image, gt_roi_idxs = load_roi(INDIVIDUAL_GT_DIR + '/' + gt_id, BASENAME, summary_image.shape)
        annotations = []
        for j in range(len(gt_roi_idxs)):
            roi_idxs = gt_roi_idxs[j]
            for i in range(1):
                annotations.append((roi_idxs[0][i], roi_idxs[1][i], 'GT_' + str(j)))

        overlay = create_contour_overlay(gt_contour_image, 'cyan')
        overlay2 = blend_images(summary_image, overlay)
        overlay = create_contour_overlay(eval_contour_image, 'yellow')
        eval_data[gt_id]['output'] = blend_images(overlay2, overlay)
        eval_data[gt_id]['output_ann'] = eval_data['output_roi_ann'] + annotations

        counts, _ = count_matches(eval_mask_images, gt_mask_images, thresholds)
        f1, precision, recall = calc_f1_scores(counts)
        eval_data[gt_id]['scores'] = (f1, precision, recall)
        
        df = pd.DataFrame(counts, columns=['TruePos', 'FalsePos', 'FalseNeg'])
        df.insert(0, 'IoU_Thresh', thresholds)
        df['Precision'] = precision
        df['Recall'] = recall
        df['F1'] = f1
        if(save == True):
            df.to_csv(EVAL_DIR + '/' + BASENAME + '_' + gt_id + '.csv', index=False)

    return eval_data





    
