import numpy as np
import pandas as pd
import tifffile as tiff
from pathlib import Path

from .f1score import count_matches, calc_f1_scores
from .roi import read_roi


NUM_THRESHOLDS = 100

def evaluate_each(in_file, gt_file, img_file, out_dir):

    eval_data = {}

    eval_masks = tiff.imread(in_file).astype(bool)
    # If there is no predicted cell, the image has one black page,
    # which should be removed in the subsequent evaluation.
    if(len(eval_masks) == 1 and not np.any(eval_masks[0])):
        eval_masks = np.zeros((0,) + eval_masks.shape[1:])
    gt_masks = read_roi(gt_file, eval_masks.shape[1:])
    eval_data['eval_masks'] = eval_masks
    eval_data['gt_masks'] = gt_masks
    eval_data['thumbnail'] = tiff.imread(img_file, key=0) # first page

    thresholds = np.array(range(1, NUM_THRESHOLDS+1)) / NUM_THRESHOLDS
    counts, IoU = count_matches(eval_masks, gt_masks, thresholds)
    f1, precision, recall = calc_f1_scores(counts)
    eval_data['thresholds'] = thresholds
    eval_data['IoU'] = IoU
    eval_data['f1'] = f1
    eval_data['precision'] = precision
    eval_data['recall'] = recall
    
    index = ['Prediction_%2.2d' % i for i in range(len(eval_masks))]
    columns = ['GT_%2.2d' % i for i in range(len(gt_masks))]
    df = pd.DataFrame(IoU, index=index, columns=columns)
    df.to_csv(Path(out_dir, Path(in_file).stem + '_IoU.csv'))

    df = pd.DataFrame(counts, columns=['TruePos', 'FalsePos', 'FalseNeg'])
    df.insert(0, 'IoU_Thresh', thresholds)
    df['Precision'] = precision
    df['Recall'] = recall
    df['F1'] = f1
    df.to_csv(Path(out_dir, Path(in_file).stem + '_counts.csv'), index=False)
    
    return eval_data
