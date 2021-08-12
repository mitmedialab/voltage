import numpy as np
import pandas as pd
import tifffile as tiff
from pathlib import Path

from .f1score import count_matches, calc_f1_scores


"""
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

"""

def evaluate_each(in_file, gt_file, out_dir):

    eval_data = {}

    eval_masks = tiff.imread(in_file) > 0.5
    gt_masks = tiff.imread(gt_file)

    thresholds = np.array(range(0, 101, 1)) / 100
    counts, IoU = count_matches(eval_masks, gt_masks, thresholds)
    f1, precision, recall = calc_f1_scores(counts)
    eval_data['thresholds'] = thresholds
    eval_data['IoU'] = IoU
    eval_data['f1'] = f1
    eval_data['precision'] = precision
    eval_data['recall'] = recall
    #overlay = create_contour_overlay(gt_contour_image, 'cyan')
    #eval_data['output_vs_gt'] = blend_images(eval_data['output_rois'], overlay)
    #eval_data['output_vs_gt_ann'] = eval_data['output_roi_ann'] + eval_data['gt_roi_ann']
    
    index = ['Prediction_%2.2d' % i for i in range(len(eval_masks))]
    columns = ['GT_%2.2d' % i for i in range(len(gt_masks))]
    df = pd.DataFrame(IoU, index=index, columns=columns)
    df.to_csv(out_dir + '/' + Path(in_file).stem + '_IoU.csv')

    df = pd.DataFrame(counts, columns=['TruePos', 'FalsePos', 'FalseNeg'])
    df.insert(0, 'IoU_Thresh', thresholds)
    df['Precision'] = precision
    df['Recall'] = recall
    df['F1'] = f1
    df.to_csv(out_dir + '/' + Path(in_file).stem + '_counts.csv', index=False)
    
    return eval_data
