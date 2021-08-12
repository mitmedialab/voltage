import glob
import numpy as np
import pandas as pd
from pathlib import Path

from .f1score import calc_f1_scores


REPRESENTATIVE_IOU = 0.4


def aggregate_scores(out_dir, filter_func=lambda x: True):

    dataset_ids = []
    precision_each = []
    recall_each = []
    f1_each = []

    first = True
    filenames = glob.glob(out_dir + '/*.html')
    filenames.sort()
    for in_file in filenames:
        #if(not filter_func(param)):
        #    continue
        basename = Path(in_file).stem
        if(basename == 'all'): # skip preexisting aggregation result
            continue
        df = pd.read_csv(out_dir + '/' + basename + '_counts.csv')
        if(first):
            df_sum = df[['TruePos', 'FalsePos', 'FalseNeg']]
            thresholds = df['IoU_Thresh']
            indices = np.where(thresholds >= REPRESENTATIVE_IOU)
            representative_iou_index = indices[0][0]
            first = False
        else:
            df_sum += df
        
        dataset_ids.append(basename)
        precision_each.append(df['Precision'][representative_iou_index])
        recall_each.append(df['Recall'][representative_iou_index])
        f1_each.append(df['F1'][representative_iou_index])

    f1_all, precision_all, recall_all = calc_f1_scores(df_sum)
    f1_rep = f1_all[representative_iou_index]

    df_sum.insert(0, 'IoU_Thresh', thresholds)
    df_sum['Precision'] = precision_all
    df_sum['Recall'] = recall_all
    df_sum['F1'] = f1_all
    df_sum.to_csv(out_dir + '/all_counts.csv', index=False)
    
    df_each = pd.DataFrame(dataset_ids, columns=['Dataset'])
    #df_each['MagnificationValue'] = magnifications
    #df_each['Magnification'] = magnifications
    df_each['Precision'] = precision_each
    df_each['Recall'] = recall_each
    df_each['F1'] = f1_each
    
    return f1_rep, df_sum, df_each


def evaluate_all(out_dir):
    
    eval_data = {}
    
    f1_rep, df_sum, df_each = aggregate_scores(out_dir)

    eval_data['representative f1'] = f1_rep
    
    eval_data['thresholds'] = df_sum['IoU_Thresh']
    eval_data['f1_all'] = df_sum['F1']
    eval_data['precision_all'] = df_sum['Precision']
    eval_data['recall_all'] = df_sum['Recall']

    eval_data['dataset'] = df_each['Dataset']
    eval_data['f1_each'] = df_each['F1']
    eval_data['precision_each'] = df_each['Precision']
    eval_data['recall_each'] = df_each['Recall']

    return eval_data
