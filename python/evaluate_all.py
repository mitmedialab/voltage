import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import ntpath
import os
try:
    from python.evaluation_helper import calc_f1_scores
except:
    from evaluation_helper import calc_f1_scores
import os
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert import HTMLExporter


def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def aggregate_scores(settings, eval_info, filter_func=lambda x: True, suffix='stats'):
   
    try:
        with open('file_params.txt') as f:
            params = json.load(f)
    except:
        with open('../file_params.txt') as f:
            params = json.load(f)

    datasets = []
    magnifications = []
    precision_each = []
    recall_each = []
    f1_each = []
    first = True
    unique_mags = []
    for key, param in params.items():
        if(not filter_func(param)):
            continue
        eval_dir = settings['output_base_path']
        eval_dir += '/' + settings['evaluation_result_path']
        basename, _ = os.path.splitext(path_leaf(param['filename']))
        df = pd.read_csv(eval_dir + '/' + basename + '_' + suffix + '.csv')
        if(first):
            df_sum = df[['TruePos', 'FalsePos', 'FalseNeg']]
            thresholds = df['IoU_Thresh']
            indices = np.where(thresholds >= eval_info['representative_iou'])
            representative_iou_index = indices[0][0]
            first = False
        else:
            df_sum += df
        datasets.append(key)
        if(param['magnification'] not in unique_mags):
            unique_mags.append(param['magnification'])
        magnifications.append(str(param['magnification']) + 'x') 
        precision_each.append(df['Precision'][representative_iou_index])
        recall_each.append(df['Recall'][representative_iou_index])
        f1_each.append(df['F1'][representative_iou_index])

    f1_all, precision_all, recall_all = calc_f1_scores(df_sum)
    
    f1_rep = f1_all[representative_iou_index]

    df_sum.insert(0, 'IoU_Thresh', thresholds)
    df_sum['Precision'] = precision_all
    df_sum['Recall'] = recall_all
    df_sum['F1'] = f1_all
    
    df_each = pd.DataFrame(datasets, columns=['Dataset'])
    df_each['MagnificationValue'] = magnifications
    df_each['Magnification'] = magnifications
    df_each['Precision'] = precision_each
    df_each['Recall'] = recall_each
    df_each['F1'] = f1_each
    
    return f1_rep, df_sum, df_each, unique_mags

def get_overall_f1():
    try:
        with open('settings.txt') as f:
            settings = json.load(f) 
    except:
        with open('../settings.txt') as f:
            settings = json.load(f) 

    eval_info = {}

    eval_info['gt_ids'] = settings['individual_gt_ids']
    eval_info['representative_iou'] = settings['representative_iou']    
    f1_rep, _, _, _ = aggregate_scores(settings, eval_info)

    return f1_rep


def evaluate_all(save = True):

    try:
        with open('settings.txt') as f:
            settings = json.load(f) 
    except:
        with open('../settings.txt') as f:
            settings = json.load(f) 

    eval_info = {}

    eval_info['gt_ids'] = settings['individual_gt_ids']
    eval_info['representative_iou'] = settings['representative_iou']

    OUT_DIR = settings['output_base_path'] + '/' + settings['evaluation_result_path']

    f1_rep, df_sum, df_each, unique_mags = aggregate_scores(settings, eval_info)
    if(save == True):
        df_sum.to_csv(OUT_DIR + '/all_stats.csv', index=False)
        df_each.to_csv(OUT_DIR + '/each_stats.csv', index=False)
    thresholds = df_sum['IoU_Thresh']
    eval_info['df_each'] = df_each
    eval_info['unique_mags'] = unique_mags
    MAGNIFICATION_THRESH = 20
    f1_rep_16x, df_sum_16x, df_each_16x, _ = aggregate_scores(settings, eval_info, lambda x: x['magnification'] <= MAGNIFICATION_THRESH)
    f1_rep_40x, df_sum_40x, df_each_40x, _ = aggregate_scores(settings, eval_info, lambda x: x['magnification'] > MAGNIFICATION_THRESH)

    eval_info['df'] = []
    eval_info['df'].append(df_sum)
    eval_info['df'].append(df_sum_16x)
    eval_info['df'].append(df_sum_40x)
    eval_info['rep'] = []
    eval_info['rep'].append(f1_rep)
    eval_info['rep'].append(f1_rep_16x)
    eval_info['rep'].append(f1_rep_40x)
    eval_info['thresholds'] = thresholds

    f1_reps = {}
    scores = {}
    for gt_id in eval_info['gt_ids']:
        eval_info[gt_id] = {}
        f1_rep, df_sum, _, _ = aggregate_scores(settings, eval_info, suffix=gt_id)
        eval_info[gt_id]['f1_reps'] = f1_rep
        eval_info[gt_id]['scores'] = df_sum

    return eval_info



def prepare_evaluate_all_notebook(outdir):
    with open('python/evaluate_all.ipynb') as f:
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

    return round(get_overall_f1(), 2)
