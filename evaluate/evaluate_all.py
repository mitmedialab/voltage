import pathlib
import numpy as np
import pandas as pd

from .f1score import calc_f1_scores



def _aggregate_scores(out_dir, representative_iou, weighted=False):
    """
    Aggregate individual evaluation scores.

    Parameters
    ----------
    out_dir : string
        Path to a directory from which individual evaluation statistics will
        be read, and in which aggregated evaluation will be saved.
    representative_iou : float
        IoU threshold at which representative F-1 score will be computed.
    weighted : boolean, optional
        There are two ways to calculate aggregate F1 scores: one is to
        calculate it from the total numbers of true/false positives/negatives,
        and the other is to average F1 scores from individual datasets.
        The former puts more weight on datasets containing more neurons,
        whereas the latter treats datasets equally. Since the former makes
        datasets with single neurons almost meaningless, the latter is set as
        default (weighted=False).

    Returns
    -------
    f1_rep : float
        Single F1 score representing the overall accuracy of predicted neuron
        masks, corresponding to an IoU threshold of representative_iou.
    df_sum : pandas.DataFrame
        Table summarizing overall evaluation statistics.
    df_each : pandas.DataFrame
        Table summarizing individual evalution statistics corresponding
        to an IoU threshold of representative_iou.

    """
    dataset_ids = []
    precision_each = []
    recall_each = []
    f1_each = []

    first = True
    filenames = sorted(out_dir.glob('*/*.html'))
    for in_file in filenames:
        basename = in_file.stem
        df = pd.read_csv(in_file.with_name(basename + '_counts.csv'))
        if(first):
            df_sum = df.drop(columns=['IoU_Thresh'])
            thresholds = df['IoU_Thresh']
            indices = np.where(thresholds >= representative_iou)
            representative_iou_index = indices[0][0]
            first = False
        else:
            df_sum += df
        
        dataset_ids.append(basename)
        precision_each.append(df['Precision'][representative_iou_index])
        recall_each.append(df['Recall'][representative_iou_index])
        f1_each.append(df['F1'][representative_iou_index])

    if(weighted):
        f1_all, precision_all, recall_all = calc_f1_scores(df_sum)
    else:
        f1_all = df_sum['F1'] / len(filenames)
        precision_all = df_sum['Precision'] / len(filenames)
        recall_all = df_sum['Recall'] / len(filenames)

    f1_rep = f1_all[representative_iou_index]

    df_sum.insert(0, 'IoU_Thresh', thresholds)
    df_sum['Precision'] = precision_all
    df_sum['Recall'] = recall_all
    df_sum['F1'] = f1_all
    df_sum.to_csv(out_dir.joinpath('all_counts.csv'), index=False)
    
    df_each = pd.DataFrame(dataset_ids, columns=['Dataset'])
    df_each['Precision'] = precision_each
    df_each['Recall'] = recall_each
    df_each['F1'] = f1_each
    df_each.to_csv(out_dir.joinpath('individual_scores.csv'), index=False)
    
    return f1_rep, df_sum, df_each


def _aggregate_times(out_dir):
    """
    Aggregate individual speed statistics.

    Parameters
    ----------
    out_dir : string
        Path to a directory from which individual speed statistics will
        be read, and in which aggregated statistics will be saved.

    Returns
    -------
    df_all : pandas.DataFrame
        Table summarizing all speed statistics.

    """
    first = True
    df_all = None
    filenames = sorted(out_dir.glob('*/*.html'))
    for in_file in filenames:
        basename = in_file.stem
        time_file = in_file.with_name(basename + '_times.csv')
        if(time_file.exists()):
            df = pd.read_csv(time_file)
            if(first):
                df_all = df
                first = False
            else:
                df_all = pd.concat([df_all, df], ignore_index=True)

    if(df_all is not None):
        df_all.to_csv(out_dir.joinpath('all_times.csv'), index=False)
        return df_all
    else:
        return 'No runtime was measured.'


def evaluate_all(out_dir, representative_iou):
    """
    Aggregate individual evaluation results and report overall evaluation.
    This assumes individual evaluations have already been performed by
    evaluate_each() and their results are stored in out_dir.

    Parameters
    ----------
    out_dir : string
        Path to a directory from which individual evaluation statistics will
        be read, and in which overall evaluation will be saved.
    representative_iou : float
        IoU threshold at which representative F-1 score will be computed.

    Returns
    -------
    eval_data : dictionary
        Dictionary containing overall evaluation statistics.

    """
    f1_rep, df_sum, df_each = _aggregate_scores(pathlib.Path(out_dir),
                                                representative_iou)

    eval_data = {}
    eval_data['representative f1'] = f1_rep
    
    eval_data['thresholds'] = df_sum['IoU_Thresh']
    eval_data['f1_all'] = df_sum['F1']
    eval_data['precision_all'] = df_sum['Precision']
    eval_data['recall_all'] = df_sum['Recall']

    eval_data['dataset'] = df_each['Dataset']
    eval_data['f1_each'] = df_each['F1']
    eval_data['precision_each'] = df_each['Precision']
    eval_data['recall_each'] = df_each['Recall']

    df_times = _aggregate_times(pathlib.Path(out_dir))
    eval_data['times'] = df_times

    return eval_data
