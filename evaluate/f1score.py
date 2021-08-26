import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching


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
            if(np.sum(U) > 0):
                IoU[i, j] = np.sum(I) / np.sum(U)
            else:
                IoU[i, j] = 0;

    counts = []
    for th in thresholds:
        matches = np.array(IoU >= th)
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
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros_like(recall), where=(true_pos > 0))
    return f1, precision, recall
