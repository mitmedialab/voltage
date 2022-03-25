import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import center_of_mass
from skimage.segmentation import find_boundaries
from pathlib import Path


REPRESENTATIVE_IOU = 0.4



def _savefig(plt, filename):
    """
    Save plots in files. This must be called before plt.show().

    Parameters
    ----------
    plt : matplotlib.pyplot
        Plotting interface holding the plot to be saved.
    filename : string or pathlib.Path
        File path to which the plot will be saved, without extension.

    Returns
    -------
    None.

    """
    p = Path(filename)
    plt.savefig(p.with_suffix('.png'), bbox_inches='tight')
    plt.savefig(p.with_suffix('.svg'), bbox_inches='tight')



def _plot_F1_sub(f1, precision, recall, thresholds):
    """
    Plot F1 scores as well as precision and recall values.

    Parameters
    ----------
    See plot_F1_and_IoU().

    Returns
    -------
    None.

    """
    plt.axis('square')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    offset = 0.005 # so points on y=1 will be visible
    plt.plot(thresholds, precision - offset, label='Precision')
    plt.plot(thresholds, recall - offset, label='Recall')
    plt.plot(thresholds, f1 - offset, label='F1 score')
    plt.legend(loc='lower left')
    plt.ylabel('Score')
    plt.xlabel('IoU Threshold')
    plt.vlines(REPRESENTATIVE_IOU, 0, 1, colors='gray', linestyles='dashed') 
    indices = np.where(thresholds >= REPRESENTATIVE_IOU)
    plt.title('F1 = %.2f at IoU = %.1f' % (f1[indices[0][0]], REPRESENTATIVE_IOU))


def _plot_IoU_sub(IoU):
    """
    Visualize an IoU matrix as a heat map.

    Parameters
    ----------
    See plot_F1_and_IoU().

    Returns
    -------
    None.

    """
    ax = plt.gca()
    ax.set_xticks([x - 0.5 for x in range(IoU.shape[1])], minor=True)
    ax.set_yticks([y - 0.5 for y in range(IoU.shape[0])], minor=True)
    ax.tick_params(axis='both', which='both', length=0)
    plt.grid(which='minor')
    plt.xticks(list(range(IoU.shape[1])))
    plt.yticks(list(range(IoU.shape[0])))
    if(IoU.shape[0] == 0): # in case there is no predicted cell
        IoU = np.zeros((1, IoU.shape[1]))
    plt.imshow(IoU, aspect='equal', vmin=0, vmax=1, cmap='inferno')
    plt.colorbar()
    plt.xlabel('Ground Truth')
    plt.ylabel('Prediction')
    plt.title('IoU Matrix')


def plot_F1(f1, precision, recall, thresholds):
    """
    Plot F1 scores as well as precision and recall values.

    Parameters
    ----------
    See plot_F1_and_IoU().

    Returns
    -------
    None.

    """
    plt.figure(figsize=(5, 5))
    _plot_F1_sub(f1, precision, recall, thresholds)
    plt.show()


def plot_F1_and_IoU(f1, precision, recall, thresholds, IoU, filename=None):
    """
    Plot F1 scores as well as precision and recall values,
    and visualize an IoU matrix as a heat map.

    Parameters
    ----------
    f1 : list of float
        F1 scores for varying IoU thresholds.
    precision : list of float
        Precision values for varying IoU thresholds.
    recall : list of float
        Recall values for varying IoU thresholds.
    thresholds : list of float
        IoU thresholds.
    IoU : 2D numpy.ndarray of float
        Matrix whose elements represent IoU values between predicted masks
        and ground truth masks.
    filename : string or pathlib.Path, optional
        File path to which the plot will be saved. The default is None,
        in which case the plot will not be saved.

    Returns
    -------
    None.

    """
    plt.figure(figsize=(11, 5))
    plt.subplot(1, 2, 1)
    _plot_F1_sub(f1, precision, recall, thresholds)
    plt.subplot(1, 2, 2)
    _plot_IoU_sub(IoU)
    if(filename is not None):
        _savefig(plt, filename)
    plt.show()



def _overlay_masks(masks, color):
    """
    Overlay masks on an image, and label each mask with its ID number.
    The underlying image is assumed to be already drawn, and the contours
    of the masks will be drawn on top of it.

    Parameters
    ----------
    masks : 3D numpy.ndarray of boolean
        Mask images.
    color : 3-tuple of float
        RGB color in [0, 1] for drawing mask contours

    Returns
    -------
    None.

    """
    contour_image = np.zeros(masks.shape[1:], dtype=bool)
    for mask in masks:
        contour = find_boundaries(mask, mode='outer')
        contour_image = np.logical_or(contour_image, contour)
    overlay = np.zeros(contour_image.shape + (4,)) # RGBA
    overlay[:, :, 0] = color[0]
    overlay[:, :, 1] = color[1]
    overlay[:, :, 2] = color[2]
    overlay[:, :, 3] = contour_image
    plt.imshow(overlay, interpolation='bilinear')
    for i, mask in enumerate(masks):
        p = center_of_mass(mask)
        plt.text(p[1], p[0], str(i), color=color)


def _plot_masks_sub(image, eval_masks, gt_masks):
    """
    Plot (show) mask images overlaid on a given image. Either of eval_masks
    or gt_masks can be None, in which case only the other masks will be shown.

    Parameters
    ----------
    See plot_masks().

    Returns
    -------
    None.

    """
    plt.axis('off')
    plt.imshow(image, interpolation='bilinear', cmap='gray')
    title = ''
    if(eval_masks is not None):
        _overlay_masks(eval_masks, (1, 1, 0))
        title += 'Prediction (yellow)'
        if(gt_masks is not None):
            title += ' vs '
    if(gt_masks is not None):
        _overlay_masks(gt_masks, (0, 1, 1))
        title += 'GT (cyan)'
    plt.title(title)


def plot_masks(image, eval_masks, gt_masks, filename=None):
    """
    Plot (show) three mask images: (1) predicted masks to be evaluated and
    the ground truth masks, (2) predicted masks only, and (3) GT masks only.
    All of them will be overlaid on a given image.

    Parameters
    ----------
    image : 2D numpy.ndarray of float
        Image on which masks are overlaid.
    eval_masks : 3D numpy.ndarray of boolean
        Masks to be evaluated. The shape should be (# masks,) + image.shape.
    gt_masks : 3D numpy.ndarray of boolean
        Ground truth masks. The shape should be (# masks,) + image.shape.
    filename : string or pathlib.Path, optional
        File path to which the plot will be saved. The default is None,
        in which case the plot will not be saved.

    Returns
    -------
    None.

    """
    plt.figure(figsize=(17, 5))
    plt.subplot(1, 3, 1)
    _plot_masks_sub(image, eval_masks, gt_masks)
    plt.subplot(1, 3, 2)
    _plot_masks_sub(image, eval_masks, None)
    plt.subplot(1, 3, 3)
    _plot_masks_sub(image, None, gt_masks)
    if(filename is not None):
        _savefig(plt, filename)
    plt.show()

    

def plot_per_dataset_scores(keys, scores, label, color):
    """
    Plot a bar chart showing scores for individual data sets.

    Parameters
    ----------
    keys : list of string
        Names of data sets.
    scores : list of float
        Score values.
    label : string
        Name of the score type. Used to label the chart.
    color : color (any color specification such as string, RGB, etc.)
        Color of the bars.

    Returns
    -------
    None.

    """
    plt.figure(figsize=(17, 3))
    plt.ylim(0, 1)
    plt.grid(axis='y')
    plt.bar(list(range(len(scores))), scores, color=color) 
    plt.xticks(list(range(len(scores))), keys, rotation='vertical')
    plt.ylabel(label)
    plt.xlabel('Dataset')
    plt.title('Per-dataset ' + label + ' at IoU = %.1f' % REPRESENTATIVE_IOU)
    plt.show()
