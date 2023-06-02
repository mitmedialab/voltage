import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import center_of_mass
from skimage.segmentation import find_boundaries



def _savefig(filename):
    """
    Save plots in files. This must be called before plt.show().

    Parameters
    ----------
    filename : string or pathlib.Path
        File path to which the plot will be saved, without extension.

    Returns
    -------
    None.

    """
    filename = str(filename)
    plt.savefig(filename + '.png', bbox_inches='tight')
    plt.savefig(filename + '.svg', bbox_inches='tight')



def _plot_F1_sub(f1, precision, recall, thresholds, representative_iou):
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
    plt.vlines(representative_iou, 0, 1, colors='gray', linestyles='dashed')
    indices = np.where(thresholds >= representative_iou)
    plt.title('F1 = %.2f at IoU = %.1f' % (f1[indices[0][0]], representative_iou))


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


def plot_F1(f1, precision, recall, thresholds, representative_iou):
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
    _plot_F1_sub(f1, precision, recall, thresholds, representative_iou)
    plt.show()


def plot_F1_and_IoU(f1, precision, recall, thresholds,
                    IoU, representative_iou, filename=None):
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
    representative_iou : float
        IoU threshold at which representative F-1 score will be computed.
    filename : string or pathlib.Path, optional
        File path to which the plot will be saved. The default is None,
        in which case the plot will not be saved.

    Returns
    -------
    None.

    """
    plt.figure(figsize=(11, 5))
    plt.subplot(1, 2, 1)
    _plot_F1_sub(f1, precision, recall, thresholds, representative_iou)
    plt.subplot(1, 2, 2)
    _plot_IoU_sub(IoU)
    if(filename is not None):
        _savefig(filename)
    plt.show()



def _overlay_masks(masks, color, base=0):
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
    base : integer, optional
        Base number for IDs. They will be offset by this number. Default is 0.

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
        plt.text(p[1], p[0], str(base + i), color=color)


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
        _savefig(filename)
    plt.show()

    

def plot_per_dataset_scores(keys, scores, label, color, representative_iou):
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
    representative_iou : float
        IoU threshold at which representative F-1 score has been computed.
        It is used only for the purpose of labeling the plot here.

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
    plt.title('Per-dataset ' + label + ' at IoU = %.1f' % representative_iou)
    plt.show()


def plot_voltage_traces(image, masks, spike_file, filename=None):
    """
    Plot voltage traces and their spike times.

    Parameters
    ----------
    image : 2D numpy.ndarray of float
        Reference image.
    masks : 3D numpy.ndarray of boolean
        Masks to show which area of the reference image the trace comes from.
        The shape should be (# masks,) + image.shape.
    spike_file : string or pathlib.Path
        HDF5 file containing extracted voltage traces and their spike times.
    filename : string or pathlib.Path, optional
        File path to which the plot will be saved. The default is None,
        in which case the plot will not be saved.

    Returns
    -------
    None.

    """
    YMIN = -5
    YMAX = 10
    with h5py.File(spike_file, 'r') as f:
        for i, mask in enumerate(masks):
            _, (a0, a1) = plt.subplots(1, 2, width_ratios=(1, 5), figsize=(17, 3))
            a0.axis('off')
            a0.imshow(image, interpolation='bilinear', cmap='gray')
            plt.axes(a0)
            _overlay_masks(mask[np.newaxis], (1, 1, 0), i)

            grp = f['neuron%d' % i]
            voltage = grp['voltage'][:]
            spikes = grp['spikes'][:]
            spike_marks = np.ones_like(spikes) * YMAX * 0.9
            a1.set_ylim(YMIN, YMAX)
            a1.plot(voltage)
            a1.plot(spikes, spike_marks, '|')
            plt.show()

        # Plot all the voltage traces in a single file, without reference
        # image/mask, and with much larger width
        if(filename is not None):
            num_neurons = len(masks)
            plt.figure(figsize=(50, 2 * num_neurons)) # larger width
            for i in range(num_neurons):
                grp = f['neuron%d' % i]
                voltage = grp['voltage'][:]
                spikes = grp['spikes'][:]
                spike_marks = np.ones_like(spikes) * YMAX * 0.9
                plt.subplot(num_neurons, 1, i+1)
                plt.axis('off')
                plt.ylim(YMIN, YMAX)
                plt.plot(voltage, linewidth=0.5)
                plt.plot(spikes, spike_marks, '|', markeredgewidth=0.5)

            _savefig(filename)
            plt.close() # do not plot on notebook
