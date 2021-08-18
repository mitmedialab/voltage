import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import center_of_mass
from skimage.segmentation import find_boundaries


REPRESENTATIVE_IOU = 0.4


    
def _plot_F1_sub(f1, precision, recall, thresholds):
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
    ax = plt.gca()
    ax.set_xticks([x - 0.5 for x in range(IoU.shape[1])], minor=True)
    ax.set_yticks([y - 0.5 for y in range(IoU.shape[0])], minor=True)
    ax.tick_params(axis='both', which='both', length=0)
    plt.grid(which='minor')
    plt.xticks(list(range(IoU.shape[1])))
    plt.yticks(list(range(IoU.shape[0])))
    plt.imshow(IoU, aspect='equal', vmin=0, vmax=1, cmap='inferno')
    plt.colorbar()
    plt.xlabel('Ground Truth')
    plt.ylabel('Prediction')
    plt.title('IoU Matrix')


def plot_F1(f1, precision, recall, thresholds):
    plt.figure(figsize=(5, 5))
    _plot_F1_sub(f1, precision, recall, thresholds)
    plt.show()


def plot_F1_and_IoU(f1, precision, recall, thresholds, IoU):
    plt.figure(figsize=(11, 5))
    plt.subplot(1, 2, 1)
    _plot_F1_sub(f1, precision, recall, thresholds)
    plt.subplot(1, 2, 2)
    _plot_IoU_sub(IoU)
    plt.show()



def _overlay_masks(masks, color):
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


def plot_masks(image, eval_masks, gt_masks):
    plt.figure(figsize=(17, 5))
    plt.subplot(1, 3, 1)
    _plot_masks_sub(image, eval_masks, gt_masks)
    plt.subplot(1, 3, 2)
    _plot_masks_sub(image, eval_masks, None)
    plt.subplot(1, 3, 3)
    _plot_masks_sub(image, None, gt_masks)
    plt.show()

    

def plot_per_dataset_scores(keys, scores, label, color):
    #mags = df['Magnification']
    plt.figure(figsize=(17, 3))
    plt.ylim(0, 1)
    plt.grid(axis='y')
    plt.bar(list(range(len(scores))), scores, color=color) 
    plt.xticks(list(range(len(scores))), keys, rotation='vertical')
    #for mag, ticklabel in zip(mags, plt.gca().get_xticklabels()):
    #    if(mag >= 40):
    #        ticklabel.set_color('green')
    #    elif(mag >= 20):
    #        ticklabel.set_color('blue')
    plt.ylabel(label)
    #plt.xlabel('Dataset  (black 16x, blue 20x, green 40x)')
    plt.xlabel('Dataset')
    plt.title('Per-dataset ' + label + ' at IoU = %.1f' % REPRESENTATIVE_IOU)
    plt.show()
