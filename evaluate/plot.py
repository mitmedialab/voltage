import numpy as np
import matplotlib.pyplot as plt


REPRESENTATIVE_IOU = 0.4


def plot_F1(f1, precision, recall, thresholds):
    plt.figure(figsize=(5, 5))
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
    plt.show()


def plot_IoU(IoU):
    plt.figure(figsize=(5, 5))
    ax = plt.subplot()
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
