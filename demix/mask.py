import math
import numpy as np
import tifffile as tiff
import multiprocessing as mp
from skimage import measure
from skimage.transform import resize, rescale
from skimage.morphology import binary_erosion, binary_dilation, disk
from skimage.filters import threshold_otsu
from sklearn.decomposition import NMF
from scipy.ndimage import gaussian_filter

from .demix import demix_cells_incrementally


def save_components(components, image_shape, filename):
    """
    Convert connected components into a multi-page binary image and save it.

    Parameters
    ----------
    components : list of skimage.measure.RegionProperties
        Connected components.
    image_shape : tuple of integer
        The shape (height, width) of the image where the connected components
        will be embedded.
    filename : string
        Name of the image file to be saved.

    Returns
    -------
    None.

    """
    num_comp = len(components)
    masks = np.zeros((num_comp,) + image_shape, dtype=bool)
    for i, c in enumerate(components):
        ymin, xmin, ymax, xmax = c.bbox
        masks[i, ymin:ymax, xmin:xmax] = c.image

    if(num_comp == 0):
        # add a blank image so the file has at least one page
        masks = np.zeros((1,) + image_shape, dtype=bool)

    tiff.imwrite(filename, masks.astype('uint8') * 255,
                 photometric='minisblack')


def filter_regions(components, cell_probability, cell_image, **kwargs):
    """
    Filter candidate cell regions by discarding unlikely ones.

    Parameters
    ----------
    components : list of skimage.measure.RegionProperties
        Connected components representing candidate cell regions.
    cell_probability : 2D numpy.ndarray of float
        Probability map of cells (probability of being firing neurons).
    cell_image : 2D numpy.ndarray of float
        Image of cells.

    Returns
    -------
    out : list of skimage.measure.RegionProperties
        Filtered cell regions (a subset of the input components)

    """
    kwargs.setdefault('AREA_THRESHOLD', 0)
    kwargs.setdefault('ACTIVITY_LEVEL_THRESHOLD_RELATIVE', 0)
    kwargs.setdefault('ACTIVITY_LEVEL_THRESHOLD_ABSOLUTE', 0)

    activity_levels = []

    for c in components:

        if(c.area < kwargs['AREA_THRESHOLD']):
            activity_levels.append(0) # mark as discarded
            continue

        ymin, xmin, ymax, xmax = c.bbox
        crop_probability = cell_probability[ymin:ymax, xmin:xmax]
        crop_image = cell_image[ymin:ymax, xmin:xmax]

        # Define activity level as the product of the mean firing
        # probability and the mean image intensity within the region
        mean_probability = np.mean(crop_probability, where=c.image)
        mean_image = np.mean(crop_image, where=c.image)
        if(mean_image < 0.0001):
            activity_levels.append(0)
        else:
            activity_levels.append(mean_probability * mean_image)

    # Discard candidate regions whose activity level is either less than
    # a fraction of the maximum level found in the data or is very small
    if(activity_levels):
        max_activity = max(activity_levels)
    out = []
    for al, c in zip(activity_levels, components):
        if(al > max_activity * kwargs['ACTIVITY_LEVEL_THRESHOLD_RELATIVE']
           and al > kwargs['ACTIVITY_LEVEL_THRESHOLD_ABSOLUTE']):
            out.append(c)

    return out


def filter_masks(masks, **kwargs):
    """
    Filter masks by merging ones having large overlaps. Because demixing does
    not explicitly look at how much demixed spatial probabilities overlap, it
    can yield masks that have large overlaps. They are merged back here.

    Parameters
    ----------
    masks : 3D numpy.ndarray of boolean
        Binary masks representing the footprints of neurons.
        The shape is (num_cells, height, width).

    Returns
    -------
    merged_masks : 3D numpy.ndarray of boolean
        Filtered binary masks that may have merged some of the input masks.

    """
    kwargs.setdefault('MERGE_THRESHOLD', 0.2)

    # Determine whether a pair of masks should be merged
    # based on the degree of overlaps between them
    num_masks = len(masks)
    merge = np.zeros((num_masks, num_masks), dtype=bool)
    for i in range(num_masks):
        area_i = np.count_nonzero(masks[i])
        for j in range(i+1, num_masks):
            area_j = np.count_nonzero(masks[j])
            intersection = np.logical_and(masks[i], masks[j])
            area_c = np.count_nonzero(intersection)
            area_u = area_i + area_j - area_c
            IoU = area_c / area_u
            merge[i, j] = IoU > kwargs['MERGE_THRESHOLD']

    # Construct a mapping table indicating whether masks should be merged:
    # mapping[j] = i means j-th mask should be merged with i-th mask
    mapping = [i for i in range(num_masks)] # each goes to itself (no merger)
    for i in range(num_masks):
        for j in range(i+1, num_masks):
            if(merge[i, j]):
                mapping[j] = mapping[i] # j goes to the same mask as i goes to

    # Merge masks using dictionary as mapping[] values are not contiguous
    mask_dict = {}
    for i in range(num_masks):
        j = mapping[i]
        if j in mask_dict:
            mask_dict[j] = np.logical_or(mask_dict[j], masks[i]) # merge
        else:
            mask_dict[j] = masks[i] # add new entry

    # Convert the dictionary into a numpy array
    merged_masks = np.zeros((0,) + masks.shape[1:])
    for mask in mask_dict.values():
        merged_masks = np.append(merged_masks, mask[np.newaxis], axis=0)
    return merged_masks




def run_demixing(masks, avg_prob, avg_image, prob_thresh, area_thresh):
    """
    Demix cells for all the candidate regions.

    Parameters
    ----------
    components : list of skimage.measure.RegionProperties
        Connected components representing candidate cell regions.
    probability_maps : 3D numpy.ndarray of float
        Sequence of probability maps of firing neurons.
        The shape is (num_frames, height, width).
    mode : string
        Execution mode. Options are 'cpu' (multithreaded C++ on CPU),
        and 'py' (pure Python implementation).
    num_threads : integer
        The number of threads for the multithreaded C++ execution (mode='cpu').
        Up to this many threads will be used for each candidate region.
        Therefore, up to (num_threads x # regions) threads will run in total
        as long as CPU cores are available. If num_threads=0, everything will
        run in serial, processing each region on a single thread.
    save_images : boolean
        If True, intermediate images will be saved for debugging.

    Returns
    -------
    demixed_all : 3D numpy.ndarray of float
        Demixed cells. The shape is (num_cells, height, width),
        where the number of cells is determined by the demixing algorithm.
    masks_all : 3D numpy.ndarray of boolean
        Binary masks representing the footprints of the demixed cells.
        The shape is the same as above (demixed_all) except that the number of
        masks may be smaller due to postprocess filtering.

    """
    t, h, w = masks.shape
    #masks = masks.astype(float)
    X = masks.reshape((t, h * w))

    component_list = []
    any_change = True
    ii = 0
    while(any_change):
        any_change = False

        model = NMF(n_components=1, init='random', random_state=0)
        W = model.fit_transform(X)
        H = model.components_
        Y = H.reshape((h, w))

        th = threshold_otsu(Y)
        binary_image = Y > th
        label_image = measure.label(binary_image)
        components = measure.regionprops(label_image)
        foreground = np.zeros_like(binary_image)
        for c in components:
            if(c.area < area_thresh):
                continue
            elif(c.eccentricity > 0.98):
                continue
            else:
                ymin, xmin, ymax, xmax = c.bbox
                tmp = np.zeros((h, w), dtype=bool)
                tmp[ymin:ymax, xmin:xmax] = c.image
                intensity = np.mean(avg_image, where=tmp)
                prob = np.mean(avg_prob, where=tmp)
                if(intensity < 0.0001):
                    continue
                elif(intensity * prob < 0.0001):
                    continue

                foreground[ymin:ymax, xmin:xmax] = np.logical_or(foreground[ymin:ymax, xmin:xmax], c.image)
                component_list.append(c)
                any_change = True

        Y[np.logical_not(binary_dilation(foreground, selem=disk(3)))] = 0
        r1a = np.matmul(W, Y.reshape((1, h * w)))
        if(np.amax(r1a) < 0.8):
            break

        Xm1 = X.astype(float) - r1a
        print('r1a in [%f, %f]' % (np.amin(r1a), np.amax(r1a)))
        print('Y in [%f, %f]' % (np.amin(Y), np.amax(Y)))
        print('Xm1 in [%f, %f]' % (np.amin(Xm1), np.amax(Xm1)))
        X = Xm1 > 0.95#prob_thresh
        #X = np.maximum(Xm1, 0)
        masks = X.reshape((t, h, w))
        #any_change = np.sum(X)
        ii += 1

    return component_list


def compute_masks(prob_file, img_file, out_file,
                  prob_thresh, area_thresh, background_sigma,
                  save_images=False):
    """
    Compute binary masks representing the footprints of neurons based on their
    firing probability maps while demixing their spatial overlaps if any.

    Parameters
    ----------
    prob_file : string
        Input file path of a multi-page tiff containig a sequence of
        probability maps of firing neurons.
    img_file : string
        Input file path of a multi-page tiff containing a sequence of
        motion/shading corrected images.
    out_file : string
        Output tiff file path to which binary masks representing the footprints
        of detected and demixed neurons will be saved.
    prob_thresh : float
        Probability value in [0, 1] above which pixels are considered
        belonging to firing neurons.
    area_thresh : int
        Regions having no less area than this will be extracted as neurons.
    background_sigma : float
        Gaussian filter size to estimate a background intensity map.
    save_images : boolean, optional
        If True, intermediate images will be saved for debugging.
        The default is False.

    Returns
    -------
    None.

    """
    prob_data = tiff.imread(prob_file).astype(float)
    avg_prob = np.mean(prob_data, axis=0)
    
    img_data = tiff.imread(img_file).astype(float)
    avg_image = np.mean(img_data, axis=0)
    # subtract background to leave the intensity of foreground cells
    background = gaussian_filter(avg_image, background_sigma)
    avg_image -= background

    t, h, w = prob_data.shape

    # extract high probabilty regions as spikes
    binary_segm = prob_data > prob_thresh
    # filter out non-cell regions
    masks = np.zeros_like(binary_segm)
    for i, mask in enumerate(binary_segm):
        label_image = measure.label(mask)
        components = measure.regionprops(label_image)
        for c in components:
            if(c.area < area_thresh):
                continue
            elif(c.eccentricity > 0.98):
                continue
            else:
                ymin, xmin, ymax, xmax = c.bbox
                masks[i, ymin:ymax, xmin:xmax] = np.logical_or(masks[i, ymin:ymax, xmin:xmax], c.image)
    foreground = np.logical_or.reduce(masks, axis=0)

    label_image = measure.label(foreground)
    components = measure.regionprops(label_image)
    cell_list = []
    for c in components:
        ys, xs, ye, xe = c.bbox
        ls = run_demixing(masks[:, ys:ye, xs:xe],
                          avg_prob[ys:ye, xs:xe],
                          avg_image[ys:ye, xs:xe],
                          prob_thresh, area_thresh)
        for c in ls:
            ymin, xmin, ymax, xmax = c.bbox
            cell_list.append((ys + ymin, xs + xmin, ys + ymax, xs + xmax, c.image))


    masks = np.zeros((0, h, w), dtype=bool)
    for c in cell_list:
        ymin, xmin, ymax, xmax, image = c
        tmp = np.zeros((h, w), dtype=bool)
        tmp[ymin:ymax, xmin:xmax] = image
        tmp = binary_dilation(tmp, selem=disk(2))
        masks = np.concatenate((masks, tmp[np.newaxis]), axis=0)

    if(len(masks) == 0):
        # add a blank image so the file has at least one page
        masks = np.zeros((1, h, w), dtype=bool)
    tiff.imwrite(out_file, masks.astype('uint8') * 255,
                 photometric='minisblack')
