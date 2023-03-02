import numpy as np
import tifffile as tiff
from skimage import measure
from skimage.transform import resize
from skimage.morphology import binary_dilation, disk
from skimage.filters import threshold_otsu
from sklearn.decomposition import NMF
from scipy.ndimage import gaussian_filter
from scipy.ndimage.morphology import binary_fill_holes


def run_demixing(masks, avg_prob, avg_image,
                 prob_thresh, area_thresh_min,
                 concavity_thresh, intensity_thresh, activity_thresh):
    """
    Demix cells for all the candidate regions.

    Parameters
    ----------
    masks : 3D numpy.ndarray of boolean
        Sequence of binary masks of firing neurons.
        The shape is (num_time_segments, height, width).
    avg_prob : 2D numpy.ndarray of float
        Average probability map of firing neurons.
        The shape is (height, width).
    avg_image : 2D numpy.ndarray of float
        Average image of the motion/shading-corrected input video.
        The shape is (height, width).

    Refer to compute_masks() for the definitions of other parameters.

    Returns
    -------
    component_list : list of (subset of) skimage.measure.RegionProperties
        Connected components representing the footprints of the demixed cells.
        Only bbox (bounding box) and image (binary footprint) will be returned
    """
    t, h, w = masks.shape
    masks = masks.astype(float)
    X = masks.reshape((t, h * w))

    component_list = []
    any_change = True
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
            if(c.area < area_thresh_min):
                continue
            elif(c.convex_area / c.area > concavity_thresh):
                continue
            elif(c.eccentricity > 0.98):
                continue
            else:
                ymin, xmin, ymax, xmax = c.bbox
                tmp = np.zeros((h, w), dtype=bool)
                tmp[ymin:ymax, xmin:xmax] = c.image
                intensity = np.mean(avg_image, where=tmp)
                prob = np.mean(avg_prob, where=tmp)
                if(intensity < intensity_thresh):
                    continue
                elif(intensity * prob < activity_thresh):
                    continue

                foreground[ymin:ymax, xmin:xmax] |= c.image
                component_list.append((c.bbox, c.image))
                any_change = True

        Y[np.logical_not(binary_dilation(foreground, disk(3)))] = 0
        r1a = np.matmul(W, Y.reshape((1, h * w)))
        if(np.amax(r1a) < 0.8):
            break

        Xm1 = X.astype(float) - r1a
        X = Xm1 > 0.95 # binarize
        masks = X.reshape((t, h, w))

    return component_list


def compute_masks(prob_data, img_data, prob_file, img_file, out_file,
                  prob_thresh, area_thresh_min, area_thresh_max,
                  concavity_thresh, intensity_thresh, activity_thresh,
                  background_sigma, background_edge, background_thresh,
                  mask_dilation, orig_image_shape):
    """
    Compute binary masks representing the footprints of neurons based on their
    firing probability maps while demixing their spatial overlaps if any.

    Parameters
    ----------
    prob_data : 3D numpy.ndarray of float, or None
        Input sequence of probability maps of firing neurons.
        If None, the data will be loaded from prob_file.
    img_data : 3D numpy.ndarray of float, or None
        Input sequence of motion/shading corrected images.
        If None, the data will be loaded from img_file.
    prob_file : string or pathlib.Path
        Input file path of a multi-page tiff containig a sequence of
        probability maps of firing neurons.
        It will not be referenced if prob_data is not None.
    img_file : string or pathlib.Path
        Input file path of a multi-page tiff containing a sequence of
        motion/shading corrected images.
        It will not be referenced if img_data is not None.
    out_file : string, pathlib.Path, or None
        Output tiff file path to which binary masks representing the footprints
        of detected and demixed neurons will be saved.
        If None, no file will be saved.
    prob_thresh : float
        Probability value in [0, 1] above which pixels are considered
        belonging to firing neurons.
    area_thresh_min : int
        Regions having no less area than this will be extracted as neurons.
    area_thresh_max : int
        Regions having no greater area than this will be extracted as neurons.
    concavity_thresh : float
        Regions having no greater concavity than this will be extracted as
        neurons. The concavity of a region is calculated as the area of the
        convex hull of the region divided by the area of the region.
        If a region is convex, its concavity is 1, and it becomes larger as
        the region becomes more concave.
    intensity_thresh : float
        Regions having no less mean intensity than this will be extracted as
        neurons.
    activity_thresh : float
        Regions having no less activity level than this will be extracted as
        neurons. An activity level is calculated as the product of the mean
        intensity and the mean firing probability.
    background_sigma : float
        Gaussian filter size to estimate a background intensity map.
    background_edge : float
        When the background is estimated with Gaussian filtering, pixels near
        the image borders tend to be dark and can produce false foreground
        there. This parameter can be used to mitigate this and specifies how
        far from the image borders this treatment should reach.
    background_thresh : float
        Image regions having larger intensity than the background by this value
        will be extracted as foreground.
    mask_dilation : int
        The computed masks will be dilated by this size. This may be useful
        for accuracy evaluation against manual annotation because humans tend
        to draw ROIs around neurons, resulting in slightly larger masks.
    orig_image_shape : tuple of int, or None
        The original image shape (height, width) before downsampling if any.

    Returns
    -------
    masks : 3D numpy.ndarray of bool
        Binary masks representing the footprints of detected and demixed neurons.

    """
    if(prob_data is None):
        prob_data = tiff.imread(prob_file).astype(float)
    avg_prob = np.mean(prob_data, axis=0)

    if(img_data is None):
        img_data = tiff.imread(img_file).astype(float)
    avg_image = np.mean(img_data, axis=0)
    # subtract background to leave the intensity of foreground cells
    background = gaussian_filter(avg_image, background_sigma, mode='nearest')

    vignette = np.ones(avg_image.shape)
    if(background_edge > 0):
        vignette = gaussian_filter(vignette, background_edge, mode='constant')
    foreground = avg_image * vignette - background > background_thresh
    foreground = binary_dilation(foreground, disk(3))
    foreground = binary_fill_holes(foreground)

    avg_image -= background

    t, h, w = prob_data.shape

    # extract high probabilty regions as spikes
    binary_segm = prob_data > prob_thresh
    binary_segm &= foreground
    # filter out non-cell regions
    masks = np.zeros_like(binary_segm)
    for i, mask in enumerate(binary_segm):
        label_image = measure.label(mask)
        components = measure.regionprops(label_image)
        for c in components:
            if(c.area < area_thresh_min):
                continue
            elif(c.area > area_thresh_max): # to reject large flashing
                continue
            elif(c.eccentricity > 0.98): # to reject response to edges
                continue
            else:
                ymin, xmin, ymax, xmax = c.bbox
                masks[i, ymin:ymax, xmin:xmax] |= c.image
    foreground = np.logical_or.reduce(masks, axis=0)

    label_image = measure.label(foreground)
    components = measure.regionprops(label_image)
    cell_list = []
    for c in components:
        ys, xs, ye, xe = c.bbox
        ls = run_demixing(masks[:, ys:ye, xs:xe] & c.image,
                          avg_prob[ys:ye, xs:xe],
                          avg_image[ys:ye, xs:xe],
                          prob_thresh, area_thresh_min,
                          concavity_thresh, intensity_thresh, activity_thresh)
        for c in ls:
            ymin, xmin, ymax, xmax = c[0]
            cell_list.append((ys + ymin, xs + xmin, ys + ymax, xs + xmax, c[1]))

    if(orig_image_shape is None):
        masks = np.zeros((0, h, w), dtype=bool)
    else:
        masks = np.zeros((0,) + orig_image_shape, dtype=bool)

    for c in cell_list:
        ymin, xmin, ymax, xmax, image = c
        tmp = np.zeros((h, w), dtype=bool)
        tmp[ymin:ymax, xmin:xmax] = image
        if(mask_dilation > 0):
            tmp = binary_dilation(tmp, disk(mask_dilation))
        if(orig_image_shape is not None and tmp.shape is not orig_image_shape):
            tmp = resize(tmp, orig_image_shape, mode='constant')
        masks = np.concatenate((masks, tmp[np.newaxis]), axis=0)

    if(out_file is not None):
        if(len(masks) == 0):
            # add a blank image so the file has at least one page
            masks = np.zeros((1,) + orig_image_shape, dtype=bool)
        tiff.imwrite(out_file, masks.astype('uint8') * 255,
                     photometric='minisblack')

    return masks
