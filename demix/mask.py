import math
import numpy as np
import tifffile as tiff
import multiprocessing as mp
from skimage import measure
from skimage.transform import resize, rescale
from skimage.morphology import binary_erosion
from skimage.filters import threshold_otsu
from scipy.ndimage import gaussian_filter

from .demix import demix_cells_incrementally


# parameters for extracting candidate regions
PROBABILITY_MAPS_SCALE = 1/4
ACTIVE_REGIONS_SIGMA = 10
ACTIVE_REGIONS_THRESHOLD = 0.001
BACKGROUND_SIGMA = 10

# parameters for filtering candidate regions
AREA_THRESHOLD = 55
ACTIVITY_LEVEL_THRESHOLD_RELATIVE = 1/9
ACTIVITY_LEVEL_THRESHOLD_ABSOLUTE = 0.0001

# parameter for demixing
DEMIX_REGION_SIZE_MAX = 15
MERGE_THRESHOLD = 0.2


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


def filter_regions(components, cell_probability, cell_image):
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

    activity_levels = []

    for c in components:

        if(c.area < AREA_THRESHOLD):
            activity_levels.append(0) # mark as discarded
            continue

        ymin, xmin, ymax, xmax = c.bbox
        crop_probability = cell_probability[ymin:ymax, xmin:xmax]
        crop_image = cell_image[ymin:ymax, xmin:xmax]

        # Define activity level as the product of the mean firing
        # probability and the mean image intensity within the region
        mean_probability = np.mean(crop_probability, where=c.image)
        mean_image = np.mean(crop_image, where=c.image)
        activity_levels.append(mean_probability * mean_image)

    # Discard candidate regions whose activity level is either less than
    # a fraction of the maximum level found in the data or is very small
    if(activity_levels):
        max_activity = max(activity_levels)
    out = []
    for al, c in zip(activity_levels, components):
        if(al > max_activity * ACTIVITY_LEVEL_THRESHOLD_RELATIVE
           and al > ACTIVITY_LEVEL_THRESHOLD_ABSOLUTE):
            out.append(c)

    return out


def filter_masks(masks):
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
            merge[i, j] = IoU > MERGE_THRESHOLD

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


def run_demixing_each(mask, probability_maps, region_id,
                      mode, num_threads, save_images):
    """
    Demix cells in a given region specified by a mask.

    The reason why this function does not take skimage.measure.RegionProperties
    directly is because RegionProperties cannot be pickled, causing "maximum
    recursion depth exceeded" error when called using multiprocessing.

    Parameters
    ----------
    mask : 2D numpy.ndarray of boolean
        Binary mask defining the region.
    probability_maps : 3D numpy.ndarray of float
        Sequence of probability maps of firing neurons within the region.
        The shape is (num_frames,) + mask.shape.
    region_id : integer
        ID of the region.
    mode : string
        Execution mode. Options are 'cpu' (multithreaded C++ on CPU),
        and 'py' (pure Python implementation).
    num_threads : integer
        The number of threads for the multithreaded C++ execution (mode='cpu').
    save_images : boolean
        If True, intermediate images will be saved for debugging.

    Returns
    -------
    demixed : 3D numpy.ndarray of float
        Demixed cells in the region. The shape is (num_cells,) + mask.shape,
        where the number of cells is determined by the demixing algorithm.
    masks : 3D numpy.ndarray of boolean
        Binary masks representing the footprints of the demixed cells.
        The shape is the same as above (demixed) except that the number of
        masks may be smaller due to postprocess filtering.

    """
    # Downsample if the region is large so the shorter side will have the
    # predetermined length (DEMIX_REGION_SIZE_MAX). This is to prevent a large
    # region from getting oversplit just because it has more pixels and tends
    # to have more variability between them. It also reduces the computation.
    size_y, size_x = mask.shape
    if(size_x >= size_y and size_y > DEMIX_REGION_SIZE_MAX):
        size_x = math.floor(size_x / size_y * DEMIX_REGION_SIZE_MAX)
        size_y = DEMIX_REGION_SIZE_MAX
    elif(size_y >= size_x and size_x > DEMIX_REGION_SIZE_MAX):
        size_y = math.floor(size_y / size_x * DEMIX_REGION_SIZE_MAX)
        size_x = DEMIX_REGION_SIZE_MAX

    down = resize(probability_maps, (len(probability_maps), size_y, size_x),
                  mode='constant', anti_aliasing=True)
    demixed = demix_cells_incrementally(down, region_id,
                                        mode, num_threads, save_images)
    demixed = resize(demixed, (len(demixed),) + mask.shape, mode='constant')

    # If we decide that there is only one cell, use the region mask as-is
    if(len(demixed) == 1):
        demixed[0] = mask.astype(float)
        masks = mask[np.newaxis]
    else:
        # threshold demixed cell probabilities to yield binary masks
        masks = np.zeros(demixed.shape, dtype=bool)
        for i, img in enumerate(demixed):
            th = threshold_otsu(img)
            masks[i] = img > th
        masks = filter_masks(masks) # postprocess filtering

    return demixed, masks


def run_demixing(components, probability_maps,
                 mode, num_threads, save_images):
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
    if(num_threads == 0): # serial execution
        results = []
        for i, c in enumerate(components):
            ymin, xmin, ymax, xmax = c.bbox
            crop_prob = np.multiply(c.image,
                                    probability_maps[:, ymin:ymax, xmin:xmax])
            demixed, masks = run_demixing_each(c.image, crop_prob, i,
                                               mode, 1, save_images)
            results.append((demixed, masks))

    elif(len(components) == 0): # to avoid num_processes=0 (div0) below
        results = []

    else: # parallel execution
        # Parallelize over regions as much as possible with multiprocessing
        num_processes = min(len(components), mp.cpu_count())
        # Then allocate threads to each process, up to num_threads
        num_threads = min(num_threads, mp.cpu_count() // num_processes)

        args = []
        for i, c in enumerate(components):
            ymin, xmin, ymax, xmax = c.bbox
            crop_prob = np.multiply(c.image,
                                    probability_maps[:, ymin:ymax, xmin:xmax])
            args.append((c.image, crop_prob, i,
                         mode, num_threads, save_images))

        pool = mp.Pool(num_processes)
        results = pool.starmap(run_demixing_each, args)
        pool.close()

    # Put the results back in the original image shape
    image_shape = probability_maps.shape[1:]
    demixed_all = np.zeros((0,) + image_shape)
    masks_all = np.zeros((0,) + image_shape, dtype=bool)
    for c, result in zip(components, results):
        ymin, xmin, ymax, xmax = c.bbox
        demixed, masks = result
        uncrop = np.zeros((len(demixed),) + image_shape)
        uncrop[:, ymin:ymax, xmin:xmax] = demixed
        demixed_all = np.append(demixed_all, uncrop, axis=0)
        uncrop = np.zeros((len(masks),) + image_shape, dtype=bool)
        uncrop[:, ymin:ymax, xmin:xmax] = masks
        masks_all = np.append(masks_all, uncrop, axis=0)
    return demixed_all, masks_all


def compute_masks(in_file, data_file, out_file,
                  mode='cpu', num_threads=0, save_images=False):
    """
    Compute binary masks representing the footprints of neurons based on their
    firing probability maps while demixing their spatial overlaps if any.

    Parameters
    ----------
    in_file : string
        Input file path of a multi-page tiff containig a sequence of
        probability maps of firing neurons.
    data_file : string
        Input file path of a multi-page tiff containing a motion/shading
        corrected voltage imaging video.
    out_file : string
        Output tiff file path to which binary masks representing the footprints
        of detected and demixed neurons will be saved.
    mode : string, optional
        Execution mode. Options are 'cpu' (multithreaded C++ on CPU, default),
        and 'py' (pure Python implementation). GPU mode is not available
        (and probably will not be in the future) because the workload is
        not massively parallelizable.
    num_threads : integer, optional
        The number of threads for the multithreaded C++ execution (mode='cpu').
        Up to this many threads will be used for each candidate cell region.
        Therefore, up to (num_threads x # regions) threads will run in total
        as long as CPU cores are available. If it is 0 (default), everything
        will run in serial, processing each region on a single thread.
    save_images : boolean, optional
        If True, intermediate images will be saved for debugging.
        Some (optimization progress) images will be saved only when mode='py'.
        The default is False.

    Returns
    -------
    None.

    """

    print('demixing ' + in_file.stem)

    probability_maps = tiff.imread(in_file).astype(float)
    image_shape = probability_maps.shape[1:]
    
    # temporal average of probability maps
    avg_probability = np.mean(probability_maps, axis=0)
    # reduce clutter by downscaling
    down = rescale(avg_probability, PROBABILITY_MAPS_SCALE,
                   mode='constant', anti_aliasing=True)
    # resize back to the original shape
    avg_probability = resize(down, image_shape, mode='constant')

    # extract high probabilty regions compared with their surroundings
    th = gaussian_filter(avg_probability, ACTIVE_REGIONS_SIGMA)
    active_regions = avg_probability - th > ACTIVE_REGIONS_THRESHOLD

    # temporal average of motion/shading-corrected input video
    video = tiff.imread(data_file).astype(float)
    avg_image = np.mean(video, axis=0)
    # extract low spatial frequency content as background
    background = gaussian_filter(avg_image, BACKGROUND_SIGMA)
    # subtract background to leave the intensity of foreground cells
    avg_image -= background

    if(save_images):
        tiff.imwrite('avg_probability.tif', avg_probability.astype('float32'),
                     photometric='minisblack')
        tiff.imwrite('active_regions.tif', active_regions.astype('float32'),
                     photometric='minisblack')
        tiff.imwrite('avg_image.tif', avg_image.astype('float32'),
                     photometric='minisblack')


    # separate active regions into connected components
    label_image = measure.label(binary_erosion(active_regions))
    components = measure.regionprops(label_image)

    if(save_images):
        save_components(components, image_shape, 'masks_initial.tif')

    # discard unlikely regions from the candidates
    components = filter_regions(components, avg_probability, avg_image)

    if(save_images):
        save_components(components, image_shape, 'masks_filtered.tif')

    # demix potentially overlapping cells in each of the regions
    demixed, masks = run_demixing(components, probability_maps,
                                  mode, num_threads, save_images)

    if(save_images):
        if(len(demixed) == 0):
            # add a blank image so the file has at least one page
            demixed = np.zeros((1,) + image_shape)
        tiff.imwrite('demixed.tif', demixed.astype('float32'),
                     photometric='minisblack')

    if(len(masks) == 0):
        # add a blank image so the file has at least one page
        masks = np.zeros((1,) + image_shape, dtype=bool)
    tiff.imwrite(out_file, masks.astype('uint8') * 255,
                 photometric='minisblack')
