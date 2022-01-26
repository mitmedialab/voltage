import math
import numpy as np
import tifffile as tiff
from skimage import measure
from skimage.transform import resize, rescale
from skimage.morphology import binary_erosion
from skimage.filters import threshold_otsu
from scipy.ndimage import gaussian_filter

from .demix import demix_cells_incrementally


AREA_THRESHOLD = 55


def compute_masks(in_file, data_file, out_file, save_images=False):

    print('demixing ' + in_file.stem)

    probability_maps = tiff.imread(in_file).astype(float)
    # remove the last frame, which tends to be errorneous
    # probably because the last segment can be shorter than specified length
    # ToDo: this should be dealt with at the preprocessing step

    # average probability maps (at the downsampled size to reduce clutter)
    down = rescale(probability_maps, (1, 0.25, 0.25), anti_aliasing=True)
    avg_prob = np.mean(down, axis=0)
    avg_prob = resize(avg_prob, probability_maps.shape[1:])

    # extract high probabilty regions compared with their surroundings
    th = gaussian_filter(avg_prob, 10)
    active_areas = avg_prob - th > 0.001


    # motion/shading corrected input data
    data = tiff.imread(data_file).astype(float)
    avg_data = np.mean(data, axis=0)
    background = gaussian_filter(avg_data, 10)
    data -= background # subtract background intensity


    # separate active areas within the foreground into connected components
    #active_foreground = np.logical_and(foreground_mask, active_areas)
    #label_image = measure.label(active_foreground)
    label_image = measure.label(binary_erosion(active_areas))
    components = measure.regionprops(label_image)
    if(save_images):
        tiff.imwrite('avg_prob.tif', avg_prob.astype('float32'),
                     photometric='minisblack')
        tiff.imwrite('active_areas.tif', active_areas,
                     photometric='minisblack')
        tiff.imwrite('avg_data.tif', avg_data.astype('float32'),
                     photometric='minisblack')


    # Process each component
    h, w = active_areas.shape
    masks = np.zeros((0, h, w), dtype=bool)
    margin = 0
    
    activity_levels = []
    
    for c in components:
        component_id = c.label
        
        if(c.area < AREA_THRESHOLD):
            continue
        
        ymin, xmin, ymax, xmax = c.bbox
        # Crop a subimage (bounding box plus some margin to
        # include background pixels) from the probability maps
        ymin = max(ymin - margin, 0)
        xmin = max(xmin - margin, 0)
        ymax = min(ymax + margin, h)
        xmax = min(xmax + margin, w)
        crop_prob = probability_maps[:, ymin:ymax, xmin:xmax]
        
        # Execlude other non-overlapping components that might
        # be within the bounding box
        label = label_image[ymin:ymax, xmin:xmax]
        self_or_background = np.logical_or(label == component_id, label == 0)
        crop_prob = np.multiply(crop_prob, self_or_background)
        # this should be excluded from the equation rather than setting it to zero

        crop_data = data[:, ymin:ymax, xmin:xmax]
            
        # Here we could demix cells in the subimage
        # but for now use the connected component as-is
        mask = c.image
        
        # Waveform from the (motion/shading corrected & background-subtracted)
        # input video by averaging the pixel values within the mask 
        wave_data = np.mean(crop_data, axis=(1, 2), where=mask)
        # Wavefrom from the probability maps
        wave_prob = np.mean(crop_prob, axis=(1, 2), where=mask)
        # Define activity level as the product of the mean intensity
        # and the mean firing probability
        activity_levels.append(np.mean(wave_data) * np.mean(wave_prob))

        # Put the resultant masks back in the original image shape
        uncrop = np.zeros((h, w))
        uncrop[ymin:ymax, xmin:xmax] = mask
        
        masks = np.concatenate((masks, uncrop[np.newaxis]), axis=0)
        
        if(save_images):
            tiff.imwrite('comp%2.2d.tif' % component_id,
                         crop_prob.astype('float32'), photometric='minisblack')
            tiff.imwrite('mask%2.2d.tif' % component_id,
                         mask.astype('float32'), photometric='minisblack')
    
    # Remove candidate cells whose activity level is either less than
    # a fraction of the maximum level found in the data or is very small
    tmp = np.zeros((0, h, w), dtype=bool)
    if(activity_levels):
        max_activity = max(activity_levels)
    for i, al in enumerate(activity_levels):
        if(al > max_activity / 9 and al > 0.0001):
            tmp = np.concatenate((tmp, masks[np.newaxis, i]), axis=0)
    masks = tmp
    
    if(save_images): # save intermediate masks before demixing
        # if no mask, add a blank mask so the image will have at least one page
        if(masks.shape[0] == 0):
            masks = np.zeros((1, h, w), dtype=bool)
        tiff.imwrite('masks.tif', masks.astype('uint8') * 255,
                     photometric='minisblack')

    
    tmp = np.zeros((0, h, w))
    
    for mask in masks:

        label_image = measure.label(mask)
        components = measure.regionprops(label_image)
        if(len(components) == 0):
            continue
        c = components[0]
        ymin, xmin, ymax, xmax = c.bbox
        crop_prob = np.multiply(c.image, probability_maps[:, ymin:ymax, xmin:xmax])
        size_max = 15
        size_x = xmax - xmin
        size_y = ymax - ymin
        if(size_x >= size_y and size_y > size_max):
            size_x = math.floor(size_x / size_y * size_max)
            size_y = size_max
        elif(size_y >= size_x and size_x > size_max):
            size_y = math.floor(size_y / size_x * size_max)
            size_x = size_max
        down = resize(crop_prob, (len(crop_prob), size_y, size_x), mode='constant', anti_aliasing=True)
        demixed = demix_cells_incrementally(down, 'cpu', False, 1)
        demixed = resize(demixed, (len(demixed),) + crop_prob.shape[1:], mode='constant')
        if(len(demixed) == 1):
            demixed[0] = c.image

        # Put the resultant masks back in the original image shape
        uncrop = np.zeros((len(demixed), h, w))
        uncrop[:, ymin:ymax, xmin:xmax] = demixed
        
        tmp = np.concatenate((tmp, uncrop), axis=0)
    
    
    if(save_images): # save demixing result
        # if no mask, add a blank mask so the image will have at least one page
        if(tmp.shape[0] == 0):
            tmp = np.zeros((1, h, w))
        tiff.imwrite('demixed.tif', tmp.astype('float32'), photometric='minisblack')

    out = np.zeros((0, h, w), dtype=bool)
    for im in tmp:
        th = threshold_otsu(im)
        bw = im > th
        #if(np.count_nonzero(bw) >= AREA_THRESHOLD):
        out = np.concatenate((out, bw[np.newaxis]), axis=0)
    if(out.shape[0] == 0):
        out = np.zeros((1, h, w), dtype=bool)

    tiff.imwrite(out_file, out.astype('uint8') * 255, photometric='minisblack')
