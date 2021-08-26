import math
import numpy as np
import tifffile as tiff
from skimage import measure
import skimage.filters as filters
from skimage.transform import resize, rescale


MAX_NUM_OVERLAPPING_CELLS = 6
ERROR_THRESH = 1e-3
MAX_ITER = 10000
UPDATE_STEP = 1.0e-2
ITER_THRESH = 1.0e-3


def compute_products_of_one_minus_c_times_z(num_cells, c, z):
    num_frames = c.shape[1]
    num_pixels = z.shape[1]

    one_minus_c_times_z = np.ones((num_cells, num_frames, num_pixels))
    for i in range(num_cells):
        one_minus_c_times_z[i] = 1 - np.outer(c[i], z[i])

    out = []
    
    # product of all i
    prod_all = np.ones((num_frames, num_pixels))
    for i in range(num_cells):
        prod_all = np.multiply(prod_all, one_minus_c_times_z[i])
    out.append(prod_all)
    
    for j in range(num_cells):
        # product of all i but one j
        prod_but_one = np.ones((num_frames, num_pixels))
        for i in range(num_cells):
            if(i == j):
                continue
            prod_but_one = np.multiply(prod_but_one, one_minus_c_times_z[i])
        out.append(prod_but_one)
    
    return out


def compute_derivatives_c(num_cells, y, c, z):
    prods = compute_products_of_one_minus_c_times_z(num_cells, c, z)
    first_term = y - 1 + prods[0]
    dc = np.zeros(c.shape)
    for j in range(num_cells):
        z_j_as_row = z[np.newaxis, j, :]
        second_term = np.multiply(prods[j+1], z_j_as_row)
        dc[j] = -np.sum(np.multiply(first_term, second_term), axis=1)
    return dc


def compute_derivatives_z(num_cells, y, c, z):
    prods = compute_products_of_one_minus_c_times_z(num_cells, c, z)
    first_term = y - 1 + prods[0]    
    dz = np.zeros(z.shape)
    for j in range(num_cells):
        c_j_as_col = c[j, :, np.newaxis]
        second_term = np.multiply(prods[j+1], c_j_as_col)
        dz[j] = -np.sum(np.multiply(first_term, second_term), axis=0)
    return dz


def demix_cells_py(probability_maps, num_cells, z_init,
                   max_iter, update_step, iter_thresh,
                   save_images):
    num_frames, h, w = probability_maps.shape
    y = np.reshape(probability_maps, (num_frames, h * w))
    c = np.random.rand(num_cells, num_frames)
    z = z_init

    # Alternate gradient descent, iterate until the masks no longer change
    # Important to clip values at [0, 1] as they represent probabilities
    # Diff should be taken after clipping, and should not be computed simply
    # as the magnitude of the derivatives (which could prevent convergence)
    num_iter = 0
    update_norm = iter_thresh + 1
    progress_img = np.zeros((max_iter, h * 2, w * num_cells))
    while(num_iter < max_iter and update_norm > iter_thresh):
        dc = compute_derivatives_c(num_cells, y, c, z)
        c_new = np.clip(c - update_step * dc, 0, 1)
        #c_dif = np.linalg.norm(c - c_new) / math.sqrt(c.size) / update_step
        c = c_new
        
        dz = compute_derivatives_z(num_cells, y, c, z)
        z_new = np.clip(z - update_step * dz, 0, 1)
        z_dif = np.linalg.norm(z - z_new) / math.sqrt(z.size) / update_step
        z = z_new

        if(save_images):
            z_img = np.hstack(z.reshape(num_cells, h, w))
            dz_img = np.hstack(dz.reshape(num_cells, h, w))
            dz_img = np.clip(dz_img + 0.5, 0, 1)
            progress_img[num_iter] = np.vstack((z_img, dz_img))
        
        num_iter += 1
        update_norm = z_dif 

    prods = compute_products_of_one_minus_c_times_z(num_cells, c, z)
    err = np.linalg.norm(y - 1 + prods[0]) / math.sqrt(y.size)
    print('%d cells: %d iterations with error %e' % (num_cells, num_iter, err))

    if(save_images):
        tiff.imwrite('progress_ncell%d.tif' % num_cells,
                     progress_img[0:num_iter].astype('float32'),
                     photometric='minisblack')

    return z, c, err


def demix_cells(probability_maps, num_cells, z_init,
                max_iter, update_step, iter_thresh,
                mode, save_images, num_threads):
    if(mode == 'cpu'):
        try:
            from libdemix import demix_cells_cython
        except ImportError:
            print('failed to import libdemix, '\
                  'using pure Python implementation instead')
            mode = 'py'
    elif(mode == 'gpu'):
        print('GPU implementation unsupported yet, '\
              'using pure Python implementation instead')
        mode = 'py'
    
    if(mode == 'cpu'):
        return demix_cells_cython(probability_maps, num_cells, z_init,
                                  max_iter, update_step, iter_thresh,
                                  num_threads)
    else:
        return demix_cells_py(probability_maps, num_cells, z_init,
                              max_iter, update_step, iter_thresh,
                              save_images)
    
    
def demix_cells_incrementally(probability_maps,
                              mode, save_images, num_threads):
    num_frames, h, w = probability_maps.shape
    y = np.reshape(probability_maps, (num_frames, h * w))
    z_init = np.mean(y, axis=0)[np.newaxis]
    # Try to demix while increasing the assumed number of overlapping cells
    prev_err = 1e10
    prev_z = None
    for num_cells in range(1, MAX_NUM_OVERLAPPING_CELLS+1):
        z, c, err = demix_cells(probability_maps, num_cells, z_init,
                                MAX_ITER, UPDATE_STEP, ITER_THRESH,
                                mode, save_images, num_threads)
        if(save_images):
            tiff.imwrite('ncell%d.tif' % num_cells,
                         z.reshape((num_cells, h, w)).astype('float32'),
                         photometric='minisblack')

        z_init = np.append(z, np.zeros((1, h * w)), axis=0)
        if(err < ERROR_THRESH):
            break
        if(err > prev_err * 0.8):
            z = prev_z
            num_cells -= 1
            break
        prev_err = err
        prev_z = z
    return z.reshape((num_cells, h, w))


def compute_masks(in_file, out_file, save_images=False):

    probability_maps = tiff.imread(in_file).astype(float)

    # downsample to make it faster
    down = rescale(probability_maps, (1, 0.25, 0.25), anti_aliasing=True)
    cell_img = demix_cells_incrementally(down, 'cpu', save_images, 4)
    cell_img = resize(cell_img, (cell_img.shape[0],) + probability_maps.shape[1:])
    
    # sum demixed cells to get a single image with all active cells
    sum_image = np.sum(cell_img, axis=0)

    # separate them into connected components
    threshold = filters.threshold_li(sum_image)
    binary_image = sum_image > threshold
    label_image = measure.label(binary_image)
    components = measure.regionprops(label_image)
    if(save_images):
        tiff.imwrite('sum.tif', sum_image.astype('float32'), photometric='minisblack')
        tiff.imwrite('binary.tif', binary_image, photometric='minisblack')
            
    # Process each component
    h, w = binary_image.shape
    out = np.zeros((0, h, w))
    margin = 0
    for c in components:
        component_id = c.label
        
        if(c.area < 10 or h * w / 10 < c.area):
            continue
        
        ymin, xmin, ymax, xmax = c.bbox
        if(xmin==0 and xmax < 10):
            continue
        if(xmax==w and xmin > w-10):
            continue
        if(ymin==0 and ymax < 10):
            continue
        if(ymax==h and ymin > h-10):
            continue

        # Crop a subimage (bounding box plus some margin to
        # include background pixels) from the probability maps
        ymin = max(ymin - margin, 0)
        xmin = max(xmin - margin, 0)
        ymax = min(ymax + margin, h)
        xmax = min(xmax + margin, w)
        crop = probability_maps[:, ymin:ymax, xmin:xmax]
        
        # Execlude other non-overlapping components that might
        # be within the bounding box
        label = label_image[ymin:ymax, xmin:xmax]
        self_or_background = np.logical_or(label == component_id, label == 0)
        crop = np.multiply(crop, self_or_background)
        
        # Here we could demix cells in the subimage
        # but for now use the connected component as-is
        masks = c.image[np.newaxis]

        # Put the resultant masks back in the original image shape
        uncrop = np.zeros((len(masks), h, w))
        uncrop[:, ymin:ymax, xmin:xmax] = masks
        
        out = np.concatenate((out, uncrop), axis=0)
        
        if(save_images):
            tiff.imwrite('comp%2.2d.tif' % component_id,
                         crop, photometric='minisblack')
            tiff.imwrite('mask%2.2d.tif' % component_id,
                         masks.astype('float32'), photometric='minisblack')
    
    tiff.imwrite(out_file, out.astype('float32'), photometric='minisblack')
