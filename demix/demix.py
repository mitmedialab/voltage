import math
import numpy as np
import tifffile as tiff
from skimage import measure
#from sklearn.decomposition import NMF


MAX_NUM_OVERLAPPING_CELLS = 3
ERROR_THRESH = 1e-3
MAX_ITER = 10000
UPDATE_STEP = 1.0e-3
ITER_THRESH = 1.0e-3


def compute_products_of_one_minus_c_times_z(num_cells, c, z):
    num_frames = c.shape[1]
    num_pixels = z.shape[1]

    one_minus_c_times_z = np.ones((num_cells, num_frames, num_pixels))
    for i in range(num_cells):
        c_i_as_col = c[i, :, np.newaxis]
        z_i_as_row = z[np.newaxis, i, :]
        one_minus_c_times_z[i] = 1 - np.matmul(c_i_as_col, z_i_as_row)

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


def init_masks(image, num_cells):
    #model = NMF(n_components=num_cells)
    #return model.components_
    num_frames, h, w = image.shape
    return np.random.rand(num_cells, h * w)


def demix_cells_with_given_number(image, num_cells, component_id, save_images):
    num_frames, h, w = image.shape
    c = np.random.rand(num_cells, num_frames)
    z = init_masks(image, num_cells)
    y = np.reshape(image, (num_frames, h * w))

    # Alternate gradient descent, iterate until the masks no longer change
    # Important to clip values at [0, 1] as they represent probabilities
    # Diff should be taken after clipping, and should not be computed simply
    # as the magnitude of the derivatives (which could prevent convergence)
    num_iter = 0
    update_norm = ITER_THRESH + 1
    progress_img = np.zeros((MAX_ITER, h * 2, w * num_cells))
    while(num_iter < MAX_ITER and update_norm > ITER_THRESH):
        dc = compute_derivatives_c(num_cells, y, c, z)
        c_new = np.clip(c - UPDATE_STEP * dc, 0, 1)
        #c_dif = np.linalg.norm(c - c_new) / math.sqrt(c.size) / UPDATE_STEP
        c = c_new
        
        dz = compute_derivatives_z(num_cells, y, c, z)
        z_new = np.clip(z - UPDATE_STEP * dz, 0, 1)
        z_dif = np.linalg.norm(z - z_new) / math.sqrt(z.size) / UPDATE_STEP
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
    
    print('component %d, # cells %d, iteration %d, update %e, error %e'
          % (component_id, num_cells, num_iter, update_norm, err))
    if(save_images):
        tiff.imwrite('comp%2.2d_ncell%d.tif' % (component_id, num_cells),
                     progress_img[0:num_iter].astype('float32'),
                     photometric='minisblack')

    return z.reshape((num_cells, h, w)), err

        
def demix_cells_subimage(image, component_id, save_images):
    # Try to demix while increasing the assumed number of overlapping cells
    # until the approximation error is small enough
    for num_cells in range(1, MAX_NUM_OVERLAPPING_CELLS+1):
        masks, err = demix_cells_with_given_number(image, num_cells,
                                                   component_id,
                                                   save_images)
        if(err < ERROR_THRESH):
            break
    return masks


def demix_cells(probability_maps, threshold=0.5, margin=5, save_images=False):

    # Extract connected components after threshoulding
    binary_image = np.any(probability_maps > threshold, axis=0)
    label_image = measure.label(binary_image)
    components = measure.regionprops(label_image)

    # Process each component
    h, w = binary_image.shape
    out = np.zeros((0, h, w))
    for c in components:
        component_id = c.label
        
        # Crop a subimage (bounding box plus some margin to
        # include background pixels) from the probability maps
        ymin, xmin, ymax, xmax = c.bbox
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
        
        # Demix cells in the subimage
        masks = demix_cells_subimage(crop, component_id, save_images)

        # Put the resultant masks back in the original image shape
        uncrop = np.zeros((len(masks), h, w))
        uncrop[:, ymin:ymax, xmin:xmax] = masks
        out = np.concatenate((out, uncrop), axis=0)
        
        if(save_images):
            tiff.imwrite('comp%2.2d.tif' % component_id,
                         crop, photometric='minisblack')
            tiff.imwrite('mask%2.2d.tif' % component_id,
                         masks.astype('float32'), photometric='minisblack')
        
    return out
