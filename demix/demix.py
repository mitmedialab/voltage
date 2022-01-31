import math
import numpy as np
import tifffile as tiff


MAX_NUM_OVERLAPPING_CELLS = 5
ERROR_THRESH_ABSOLUTE = 1e-3
ERROR_THRESH_RELATIVE = 0.5
MAX_ITER = 10000
UPDATE_STEP = 1.0e-2
ITER_THRESH = 1.0e-3


def compute_products_of_one_minus_c_times_z(num_cells, c, z):
    """
    Compute various products of (1 - c[i, t] . z[i, x]),
    which are commonly used in the optimization calculation.

    Parameters
    ----------
    num_cells : integer
        The number of cells.
    c : 2D numpy.ndarray of float
        Current estimate of the temporal probability.
        The shape is (num_cells, num_frames).
    z : 2D numpy.ndarray of float
        Current estimate of the spatial probability where pixels are flattened.
        The shape is (num_cells, num_pixels).

    Returns
    -------
    out : list of 2D numpy.ndarray of float
        Product values.
        Each element of the list has the shape (num_frames, num_pixels).
        The first element is the product for all i, that is,
        Prod_{i=0}^{N-1} (1 - c[i, t] . z[i, x]) for various (t, x).
        The second element is the product for all i but 0, that is,
        Prod_{i=1}^{N-1} (1 - c[i, t] . z[i, x]) for various (t, x).
        The third element excludes i=1, the fourth i=2, and so on.

    """
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
    """
    Compute the derivatives of the least squares objective function w.r.t. c.

    Parameters
    ----------
    num_cells : integer
        The number of cells.
    y : 2D numpy.ndarray of float
        Observed probability maps where pixels are flattened.
        The shape is (num_frames, num_pixels).
    c : 2D numpy.ndarray of float
        Current estimate of the temporal probability.
        The shape is (num_cells, num_frames).
    z : 2D numpy.ndarray of float
        Current estimate of the spatial probability where pixels are flattened.
        The shape is (num_cells, num_pixels).

    Returns
    -------
    dc : 2D numpy.ndarray of float
        Derivatives w.r.t. c.

    """
    prods = compute_products_of_one_minus_c_times_z(num_cells, c, z)
    first_term = y - 1 + prods[0]
    dc = np.zeros(c.shape)
    for j in range(num_cells):
        z_j_as_row = z[np.newaxis, j, :]
        second_term = np.multiply(prods[j+1], z_j_as_row)
        dc[j] = -np.sum(np.multiply(first_term, second_term), axis=1)
    return dc


def compute_derivatives_z(num_cells, y, c, z):
    """
    Compute the derivatives of the least squares objective function w.r.t. z.

    Parameters
    ----------
    num_cells : integer
        The number of cells.
    y : 2D numpy.ndarray of float
        Observed probability maps where pixels are flattened.
        The shape is (num_frames, num_pixels).
    c : 2D numpy.ndarray of float
        Current estimate of the temporal probability.
        The shape is (num_cells, num_frames).
    z : 2D numpy.ndarray of float
        Current estimate of the spatial probability where pixels are flattened.
        The shape is (num_cells, num_pixels).

    Returns
    -------
    dz : 2D numpy.ndarray of float
        Derivatives w.r.t. z.

    """
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
    """
    Pure Python implementation of demix_cells().
    See demix_cells() for parameter definitions.
    """
    num_frames, h, w = probability_maps.shape
    # Flatten pixels as all the computation will be pixel-wise
    y = np.reshape(probability_maps, (num_frames, h * w))
    # Initialize the temporal probability randomly
    c = np.random.rand(num_cells, num_frames)
    # Initialize the spatial probability to the given value while flattening
    z = np.reshape(z_init, (num_cells, h * w))

    if(save_images):
        progress_img = np.zeros((max_iter, h * 2, w * num_cells))

    # Alternate gradient descent, iterate until the masks no longer change.
    # Important to clip values at [0, 1] as they represent probabilities.
    # Diff should be taken after clipping, and should not be computed simply
    # as the magnitude of the derivatives (which could prevent convergence).
    num_iter = 0
    update_norm = iter_thresh + 1
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

    if(save_images):
        tiff.imwrite(save_images + '_progress_num_cells%d.tif' % num_cells,
                     progress_img[0:num_iter].astype('float32'),
                     photometric='minisblack')

    return z.reshape(num_cells, h, w), c, err, num_iter


def demix_cells(probability_maps, num_cells, z_init,
                max_iter, update_step, iter_thresh,
                mode, num_threads, save_images):
    """
    Demix cells from a sequence of probability maps of firing neurons,
    assuming there are a given number N of overlapping cells.

    This is done by solving the following equation in the least squares sense:

        1 - y[t, x, y] = Prod_{i=0}^{N-1} (1 - z[i, x, y] . c[i, t]),

    where y represents the input sequence of probability maps, and z and c
    are as defined in the output. In words, the probability that pixel (x, y)
    at time t does NOT correspond to any firing neurons is equal to the
    probability that the following does NOT happen for ALL i: the pixel (x, y)
    belongs to the i-th firing neuron AND the i-th neuron fires at time t.

    The least squares minimization is solved for z and c alternately using
    gradient descent. As a result, the input observation y will be decomposed
    into the spatial probability z and temporal probability c, and z represents
    the demixed cell footprints.

    Parameters
    ----------
    probability_maps : 3D numpy.ndarray of float
        Sequence of probability maps output by the U-Net indicating how likely
        each pixel at each time instance corresponds to firing neurons.
        The shape is (num_frames, height, width).
    num_cells : integer
        The assumed number of overlapping cells.
    z_init : 3D numpy.ndarray of float
        Initial estimate of the probability maps of demixed cells. The shape
        is (num_cells, height, width).
    max_iter : integer
        The maximum number of gradient descent interations.
    update_step : float
        At each iteration, the estimate will be moved by the negative gradient
        scaled by this number.
    iter_thresh : float
        Iteration stops when the update becomes smaller than this number.
    mode : string
        Execution mode. Options are 'cpu' (multithreaded C++ on CPU),
        and 'py' (pure Python implementation).
    num_threads : integer
        The number of threads for the multithreaded C++ execution (mode='cpu').
    save_images : string
        If non-empty, intermediate images will be saved for debugging.
        The specified string will be used as a prefix for image filenames.
        Some (optimization progress) images will be saved only when mode='py'.

    Returns
    -------
    z : 3D numpy.ndarray of float
        Spatial probability maps of demixed cells.
        The shape is (num_cells, height, width). The i-th image represents
        the probability that a given pixel belongs to the i-th cell.

    c : 2D numpy.ndarray of float
        Temporal probability maps of demixed cells.
        The shape is (num_cells, num_frames). It represents the probability
        that a given cell fires at a given frame.

    err : float
        Error representing how well the specified number of cells explains
        the observed probability maps.

    n : integer
        The number of interations performed before convergence.

    """
    if(mode == 'cpu'):
        try:
            from libdemix import demix_cells_cython
        except ImportError:
            print('failed to import libdemix, '\
                  'using pure Python implementation instead')
            mode = 'py'
    
    if(mode == 'cpu'):
        z, c, err, n = demix_cells_cython(probability_maps, num_cells, z_init,
                                          max_iter, update_step, iter_thresh,
                                          num_threads)
    else:
        z, c, err, n = demix_cells_py(probability_maps, num_cells, z_init,
                                      max_iter, update_step, iter_thresh,
                                      save_images)
    return z, c, err, n


def demix_cells_incrementally(probability_maps,
                              mode, num_threads, save_images):
    """
    Demix potentially overlapping cells from a sequence of probability maps
    of firing neurons. It begins with a single-cell explanation and looks
    for the best explanation by incrementing the number of overlapping cells.

    Parameters
    ----------
    probability_maps : 3D numpy.ndarray of float
        Sequence of probability maps output by the U-Net indicating how likely
        each pixel at each time instance corresponds to firing neurons.
        The shape is (num_frames, height, width).
    mode : string
        Execution mode. Options are 'cpu' (multithreaded C++ on CPU),
        and 'py' (pure Python implementation). GPU mode is not available
        (and probably will not be in the future) because the workload is
        not massively parallelizable.
    num_threads : integer
        The number of threads for the multithreaded C++ execution (mode='cpu').
    save_images : string
        If non-empty, intermediate images will be saved for debugging.
        The specified string will be used as a prefix for image filenames.
        Some (optimization progress) images will be saved only when mode='py'.

    Returns
    -------
    3D numpy.ndarray of float
        Demixed cells. The shape is (num_cells, height, width), consisting of
        as many images as the estimated number of overlapping cells. Each
        image represents the spatial probability map of each firing cell.

    """
    num_frames, h, w = probability_maps.shape
    # Initial cell probability for a single-cell explanation
    # is simply the temporal average of the probability maps
    z_init = np.mean(probability_maps, axis=0)[np.newaxis]
    # Try to demix while incrementing the assumed number of overlapping cells
    prev_err = 1e10
    prev_z = None
    for num_cells in range(1, MAX_NUM_OVERLAPPING_CELLS+1):
        z, c, err, n = demix_cells(probability_maps, num_cells, z_init,
                                   MAX_ITER, UPDATE_STEP, ITER_THRESH,
                                   mode, num_threads, save_images)
        if(save_images):
            tiff.imwrite(save_images + '_num_cells%d.tif' % num_cells,
                         z.astype('float32'), photometric='minisblack')

        print('%d cells: %d iterations with error %e' % (num_cells, n, err))

        # If the error is small enough, the current estimate is already good,
        # and even though further incrementing the number of cells will likely
        # further reduce the error, it would likely lead to oversplitting
        if(err < ERROR_THRESH_ABSOLUTE):
            print('absolute threshold reached')
            break

        # If the error did not decrease sufficiently from the previous
        # iteration, it is likely that the current estimate oversplitted
        # the cell(s) even if the absolute error is still large.
        # If that happens, output the previous estimate.
        if(err > prev_err * ERROR_THRESH_RELATIVE):
            print('relative threshold reached')
            return prev_z

        # For the next iteration with one more cell, use the current estimate
        # for N cells along with zero for N+1-th cell as the initial estimate
        z_init = np.append(z, np.zeros((1, h, w)), axis=0)
        prev_err = err
        prev_z = z

    return z
