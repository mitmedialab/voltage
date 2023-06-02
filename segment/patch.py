import time
import numpy as np
import tifffile as tiff
from skimage.transform import resize
from scipy.signal.windows import gaussian


def _merge_frame_patches(image_shape, patches, Xs, Ys,
                         mode, weight, tile_strides):
    """
    Merge spatially-overlapping patches of probability maps in a single frame
    into a single probability map having a given shape.

    Parameters
    ----------
    image_shape : 2-tuple of integer
        Image size (H x W) over which the patches will be merged.
    patches : 3D numpy.ndarray of float
        A sequence of patches of probability maps. The shape is
        (number_of_patches, patch_height, patch_width).
    Xs : list of float
        X coordinates specifying the top left corners of the patches.
    Ys : list of float
        Y coordinates specifying the top left corners of the patches.
    mode : string
        Merger mode. Weighted average ('average') or median ('median').
    weight : 2D numpy.ndarray of float
        Patch weight map for the weighted average mode. This will give
        varying confidence to different pixels (probability values) in
        the patch depending on their locations within the patch. Typically,
        center pixels should receive more confidence than border pixels.
        The shape of the weight map must match that of the patches.
        Ignored for the median merger mode (mode='median').
    tile_strides : 2-tuple of integer
        Spacing between adjacent tiled patches for the median mode.
        This is used to calculate how many probability value samples
        each merged pixel receives before computing its median.
        Ignored for the average merger mode (mode='average').

    Returns
    -------
    2D numpy.ndarray
        Merged probability map whose shape is image_shape.

    """
    patch_shape = patches.shape[1:]

    if(mode == 'average'):
        pred_img = np.zeros(image_shape)
        pred_count = np.zeros(image_shape)
        for i, (xs, ys) in enumerate(zip(Xs, Ys)):
            ye = ys + patch_shape[0] # no greater than image_shape[0]
            xe = xs + patch_shape[1] # no greater than image_shape[1]
            pred_img[ys:ye, xs:xe] += np.multiply(weight, patches[i])
            pred_count[ys:ye, xs:xe] += weight

        pred_count[pred_count == 0] = 1 # to avoid zero division
        return pred_img / pred_count

    elif(mode == 'median'):
        # We can't weight samples in the median mode, but instead could
        # discard patch border samples. However, naive implementation would
        # leave the image border with no values. For now, we use all the
        # samples in the patches for simplicity in the hope that median
        # is robust against errors that may occur around patch borders.
        num_subtiles_y = patch_shape[0] // tile_strides[0]
        num_subtiles_x = patch_shape[1] // tile_strides[1]
        num_subtiles = num_subtiles_y * num_subtiles_x
        pred_img = np.zeros((num_subtiles,) + image_shape)
        pred_mask = np.ones((num_subtiles,) + image_shape)
        for i, (xs, ys) in enumerate(zip(Xs, Ys)):
            ye = ys + patch_shape[0] # no greater than image_shape[0]
            xe = xs + patch_shape[1] # no greater than image_shape[1]
            subtile_idx_y = (ys % patch_shape[0]) // tile_strides[0]
            subtile_idx_x = (xs % patch_shape[1]) // tile_strides[1]
            subtile_idx = subtile_idx_x * num_subtiles_y + subtile_idx_y
            pred_img[subtile_idx, ys:ye, xs:xe] = patches[i]
            pred_mask[subtile_idx, ys:ye, xs:xe] = 0

        pred_img = np.ma.masked_array(pred_img, mask=pred_mask)
        return np.ma.median(pred_img, axis=0)

    else:
        print('invalid mode: ' + mode)
        return None


def merge_patches(patches, seq, tile_strides,
                  input_paths, target_paths, out_paths, ref_paths):
    """
    Merge sliding tiles/patches predicted by U-Net into single probability maps.

    Parameters
    ----------
    patches : 3D numpy.ndarray of float
        Patches of probability maps from multiple frames of the U-Net output.
        The shape is (number_of_patches, patch_height, patch_width).
    seq : VI_Sequence
        Sequence object used to feed data to the model for prediction.
    tile_strides : tuple (y, x) of integer
        Spacing between adjacent tiles.
    input_paths : list of list of pathlib.Path, or None
        List of file paths to input images. Each element of the list is
        a list of file paths corresponding to multiple channels. It may be
        None, in which case input images will not be saved to ref_paths.
    target_paths : list of pathlib.Path, or None
        List of file paths to target images specifing expected outputs. It may
        be None, in which case target images will not be saved to ref_paths.
    out_paths : list of pathlib.Path, or None
        List of file paths to which merged U-Net outputs will be saved. It may
        be None, in which case the outputs will not be saved.
    ref_paths : list of pathlib.Path, or None
        List of file paths to which U-Net inputs, merged outputs, and targets
        (ground truth) if any, are juxtaposed and saved for visual inspection.
        It may be None, in which case the reference images will not be saved.

    Returns
    -------
    out : 3D numpy.ndarray of float
        Merged probability maps.
        The shape is (number_of_frames, image_height, image_width).

    """
    tic = time.perf_counter()

    Ys, Xs = seq.get_tile_pos()
    num_tiles = len(Ys)
    num_frames = seq.num_frames
    image_shape = seq.image_shape
    patch_h, patch_w = patches.shape[1:]

    # construct a weight matrix for average patch merger
    sigma = (patch_h + patch_w) / 2 / 3
    gauss_y = gaussian(patch_h, sigma)
    gauss_x = gaussian(patch_w, sigma)
    weight = np.outer(gauss_y, gauss_x)

    if(input_paths is None):
        input_paths = [None] # assume there is one input video

    for i, paths in enumerate(input_paths):
        out = np.zeros((num_frames,) + image_shape)
        for j in range(num_frames):
            idx = (i * num_frames + j) * num_tiles
            frame_patches = patches[idx:idx+num_tiles]
            out[j] = _merge_frame_patches(image_shape, frame_patches, Xs, Ys,
                                          'average', weight, tile_strides)

        if(seq.needs_padding == 'magnify'): # undo magnification
            out = resize(out, (len(out),) + seq.orig_image_shape,
                         mode='constant', anti_aliasing=True)
        elif(seq.needs_padding == 'padding'): # remove padding
            h, w = seq.orig_image_shape
            y = max((patch_h - h) // 2, 0)
            x = max((patch_w - w) // 2, 0)
            out = out[:, y:y+h, x:x+w]

        if(out_paths is not None):
            tiff.imwrite(out_paths[i], out.astype('float32'),
                         photometric='minisblack')

        # reference output for visual inspection
        if(ref_paths is not None):
            video = np.zeros(out.shape[:2] + (0,))
            if(paths is not None):
                for path in paths:
                    img = tiff.imread(path)
                    video = np.append(video, img / np.max(img), axis=2)
            if(target_paths is not None):
                target_img = tiff.imread(target_paths[i])
                video = np.append(video, target_img, axis=2)
            video = np.append(video, out, axis=2)
            tiff.imwrite(ref_paths[i], video.astype('float32'),
                         photometric='minisblack')

    toc = time.perf_counter()
    print('Patch merger: %.1f sec' % (toc - tic))

    return out
