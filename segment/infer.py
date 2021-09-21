import numpy as np
import tifffile as tiff
from skimage.transform import resize
from scipy.signal.windows import gaussian
from keras import models

from .sequence import VI_Sequence
from .data import get_training_data, get_inference_data


def _merge_patches(image_shape, patches, Xs, Ys,
                   mode, weight, tile_strides):
    """
    Merge spatially-overlapping patches of probability maps
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


def predict_and_merge(model, data_seq, tile_strides,
                      input_paths, target_paths, out_dir, ref_dir, batch_size):
    
    preds = np.zeros((batch_size * len(data_seq), 64, 64, 1), np.float32)
    for i in range(len(data_seq)):
        preds[i * batch_size : (i * batch_size) + batch_size] = model.predict(data_seq.__getitem__(i)[0], batch_size=batch_size)

    preds = preds[:, :, :, 0] # remove last dimension (its length is one)
    
    Ys, Xs = data_seq.get_tile_pos()
    num_tiles = len(Ys)
    num_frames = data_seq.num_frames
    image_shape = data_seq.image_shape
    patch_shape = preds.shape[1:]
    
    std = (patch_shape[0] + patch_shape[1]) / 2 / 3
    gauss_y = gaussian(patch_shape[0], std)
    gauss_x = gaussian(patch_shape[1], std)
    weight = np.outer(gauss_y, gauss_x)
    
    for i, paths in enumerate(input_paths):
        pred_img = np.zeros((num_frames,) + image_shape)
        for j in range(num_frames):
            idx = (i * num_frames + j) * num_tiles
            patches = preds[idx:idx+num_tiles]
            pred_img[j] = _merge_patches(image_shape, patches, Xs, Ys,
                                         'average', weight, tile_strides)
        
        input_imgs = []
        for path in paths:
            input_imgs.append(tiff.imread(path))

        if(data_seq.needs_resizing):
            pred_img = resize(pred_img, input_imgs[0].shape,
                              anti_aliasing=True)
        tiff.imwrite(out_dir.joinpath(paths[0].name),
                     pred_img.astype('float32'), photometric='minisblack')
        
        # reference output for visual inspection
        video = np.zeros(pred_img.shape[:2] + (0,))
        for img in input_imgs:
            video = np.append(video, img / np.max(img), axis=2)
        if(target_paths is not None):
            target_img = tiff.imread(target_paths[i])
            video = np.append(video, target_img, axis=2)
        video = np.append(video, pred_img, axis=2)
        tiff.imwrite(ref_dir.joinpath(paths[0].name),
                     video.astype('float32'), photometric='minisblack')


def validate_model(input_dir_list, target_dir, model_dir, out_dir, ref_dir,
                   seed, validation_ratio,
                   patch_shape, tile_strides, batch_size):

    data = get_training_data(input_dir_list, target_dir,
                             seed, validation_ratio)
    valid_input_paths = data[2]
    valid_target_paths = data[3]
    
    valid_seq = VI_Sequence(batch_size, patch_shape,
                            valid_input_paths, valid_target_paths,
                            tiled=True, tile_strides=tile_strides)

    model = models.load_model(model_dir.joinpath('model.h5'))

    predict_and_merge(model, valid_seq, tile_strides,
                      valid_input_paths, valid_target_paths, out_dir, ref_dir, batch_size)


def apply_model(input_dir_list, model_dir, out_dir, ref_dir, filename,
                patch_shape, tile_strides, batch_size):

    input_files = get_inference_data(input_dir_list)
    model = models.load_model(model_dir.joinpath('model.h5'))
    for paths in input_files:
        if(filename and paths[0].stem != filename):
            continue
        
        data_seq = VI_Sequence(batch_size, patch_shape,
                               [paths], None,
                               tiled=True, tile_strides=tile_strides)
        
        predict_and_merge(model, data_seq, tile_strides,
                          [paths], None, out_dir, ref_dir, batch_size)
