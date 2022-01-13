import math
import numpy as np
import tifffile as tiff
from skimage.transform import resize
from scipy.signal.windows import gaussian
from keras import models

from .sequence import VI_Sequence
from .data import get_training_data, get_inference_data
from .loss import weighted_bce, dice_loss, bce_dice_loss, iou_loss


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


def predict_and_merge(model, data_seq, tile_strides, gpu_mem_size,
                      input_paths, target_paths, out_dir, ref_dir):
    """
    Make prediction using a given U-Net model on sliding patches, and merge
    them into single probability maps.

    Parameters
    ----------
    model : keras.Model
        Learned U-Net model.
    data_seq : VI_Sequence
        Sequence object used to feed data to the model.
    tile_strides : tuple (y, x) of integer
        Spacing between adjacent tiles/patches.
    gpu_mem_size : float or None
        GPU memory size in gigabytes (GB) that can be allocated for buffering
        prediction outputs. If None, no limit is assumed.
    input_paths : list of list of pathlib.Path
        List of file paths to input images. Each element of the list is
        a list of file paths corresponding to multiple channels.
    target_paths : list of pathlib.Path
        List of file paths to target images specifing expected outputs.
        It can be None, in which case only U-Net inputs and outputs will
        be saved to ref_dir.
    out_dir : pathlib.Path
        Directory path to which U-Net outputs will be saved.
    ref_dir : pathlib.Path
        Directory path to which U-Net inputs, outputs, and targets (ground
        truth) if any, are juxtaposed and saved for visual inspection.

    Returns
    -------
    None.

    """
    
    # model.predict() allocates a buffer on the GPU to hold all the output
    # prediction data before passing it back to the main memory, which can
    # cause GPU out-of-memory error if data_seq feeds too many samples.
    # When this happens, it cannot be avoided by reducing the batch size, as
    # the number of batches increases and the output data size stays the same.
    # To avoid the error, we can split the data_seq sequence into multiple
    # subsequences and run predict() for each of them, after which we join
    # the outputs together on the main memory.
    data_size = data_seq.output_data_size() / 1000000000 # in gigabytes
    if(gpu_mem_size is None):
        gpu_mem_size = data_size
    num_splits = math.ceil(data_size / gpu_mem_size)
    if(num_splits > 1):
        print('splitting %.1f GB output data into %d parts'
              % (data_size, num_splits))
    for i in range(num_splits):
        data_seq.split_samples(num_splits, i)
        ret = model.predict(data_seq, verbose=1)
        if(i == 0):
            preds = ret
        else:
            preds = np.concatenate((preds, ret), axis=0)
    preds = preds[:, :, :, 0] # remove last dimension (its length is one)
    
    Ys, Xs = data_seq.get_tile_pos()
    num_tiles = len(Ys)
    num_frames = data_seq.num_frames
    image_shape = data_seq.image_shape
    patch_shape = preds.shape[1:]

    # construct a weight matrix for average patch merger
    sigma = (patch_shape[0] + patch_shape[1]) / 2 / 3
    gauss_y = gaussian(patch_shape[0], sigma)
    gauss_x = gaussian(patch_shape[1], sigma)
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
                   patch_shape, tile_strides, batch_size, gpu_mem_size=None):
    """
    Validate a learned U-Net by making inference on a validation data set.

    This is different from validation during training, which is performed on
    a patch-by-patch basis for randomly chosen patches from a validation set.
    In contrast, this function performs inference on sliding patches and merge
    them into probability maps having the same size as input images, which is
    the same process as performing inference on test/real data as implemented
    in apply_model(), making the results more suitable for visual inspection.

    In order to use the same validation set as training, the parameters
    input_dir_list, target_dir, seed, validation_ratio, and patch_shape
    must be the same as supplied to the training function train_model().

    Parameters
    ----------
    input_dir_list : list of pathlib.Path
        List of directory paths containing input files. Each directory path
        corresponds to one channel of the input.
    target_dir : pathlib.Path
        Directory path containing target files.
    model_dir : pathlib.Path
        Directory path containing a learned model.
    out_dir : pathlib.Path
        Directory path to which U-Net outputs will be saved.
    ref_dir : pathlib.Path
        Directory path to which U-Net inputs, outputs, and targets (ground
        truth) are juxtaposed and saved for visual inspection.
    seed : integer
        Seed for randomized splitting into traning and validation data.
    validation_ratio : integer
        What fraction of the inputs are used for validation. If there are
        N inputs, N/validation_ratio of them will be used for validation.
    patch_shape : tuple (height, width) of integer
        Size of patches to be extracted from images.
    tile_strides : tuple (y, x) of integer
        Spacing between adjacent tiles/patches.
    batch_size : integer
        Batch size for inference.
    gpu_mem_size : float or None
        GPU memory size in gigabytes (GB) that can be allocated for buffering
        prediction outputs. The default is None (no limit is assumed).

    Returns
    -------
    None.

    """

    data = get_training_data(input_dir_list, target_dir,
                             seed, validation_ratio)
    valid_input_paths = data[2]
    valid_target_paths = data[3]
    
    valid_seq = VI_Sequence(batch_size, patch_shape,
                            valid_input_paths, valid_target_paths,
                            tiled=True, tile_strides=tile_strides)

    model = models.load_model(model_dir.joinpath('model.h5'),
                              custom_objects={'weighted_bce': weighted_bce,
                                              'dice_loss': dice_loss,
                                              'bce_dice_loss': bce_dice_loss,
                                              'iou_loss': iou_loss})

    predict_and_merge(model, valid_seq, tile_strides, gpu_mem_size,
                      valid_input_paths, valid_target_paths, out_dir, ref_dir)


def apply_model(input_dir_list, model_dir, out_dir, ref_dir, filename,
                patch_shape, tile_strides, batch_size, gpu_mem_size=None):
    """
    Apply a learned U-Net by making inference on a test/real data set.

    Parameters
    ----------
    input_dir_list : list of pathlib.Path
        List of directory paths containing input files. Each directory path
        corresponds to one channel of the input.
    model_dir : pathlib.Path
        Directory path containing a learned model.
    out_dir : pathlib.Path
        Directory path to which U-Net outputs will be saved.
    ref_dir : pathlib.Path
        Directory path to which U-Net inputs and outputs are juxtaposed
        and saved for visual inspection.
    filename : string
        If non-empty, only the input whose stem (filename excluding directory
        and extension) matches the specified string will be processed.
        If empty, all the inputs in input_dir_list will be processed.
    patch_shape : tuple (height, width) of integer
        Size of patches to be extracted from images.
    tile_strides : tuple (y, x) of integer
        Spacing between adjacent tiles/patches.
    batch_size : integer
        Batch size for inference.
    gpu_mem_size : float or None
        GPU memory size in gigabytes (GB) that can be allocated for buffering
        prediction outputs. The default is None (no limit is assumed).

    Returns
    -------
    None.

    """

    input_files = get_inference_data(input_dir_list)
    model = models.load_model(model_dir.joinpath('model.h5'),
                              custom_objects={'weighted_bce': weighted_bce,
                                              'dice_loss': dice_loss,
                                              'bce_dice_loss': bce_dice_loss,
                                              'iou_loss': iou_loss})
    for paths in input_files:
        if(filename and paths[0].stem != filename):
            continue

        print('processing ' + filename)

        data_seq = VI_Sequence(batch_size, patch_shape,
                               [paths], None,
                               tiled=True, tile_strides=tile_strides)

        predict_and_merge(model, data_seq, tile_strides, gpu_mem_size,
                          [paths], None, out_dir, ref_dir)
