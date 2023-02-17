import math
import keras
import tensorflow as tf
import numpy as np
import tifffile as tiff
from pathlib import Path
from skimage.transform import resize
from scipy.signal.windows import gaussian

from .sequence import VI_Sequence
from .data import get_training_data, get_inference_data
from .loss import weighted_bce, dice_loss, bce_dice_loss, iou_loss


def _prevent_tf_from_occupying_entire_gpu_memory():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


def _load_model(model_file):
    """
    Load a U-Net model from a file.

    Parameters
    ----------
    model_file : string or pathlib.Path
        File path containing a model.

    Returns
    -------
    model : keras.Model
        The loaded model.
    io_shape : tuple (height, width) of integer
        Input/output shape of the model.

    """
    keras.backend.clear_session()
    loss_dict = {'weighted_bce': weighted_bce,
                 'dice_loss': dice_loss,
                 'bce_dice_loss': bce_dice_loss,
                 'iou_loss': iou_loss}
    model = keras.models.load_model(model_file, custom_objects=loss_dict)
    # model.input_shape = (None, height, width, num_channels), where the first
    # element represents the batch size, which is undefined at this point
    io_shape = model.input_shape[1:3]
    return model, io_shape


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


def _predict_and_merge(model, data_seq, tile_strides, gpu_mem_size,
                       input_paths, target_paths, out_paths, ref_paths):
    """
    Make prediction using a given U-Net model on sliding tiles/patches,
    and merge them into single probability maps.

    Parameters
    ----------
    model : keras.Model
        Learned U-Net model.
    data_seq : VI_Sequence
        Sequence object used to feed data to the model.
    tile_strides : tuple (y, x) of integer
        Spacing between adjacent tiles.
    gpu_mem_size : float or None
        GPU memory size in gigabytes (GB) that can be allocated for buffering
        prediction outputs. If None, no limit is assumed.
    input_paths : list of list of pathlib.Path
        List of file paths to input images. Each element of the list is
        a list of file paths corresponding to multiple channels.
    target_paths : list of pathlib.Path
        List of file paths to target images specifing expected outputs.
        It can be None, in which case only U-Net inputs and outputs will
        be saved to ref_paths.
    out_paths : list of pathlib.Path
        List of file paths to which U-Net outputs will be saved.
    ref_paths : list of pathlib.Path
        List of file paths to which U-Net inputs, outputs, and targets (i.e.,
        ground truth) if any, are juxtaposed and saved for visual inspection.

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
            preds = np.append(preds, ret, axis=0)
    preds = preds[:, :, :, 0] # remove last dimension (its length is one)
    if(preds.shape[1:] != data_seq.patch_shape):
        preds = resize(preds, (len(preds),) + data_seq.patch_shape,
                       mode='edge', anti_aliasing=True)

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

        if(data_seq.needs_padding == 'magnify'): # undo magnification
            pred_img = resize(pred_img, input_imgs[0].shape,
                              mode='constant', anti_aliasing=True)
        elif(data_seq.needs_padding == 'padding'): # remove padding
            h, w = input_imgs[0].shape[1:]
            y = max((patch_shape[0] - h) // 2, 0)
            x = max((patch_shape[1] - w) // 2, 0)
            pred_img = pred_img[:, y:y+h, x:x+w]
        tiff.imwrite(out_paths[i], pred_img.astype('float32'),
                     photometric='minisblack')

        # reference output for visual inspection
        video = np.zeros(pred_img.shape[:2] + (0,))
        for img in input_imgs:
            video = np.append(video, img / np.max(img), axis=2)
        if(target_paths is not None):
            target_img = tiff.imread(target_paths[i])
            video = np.append(video, target_img, axis=2)
        video = np.append(video, pred_img, axis=2)
        tiff.imwrite(ref_paths[i], video.astype('float32'),
                     photometric='minisblack')


def validate_model(input_dir_list, target_dir, model_dir, out_dir, ref_dir,
                   seed, validation_ratio,
                   norm_channel, norm_shifts,
                   tile_shape, tile_strides, batch_size, gpu_mem_size=None):
    """
    Validate a learned U-Net by making inference on a validation data set.

    This is different from validation during training, which is performed on
    a patch-by-patch basis for randomly chosen patches from a validation set.
    In contrast, this function performs inference on sliding tiles and merge
    them into probability maps having the same size as input images, which is
    the same process as performing inference on test/real data as implemented
    in apply_model(), making the results more suitable for visual inspection.

    In order to use the same validation set as training, the parameters
    input_dir_list, target_dir, seed, and validation_ratio
    must be the same as supplied to the training function train_model().

    Parameters
    ----------
    input_dir_list : list of pathlib.Path
        List of directory paths containing input files. Each directory path
        corresponds to one channel of the input.
    target_dir : pathlib.Path
        Directory path containing target files.
    model_dir : string or pathlib.Path
        Directory path containing learned models. The model with the smallest
        validation loss will be used.
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
    tile_shape : tuple (height, width) of integer
        Size of tiles to be extracted from images.
    tile_strides : tuple (y, x) of integer
        Spacing between adjacent tiles.
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

    # sort model file names by validation loss
    model_files = sorted(Path(model_dir).glob('model*.h5'),
                         key=lambda x: float(x.stem.split('v')[-1]))
    # first item of the sorted list has the lowest validation loss
    model, model_io_shape = _load_model(model_files[0])

    valid_seq = VI_Sequence(batch_size, model_io_shape, tile_shape,
                            valid_input_paths, valid_target_paths,
                            norm_channel, norm_shifts,
                            tiled=True, tile_strides=tile_strides)

    out_paths = [out_dir.joinpath(p.name) for p in valid_target_paths]
    ref_paths = [ref_dir.joinpath(p.name) for p in valid_target_paths]

    _predict_and_merge(model, valid_seq, tile_strides, gpu_mem_size,
                       valid_input_paths, valid_target_paths,
                       out_paths, ref_paths)

    keras.backend.clear_session()


def apply_model(input_files, model_file, out_file, ref_file,
                norm_channel, norm_shifts,
                tile_shape, tile_strides, batch_size, gpu_mem_size=None):
    """
    Apply a learned U-Net by making inference on a test/real data set.

    Parameters
    ----------
    input_files : list of pathlib.Path
        List of input file paths. Each corresponds to one channel of the input.
    model_file : string or pathlib.Path
        File path containing a learned model.
    out_file : pathlib.Path
        File path to which the U-Net output will be saved.
    ref_file : pathlib.Path
        File path to which the U-Net input and output are juxtaposed
        and saved for visual inspection.
    tile_shape : tuple (height, width) of integer
        Size of tiles to be extracted from images.
    tile_strides : tuple (y, x) of integer
        Spacing between adjacent tiles.
    batch_size : integer
        Batch size for inference.
    gpu_mem_size : float or None
        GPU memory size in gigabytes (GB) that can be allocated for buffering
        prediction outputs. The default is None (no limit is assumed).

    Returns
    -------
    None.

    """
    _prevent_tf_from_occupying_entire_gpu_memory()

    model, model_io_shape = _load_model(model_file)

    data_seq = VI_Sequence(batch_size, model_io_shape, tile_shape,
                           [input_files], None,
                           norm_channel, norm_shifts,
                           tiled=True, tile_strides=tile_strides)

    _predict_and_merge(model, data_seq, tile_strides, gpu_mem_size,
                       [input_files], None, [out_file], [ref_file])

    keras.backend.clear_session()
