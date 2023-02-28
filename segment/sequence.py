import random
import math
import numpy as np
import tifffile as tiff
from skimage.transform import resize
from keras.utils import Sequence


class VI_Sequence(Sequence):

    def __init__(self, batch_size, model_io_shape, patch_shape,
                 input_img_paths, target_img_paths, input_imgs=None,
                 norm_channel=-1, norm_shifts=[],
                 num_darts=1, tiled=False, tile_strides=(1, 1),
                 shuffle=False, padding='magnify'):
        """
        Initializes VI_Sequence (Voltage Imaging) instance.

        Parameters
        ----------
        batch_size : integer
            Batch size for training/inference.
        model_io_shape : tuple (height, width) of integers
            The U-Net model's input/output shape. Images of this size will be
            produced by this sequence (and fed into the U-Net).
        patch_shape : tuple (height, width) of integers
            Image patch size to be extracted from input image files.
            For training, it is recommended that patch_shape = model_io_shape,
            so that extracted patches will be fed to the model as they are.
            For inference, patch_shape may be smaller or larger, in which case
            patches will be resized to model_io_shape expected by the model.
            The training image size should be equal to (when num_darts=1) or
            larger than (when num_darts>1) patch_shape. The test image size is
            expected to be larger than patch_shape in general, but when it is
            smaller, images will be padded.
        input_img_paths : list of list of pathlib.Path
            List of file paths to input images to be fed into the U-Net. Each
            element of the list is a list of file paths corresponding to
            multiple channels. Each file (tiff) can contain multiple images.
            The number of images per file and the image size must be the same
            for all the files. This parameter may be None, in which case
            input_imgs will be used as input.
        target_img_paths : list of pathlib.Path
            List of file paths to target images specifing expected outputs from
            the U-Net. Each file (tiff) must contain the same number of images
            of the same size as the corresponding input file. This parameter is
            used only for training. It should be None for inference.
        input_imgs : list of list of 3D numpy.ndarray of float, optional
            List of input images to be fed into the U-Net. Each element of the
            list is a list of images corresponding to multiple channels.
            The default is None, in which case the input images will be loaded
            from input_image_paths.
        norm_channel : integer, optional
            The channel used to determine the scale for patch normalization.
            If -1 (default), the max/min intensities of a given patch across
            all the channels will be used. If nonnegative, the max/min of the
            specified channel will be used.
        norm_shifts : list of boolean, optional
            Whether or not to shift the intensities of an image patch so the
            channel-wise minimum becomes zero. The default is [] (no shifting).
        num_darts : integer, optional
            The number of darts to be thrown per image to extract patches
            from the image. If num_darts=1 (default), one image patch is
            extracted from the center of the image. If >1, each dart randomly
            picks a patch location within the image, and the total number of
            image patches in the sequence will be the number of input/target
            images multiplied by num_darts.
            The default is 1.
        tiled : boolean, optional
            If True, num_darts is ignored, and patches are exctracted with
            regular spacing specified by tile_strides.
            The default is False.
        tile_strides: tuple (y, x) of integers, optional
            Spacing between adjacent tiles when tiled=True. Ignored otherwise.
            The default is (1, 1).
        shuffle : boolean, optional
            If True, samples are shuffled in the beginning and every epoch.
            The default is False.
        padding : string, optional
            When the image size is smaller than patch_shape on either side,
            it is extended to patch_shape according to the specified method.
            If padding='magnify' (default), the images will be isotropically
            magnified. Otherwise, the outside of the image boundaries will be
            filled using numpy.pad(mode=padding).

        Returns
        -------
        None.

        """
        self.batch_size = batch_size
        self.model_io_shape = model_io_shape
        self.patch_shape = patch_shape
        self.norm_channel = norm_channel
        self.norm_shifts = norm_shifts
        self.num_darts = num_darts
        self.tiled = tiled
        self.shuffle = shuffle
        self.num_splits = 1
        self.split_idx = 0

        if(input_img_paths is not None):
            self.num_videos = len(input_img_paths)
            self.num_channels = len(input_img_paths[0])
            tmp = tiff.imread(input_img_paths[0][0])
        else:
            self.num_videos = len(input_imgs)
            self.num_channels = len(input_imgs[0])
            tmp = input_imgs[0][0]
        self.num_frames = tmp.shape[0]
        self.image_shape = tmp.shape[1:] # this may get modified later
        self.orig_image_shape = tmp.shape[1:] # this will stay the same
        # handle cases where input images are smaller than the patch
        ratio_y = patch_shape[0] / self.image_shape[0]
        ratio_x = patch_shape[1] / self.image_shape[1]
        if(ratio_y > 1 or ratio_x > 1): # smaller and needs padding
            if(padding == 'magnify'):
                self.needs_padding = 'magnify'
                ratio = max(ratio_y, ratio_x)
                h = math.floor(self.image_shape[0] * ratio + 0.5)
                w = math.floor(self.image_shape[1] * ratio + 0.5)
                self.image_shape = (h, w)
            else:
                self.needs_padding = 'padding'
                h = max(self.image_shape[0], patch_shape[0])
                w = max(self.image_shape[1], patch_shape[1])
                before_h = (h - self.image_shape[0]) // 2
                after_h = h - self.image_shape[0] - before_h
                before_w = (w - self.image_shape[1]) // 2
                after_w = w - self.image_shape[1] - before_w
                self.image_shape = (h, w)
                pad_width = ((0, 0), (before_h, after_h), (before_w, after_w))
        else:
            self.needs_padding = 'no'

        num_images = self.num_videos * self.num_frames
        buf_shape = (num_images,) + self.image_shape
        self.input_images = np.zeros(buf_shape + (self.num_channels,),
                                     dtype='float32')
        self.target_images = np.zeros(buf_shape, dtype='uint8')
        for i in range(self.num_videos):
            s = i * self.num_frames
            e = s + self.num_frames
            for j in range(self.num_channels):
                if(input_img_paths is not None):
                    tmp = tiff.imread(input_img_paths[i][j])
                else:
                    tmp = input_imgs[i][j]
                if(self.needs_padding == 'magnify'):
                    # mode='constant' leads to false positives near boundaries
                    tmp = resize(tmp, (self.num_frames,) + self.image_shape,
                                 mode='reflect')
                elif(self.needs_padding == 'padding'):
                    tmp = np.pad(tmp, pad_width, padding)
                self.input_images[s:e, :, :, j] = tmp
            if(target_img_paths is not None):
                self.target_images[s:e] = tiff.imread(target_img_paths[i])


        h = max(self.image_shape[0] - self.patch_shape[0], 0)
        w = max(self.image_shape[1] - self.patch_shape[1], 0)

        if(tiled): # regular tiling
            y, x = np.mgrid[0:h+1:tile_strides[0], 0:w+1:tile_strides[1]]
            self.num_darts = x.size
            # same tile positions for all the images
            self.Ys = np.tile(y.flatten(), num_images)
            self.Xs = np.tile(x.flatten(), num_images)

        else: # random dart throwing
            if(num_darts == 1): # center patch
                self.Ys = [h // 2 for i in range(num_images)]
                self.Xs = [w // 2 for i in range(num_images)]
            elif(num_darts > 1): # randomly choose patch positions
                self.Ys = np.random.randint(h+1, size=num_images * num_darts)
                self.Xs = np.random.randint(w+1, size=num_images * num_darts)

        self.sample_indices = list(range(num_images * self.num_darts))
        if(shuffle):
            random.shuffle(self.sample_indices)


    def _get_lengths(self):
        """
        Calculate the number of batches in the sequence, along with the number
        of batches in each subsequence if the sequence is split.

        If the number of samples is not divisible by the specified bach size,
        the result is rounded down so that a batch having less samples than
        the batch size will not be added to the end of the sequence for trainig
        in the case of tiled=False, whereas it is rounded up so all the samples
        are evaluated in the case of tiled=True.

        Returns
        -------
        length : integer
            The number of batches in the sequence.
        sublength : integer
            The number of batches in each subsequence. If the sequence is not
            split (self.num_splits=1), this will be the same as length.

        """
        num_samples = len(self.sample_indices)
        if(self.tiled):
            length = (num_samples + self.batch_size - 1) // self.batch_size
        else:
            length = num_samples // self.batch_size

        sublength = (length + self.num_splits - 1) // self.num_splits
        return length, sublength


    def __len__(self):
        """
        Return the number of batches in the sequence if it is not split
        (self.num_splits=1). If split (self.num_splits>1), return the number
        of batches in the current subsequence pointed to by self.split_idx.

        Returns
        -------
        Integer
            The number of batches in the (sub)sequence.

        """
        length, sublength = self._get_lengths()
        if(self.num_splits == 1):
            return length
        else:
            start = sublength * self.split_idx
            end = min(start + sublength, length)
            return end - start


    def __getitem__(self, idx):
        """
        Return the idx-th batch in the sequence.

        Parameters
        ----------
        idx : integer
            Index specifying a batch in the (possibly shuffled) sequence.

        Returns
        -------
        inputs : 4D numpy.ndarray of float32
            Input image patch data corresponding to the indexed batch.
            The shape is (batch_size, patch_height, patch_width, num_channels).
        targets : 4D numpy.ndarray of float32
            Target image patch data corresponding to the indexed batch.
            The shape is (batch_size, patch_height, patch_width, 1).

        """
        # For split sequence, move the index to the start of the subsequence
        if(self.num_splits > 1):
            _, sublength = self._get_lengths()
            idx += sublength * self.split_idx

        buf_shape = (self.batch_size,) + self.model_io_shape
        inputs = np.zeros(buf_shape + (self.num_channels,), dtype='float32')
        targets = np.zeros(buf_shape, dtype='float32')
        for i in range(self.batch_size):
            ofs = self.batch_size * idx + i
            if(ofs >= len(self.sample_indices)):
                break
            sample_idx = self.sample_indices[ofs]
            img_idx = sample_idx // self.num_darts
            ys = self.Ys[sample_idx]
            xs = self.Xs[sample_idx]
            ye = ys + self.patch_shape[0] # no greater than self.image_shape[0]
            xe = xs + self.patch_shape[1] # no greater than self.image_shape[1]
            if(self.patch_shape == self.model_io_shape):
                in_tmp = self.input_images[img_idx, ys:ye, xs:xe]
                targets[i] = self.target_images[img_idx, ys:ye, xs:xe]
            else:
                in_tmp = resize(self.input_images[img_idx, ys:ye, xs:xe],
                                self.model_io_shape, mode='edge')
                targets[i] = resize(self.target_images[img_idx, ys:ye, xs:xe],
                                    self.model_io_shape, mode='edge')
            # patch-wise normalization
            if(self.norm_channel in range(self.num_channels)):
                # use max/min of specified channel
                vmax = np.amax(in_tmp[:, :, self.norm_channel])
                vmin = np.amin(in_tmp[:, :, self.norm_channel])
            else: # use max/min of all channels
                vmax = np.amax(in_tmp)
                vmin = np.amin(in_tmp)
            norm_scale = 1 / (vmax - vmin)
            norm_shift_vals = np.zeros((1, 1, self.num_channels))
            for j in range(self.num_channels):
                if(self.norm_shifts and self.norm_shifts[j]):
                    norm_shift_vals[:, :, j] = np.amin(in_tmp[:, :, j])
            inputs[i] = (in_tmp - norm_shift_vals) * norm_scale

        targets = targets[:, :, :, np.newaxis] # add 4th dimension of size 1
        return inputs, targets


    def on_epoch_end(self):
        """
        Shuffle samples at the end of every epoch if self.shuffle=True.

        Returns
        -------
        None.

        """
        if(self.shuffle):
            random.shuffle(self.sample_indices)


    def get_tile_pos(self):
        """
        Return the top left corner positions of the tiles when patches are
        tiled (i.e., self.tiled=True).

        Returns
        -------
        Array of integer
            Y coordinates of the tiled patches.
        Array of integer
            X coordinates of the tiled patches.

        """
        if(self.tiled):
            return self.Ys[0:self.num_darts], self.Xs[0:self.num_darts]
        else:
            return None


    def output_data_size(self):
        """
        Return the total output data size necessary to hold the prediction
        results for all the samples in the sequence, assuming 32-bit float.

        Returns
        -------
        size : integer
            Total data size in bytes.

        """
        num_samples = len(self.sample_indices)
        size = num_samples * self.model_io_shape[0] * self.model_io_shape[1] * 4
        return size


    def split_samples(self, num_splits=1, split_idx=0):
        """
        Split the sequence into multiple subsequences. Useful when the sequence
        contains many samples leading to too large output prediction data for
        the available GPU memory. Passing non-default argument values changes
        the behaviors of __len__() and __getitem__() to realize the splitting.

        Parameters
        ----------
        num_splits : integer
            The sequence will be split into this many subsequences.
            The default is 1 (no splitting).
        split_idx : integer
            Only the split_idx-th subsequence will be output.
            The default is 0 (no splitting).

        Returns
        -------
        None.

        """
        self.num_splits = num_splits
        self.split_idx = split_idx
