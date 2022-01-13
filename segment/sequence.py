import random
import math
import numpy as np
import tifffile as tiff
from skimage.transform import resize
from keras.utils import Sequence


class VI_Sequence(Sequence):

    def __init__(self, batch_size, patch_shape,
                 input_img_paths, target_img_paths,
                 num_darts=1, tiled=False, tile_strides=(1, 1),
                 shuffle=False):
        """
        Initializes VI_Sequence (Voltage Imaging) instance.

        Parameters
        ----------
        batch_size : integer
            Batch size for training/inference.
        patch_shape : tuple (height, width) of integers
            Image patch size to be extracted from image files.
            The size should be equal to (when num_darts=1) or smaller than
            (when num_darts>1) the input images for training. For inference,
            the input images may be smaller than the patch size, in which case
            the input images will be magnified.
        input_img_paths : list of list of pathlib.Path
            List of file paths to input images to be fed into the U-Net.
            Each element of the list is a list of file paths corresponding
            to multiple channels.
            Each file (tiff) can contain multiple images. The number of images
            per file and the image size must be the same for all the files.
        target_img_paths : list of pathlib.Path
            List of file paths to target images specifing expected outputs
            from the U-Net. Each file (tiff) must contain the same number of
            images of the same size as the corresponding input file.
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

        Returns
        -------
        None.

        """
        self.batch_size = batch_size
        self.patch_shape = patch_shape
        self.num_darts = num_darts
        self.tiled = tiled
        self.shuffle = shuffle
        self.num_splits = 1
        self.split_idx = 0

        self.num_channels = len(input_img_paths[0])
        tmp = tiff.imread(input_img_paths[0][0])
        self.num_frames = tmp.shape[0]
        self.image_shape = tmp.shape[1:]
        # handle cases where input images are smaller than the patch
        ratio_y = patch_shape[0] / self.image_shape[0]
        ratio_x = patch_shape[1] / self.image_shape[1]
        if(ratio_y > 1 or ratio_x > 1):
            self.needs_resizing = True
            ratio = max(ratio_y, ratio_x)
            h = math.floor(self.image_shape[0] * ratio + 0.5)
            w = math.floor(self.image_shape[1] * ratio + 0.5)
            self.image_shape = (h, w)
        else:
            self.needs_resizing = False

        num_images = len(input_img_paths) * self.num_frames
        buf_shape = (num_images,) + self.image_shape
        self.input_images = np.zeros(buf_shape + (self.num_channels,),
                                     dtype='float32')
        self.target_images = np.zeros(buf_shape, dtype='uint8')
        for i, paths in enumerate(input_img_paths):
            s = i * self.num_frames
            e = s + self.num_frames
            for j, path in enumerate(paths):
                tmp = tiff.imread(path)
                if(self.needs_resizing):
                    tmp = resize(tmp, (self.num_frames,) + self.image_shape)
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
            The shape is batch_size x patch_height x patch_width x num_channels.
        targets : 4D numpy.ndarray of float32
            Target image patch data corresponding to the indexed batch.
            The shape is batch_size x patch_height x patch_width x 1.

        """

        # For split sequence, move the index to the start of the subsequence
        if(self.num_splits > 1):
            _, sublength = self._get_lengths()
            idx += sublength * self.split_idx

        buf_shape = (self.batch_size,) + self.patch_shape
        inputs = np.zeros(buf_shape + (self.num_channels,), dtype='float32')
        targets = np.zeros(buf_shape + (1,), dtype='float32')
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
            inputs[i] = self.input_images[img_idx, ys:ye, xs:xe]
            targets[i, :, :, 0] = self.target_images[img_idx, ys:ye, xs:xe]

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
        size = num_samples * self.patch_shape[0] * self.patch_shape[1] * 4
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
