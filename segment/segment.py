import math
import keras
import tensorflow as tf
import numpy as np
from pathlib import Path
from skimage.transform import resize

from .sequence import VI_Sequence
from .data import get_training_data
from .model import get_model, load_model
from .patch import merge_patches


class VI_Segment:

    def __init__(self):
        pass


    def _prevent_tf_from_occupying_entire_gpu_memory(self):
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)


    def _predict_and_merge(self, data_seq, tile_strides, gpu_mem_size,
                           input_paths, target_paths, out_paths, ref_paths):
        """
        Make prediction on sliding tiles/patches and merge them into single
        probability maps.

        Parameters
        ----------
        data_seq : VI_Sequence
            Sequence object used to feed data to the model.
        tile_strides : tuple (y, x) of integer
            Spacing between adjacent tiles.
        gpu_mem_size : float or None
            GPU memory size in gigabytes (GB) that can be allocated for
            buffering prediction outputs. If None, no limit is assumed.
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
        3D numpy.ndarray of float
            U-Net outputs.

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
            ret = self.model.predict(data_seq, verbose=1)
            if(i == 0):
                preds = ret
            else:
                preds = np.append(preds, ret, axis=0)
        preds = preds[:, :, :, 0] # remove last dimension (its length is one)
        if(preds.shape[1:] != data_seq.patch_shape):
            preds = resize(preds, (len(preds),) + data_seq.patch_shape,
                           mode='edge', anti_aliasing=True)

        return merge_patches(preds, data_seq, tile_strides,
                             input_paths, target_paths, out_paths, ref_paths)


    def set_training(self, input_dir_list, target_dir, seed, validation_ratio,
                     model_io_shape, norm_channel, norm_shifts):
        """
        Set parameters for training.

        Parameters
        ----------
        input_dir_list : list of pathlib.Path
            List of directory paths containing input files. Each directory path
            corresponds to one channel of the input.
        target_dir : pathlib.Path
            Directory path containing target files.
        seed : integer
            Seed for randomized splitting into traning and validation data.
        validation_ratio : integer
            What fraction of the inputs are used for validation. If there are
            N inputs, N/validation_ratio of them will be used for validation,
            while the rest will be used for training.
        model_io_shape : tuple (height, width) of integer
            The U-Net model's input/output shape. Patches of this size will be
            extracted from input/target images and used for training.
        norm_channel : integer, optional
            The channel used to determine the scale for patch normalization.
            If -1 (default), the max/min intensities of a given patch across
            all the channels will be used. If nonnegative, the max/min of the
            specified channel will be used.
        norm_shifts : list of boolean, optional
            Whether or not to shift the intensities of an image patch so the
            channel-wise minimum becomes zero. The default is [] (no shifting).

        Returns
        -------
        None.
    
        """
        self.data = get_training_data(input_dir_list, target_dir,
                                      seed, validation_ratio)
        self.model_io_shape = model_io_shape
        self.num_channels = len(input_dir_list)
        self.norm_channel = norm_channel
        self.norm_shifts = norm_shifts


    def train(self, model_dir, log_file, num_darts, batch_size, epochs):
        """
        Train the U-Net for cell segmentation.

        Parameters
        ----------
        model_dir : string or pathlib.Path
            Directory path to which the trained models will be saved.
        log_file : string or pathlib.Path
            File path to which the training log will be saved.
        num_darts : integer
            The number of darts to be thrown per image to extract patches
            from the image. If num_darts=1, one image patch is extracted from
            the center of the image. If num_darts>1, each dart randomly picks
            a patch location within the image.
        batch_size : integer
            Batch size for training.
        epochs : integer
            The number of epochs to run.

        Returns
        -------
        None.
    
        """
        train_input_paths = self.data[0]
        train_target_paths = self.data[1]
        valid_input_paths = self.data[2]
        valid_target_paths = self.data[3]

        train_seq = VI_Sequence(batch_size,
                                self.model_io_shape, self.model_io_shape,
                                train_input_paths, train_target_paths, None,
                                self.norm_channel, self.norm_shifts,
                                num_darts=num_darts, shuffle=True)

        valid_seq = VI_Sequence(batch_size,
                                self.model_io_shape, self.model_io_shape,
                                valid_input_paths, valid_target_paths, None,
                                self.norm_channel, self.norm_shifts,
                                num_darts=num_darts)

        keras.backend.clear_session()

        model = get_model(self.model_io_shape, self.num_channels,
                          3, 32, 0.5, 'conv_transpose', False)
        model.summary()
        opt = keras.optimizers.RMSprop()
        model.compile(optimizer=opt, loss='binary_crossentropy')    
        model_file = model_dir.joinpath('model_e{epoch:02d}_v{val_loss:.4f}.h5')
        callbacks = [
            keras.callbacks.ModelCheckpoint(model_file,
                                            monitor='val_loss',
                                            verbose=1,
                                            save_weigts_only=False,
                                            mode='min'),
            keras.callbacks.CSVLogger(log_file)
        ]

        model.fit(train_seq, validation_data=valid_seq,
                  epochs=epochs, callbacks=callbacks)

        keras.backend.clear_session()


    def validate(self, model_dir, out_dir, ref_dir,
                 tile_strides, batch_size, gpu_mem_size=None):
        """
        Validate a learned U-Net by making inference on a validation data set.
    
        This is different from validation during training, which is performed
        on a patch-by-patch basis for randomly chosen patches from a validation
        set. In contrast, this function performs inference on sliding tiles and
        merge them into probability maps having the same size as input images,
        which is the same process as performing inference on test/real data as
        implemented in predict(), making the results more suitable for visual
        inspection.

        Parameters
        ----------
        model_dir : string or pathlib.Path
            Directory path containing learned models. The model with the
            smallest validation loss will be used.
        out_dir : pathlib.Path
            Directory path to which U-Net outputs will be saved.
        ref_dir : pathlib.Path
            Directory path to which U-Net inputs, outputs, and targets (ground
            truth) are juxtaposed and saved for visual inspection.
        tile_strides : tuple (y, x) of integer
            Spacing between adjacent tiles.
        batch_size : integer
            Batch size for inference.
        gpu_mem_size : float or None
            GPU memory size in gigabytes (GB) that can be allocated for
            buffering prediction outputs. The default is None, in which case
            no limit is assumed.

        Returns
        -------
        None.

        """
        valid_input_paths = self.data[2]
        valid_target_paths = self.data[3]

        keras.backend.clear_session()

        # sort model file names by validation loss
        model_files = sorted(Path(model_dir).glob('model*.h5'),
                             key=lambda x: float(x.stem.split('v')[-1]))
        # first item of the sorted list has the lowest validation loss
        self.model, model_io_shape = load_model(model_files[0])
        assert(model_io_shape == self.model_io_shape)

        valid_seq = VI_Sequence(batch_size,
                                self.model_io_shape, self.model_io_shape,
                                valid_input_paths, valid_target_paths, None,
                                self.norm_channel, self.norm_shifts,
                                tiled=True, tile_strides=tile_strides)

        out_paths = [out_dir.joinpath(p.name) for p in valid_target_paths]
        ref_paths = [ref_dir.joinpath(p.name) for p in valid_target_paths]

        self._predict_and_merge(valid_seq, tile_strides, gpu_mem_size,
                                valid_input_paths, valid_target_paths,
                                out_paths, ref_paths)

        keras.backend.clear_session()


    def set_inference(self, model_file):
        """
        Set parameters for inference.

        Parameters
        ----------
        model_file : string or pathlib.Path
            File path containing a learned model.
        """
        self._prevent_tf_from_occupying_entire_gpu_memory()    
        keras.backend.clear_session()
        self.model, self.model_io_shape = load_model(model_file)


    def predict(self, input_files, out_file, ref_file,
                norm_channel, norm_shifts,
                tile_shape, tile_strides, batch_size, gpu_mem_size=None):
        """
        Make predictions on a test/real data set.

        Parameters
        ----------
        input_files : list of pathlib.Path
            List of input file paths. Each corresponds to one channel of the input.
        out_file : pathlib.Path
            File path to which the U-Net output will be saved.
        ref_file : pathlib.Path
            File path to which the U-Net input and output are juxtaposed
            and saved for visual inspection.
        norm_channel : integer, optional
            The channel used to determine the scale for patch normalization.
            If -1 (default), the max/min intensities of a given patch across
            all the channels will be used. If nonnegative, the max/min of the
            specified channel will be used.
        norm_shifts : list of boolean, optional
            Whether or not to shift the intensities of an image patch so the
            channel-wise minimum becomes zero. The default is [] (no shifting).
        tile_shape : tuple (height, width) of integer
            Size of tiles to be extracted from images.
        tile_strides : tuple (y, x) of integer
            Spacing between adjacent tiles.
        batch_size : integer
            Batch size for inference.
        gpu_mem_size : float or None
            GPU memory size in gigabytes (GB) that can be allocated for
            buffering prediction outputs. The default is None, in which case
            no limit is assumed.

        Returns
        -------
        None.

        """
        data_seq = VI_Sequence(batch_size, self.model_io_shape, tile_shape,
                               [input_files], None, None,
                               norm_channel, norm_shifts,
                               tiled=True, tile_strides=tile_strides)

        self._predict_and_merge(data_seq, tile_strides, gpu_mem_size,
                                [input_files], None, [out_file], [ref_file])

        keras.backend.clear_session()


    def predict_online(self, input_images, norm_channel, norm_shifts,
                       tile_shape, tile_strides, batch_size, gpu_mem_size=None):

        data_seq = VI_Sequence(batch_size, self.model_io_shape, tile_shape,
                               None, None, [input_images],
                               norm_channel, norm_shifts,
                               tiled=True, tile_strides=tile_strides)

        return self._predict_and_merge(data_seq, tile_strides, gpu_mem_size,
                                       None, None, None, None)
