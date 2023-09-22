import math
import time
import copy
import tensorflow as tf
from tensorflow import keras
import numpy as np
from pathlib import Path
from threading import Thread
from skimage.transform import resize

from .sequence import VI_Sequence
from .data import get_training_data
from .model import get_model
from .patch import merge_patches



def _predict_sub(model, data_seq, gpu_mem_size,
                 num_gpus, gpu_id=None, gpu_index=0):
    """
    Make prediction on sliding tiles/patches.

    Parameters
    ----------
    model : tensorflow.keras.Model
        U-Net model.
    data_seq : VI_Sequence
        Sequence object used to feed data to the model.
    gpu_mem_size : float or None
        GPU memory size in gigabytes (GB) that can be allocated for
        buffering prediction outputs. If None, no limit is assumed.
    num_gpus : int
        The number of available GPUs.
    gpu_id : tensorflow.config.PhysicalDevice or None, optional
        ID of the GPU for which this function is invoked. The default is None.
    gpu_index : int, optional
        Index of the GPU for which this function is invoked.
        The range is [0, num_gpus). The default is 0.

    Returns
    -------
    preds : 3D numpy.ndarray of float
        U-Net outputs in patches.

    """
    # model.predict() allocates a buffer on the GPU to hold all the output
    # prediction data before passing it back to the main memory, which can
    # cause GPU out-of-memory error if data_seq feeds too many samples.
    # When this happens, it cannot be avoided by reducing the batch size, as
    # the number of batches increases and the output data size stays the same.
    # To avoid the error, we can split the data_seq sequence into multiple
    # subsequences and run predict() for each of them, after which we join
    # the outputs together on the main memory.
    if(num_gpus == 0):
        num_splits = 1 # no splitting by assuming we have enough CPU RAM
    elif(gpu_mem_size is None):
        num_splits = 1 # no splitting if there is no GPU memory limit
    else:
        data_size = data_seq.output_data_size() / num_gpus / 1e9 # in gigabytes
        num_splits = math.ceil(data_size / gpu_mem_size)
        if(num_splits > 1):
            print('splitting %.1f GB output data into %d parts on %s'
                  % (data_size, num_splits, gpu_id.name))

    tic = time.perf_counter()
    for i in range(num_splits):
        data_seq.split_samples(num_splits * max(num_gpus, 1),
                               num_splits * gpu_index + i)
        ret = model.predict(data_seq, verbose=1)
        if(i == 0):
            preds = ret
        else:
            preds = np.append(preds, ret, axis=0)
    preds = preds[:, :, :, 0] # remove last dimension (its length is one)
    toc = time.perf_counter()
    print('U-Net prediction: %.1f sec' % (toc - tic))

    if(preds.shape[1:] != data_seq.patch_shape):
        preds = resize(preds, (len(preds),) + data_seq.patch_shape,
                       mode='edge', anti_aliasing=True)

    return preds


def _predict_multi(model, data_seq, gpu_mem_size,
                   num_gpus, gpu_id, gpu_index, results):
    """
    Make prediction on sliding tiles/patches using one of multiple GPUs.

    Parameters
    ----------
    See _predict_sub().

    results : list of num_gpus elements
        The prediction result will be stored in results[gpu_index].

    Returns
    -------
    None.

    """
    with tf.device(gpu_id.name.replace('physical_', '')):
        # data_seq needs to be shallow-copied as _predict_sub() will modify it
        # differently via split_samples() depending on gpu_index
        results[gpu_index] = _predict_sub(model, copy.copy(data_seq),
                                          gpu_mem_size,
                                          num_gpus, gpu_id, gpu_index)


def _prerun_predict(model):
    """
    Run dummy prediction to prime the model, which seems to allow faster
    launch of subsequent prediction.

    Parameters
    ----------
    model : tensorflow.keras.Model
        U-Net model.

    Returns
    -------
    None.

    """
    _, h, w, num_channels = model.input_shape # first element is batch size
    tmp = np.random.random((num_channels, 1, h, w)) # single frame
    seq = VI_Sequence(1, (h, w), (h, w), None, None, [tmp])
    model.predict(seq, verbose=0)


class VI_Segment:

    def __init__(self):
        self.gpus = tf.config.list_physical_devices('GPU')


    def _prevent_tf_from_occupying_entire_gpu_memory(self):
        for gpu in self.gpus:
            tf.config.experimental.set_memory_growth(gpu, True)




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
        model = keras.models.load_model(model_files[0])
        assert(model.input_shape[1:3] == self.model_io_shape)

        valid_seq = VI_Sequence(batch_size,
                                self.model_io_shape, self.model_io_shape,
                                valid_input_paths, valid_target_paths, None,
                                self.norm_channel, self.norm_shifts,
                                tiled=True, tile_strides=tile_strides)

        # For simplicity, use single GPU even if more are available
        patches = _predict_sub(model, valid_seq, gpu_mem_size,
                               min(len(self.gpus), 1))

        keras.backend.clear_session()

        out_paths = [out_dir.joinpath(p.name) for p in valid_target_paths]
        ref_paths = [ref_dir.joinpath(p.name) for p in valid_target_paths]
        merge_patches(patches, valid_seq, tile_strides,
                      valid_input_paths, valid_target_paths,
                      out_paths, ref_paths)


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
        if(len(self.gpus) < 2): # CPU or single GPU
            self.model = keras.models.load_model(model_file)
            _prerun_predict(self.model)
            self.model_io_shape = self.model.input_shape[1:3]
        else:
            self.models = []
            for gpu in self.gpus:
                with tf.device(gpu.name.replace('physical_', '')):
                    model = keras.models.load_model(model_file)
                    self.models.append(model)
                    _prerun_predict(model)
            self.model_io_shape = self.models[0].input_shape[1:3]


    def predict(self, input_files, out_file, ref_file,
                norm_channel, norm_shifts,
                tile_shape, tile_strides, tile_margin,
                batch_size, gpu_mem_size=None):
        """
        Make offline predictions on a test/real data set stored in files.

        Parameters
        ----------
        input_files : list of pathlib.Path
            List of input file paths. Each corresponds to one channel of the input.
        out_file : pathlib.Path or None
            File path to which the U-Net output will be saved. If None, it
            will not be saved.
        ref_file : pathlib.Path or None
            File path to which the U-Net input and output are juxtaposed
            and saved for visual inspection. If None, they will not be saved.
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
        tile_margin : tuple (y, x) of float
            Margin that tiles will leave on the image border, specified as
            ratios to the image height and width.
        batch_size : integer
            Batch size for inference.
        gpu_mem_size : float or None
            GPU memory size in gigabytes (GB) that can be allocated for
            buffering prediction outputs. The default is None, in which case
            no limit is assumed.

        Returns
        -------
        3D numpy.ndarray of float
            U-Net output representing probability maps of firing neurons.
            The shape is (number_of_frames, image_height, image_width).

        """
        data_seq = VI_Sequence(batch_size, self.model_io_shape, tile_shape,
                               [input_files], None, None,
                               norm_channel, norm_shifts,
                               tiled=True, tile_strides=tile_strides,
                               tile_margin=tile_margin)

        num_gpus = len(self.gpus)
        if(num_gpus < 2): # CPU or single GPU
            patches = _predict_sub(self.model, data_seq, gpu_mem_size, num_gpus)
        else:
            # tensorflow.keras.model is not picklable
            # need to use thread rather than multiprocessing
            threads = [None] * num_gpus
            results = [None] * num_gpus
            for i in range(num_gpus):
                threads[i] = Thread(target=_predict_multi,
                                    args=(self.models[i], data_seq, gpu_mem_size,
                                          num_gpus, self.gpus[i], i, results))
                threads[i].start()

            for t in threads:
                t.join()

            patches = np.concatenate(results, axis=0)

        keras.backend.clear_session()

        out_files = None if(out_file is None) else [out_file]
        ref_files = None if(ref_file is None) else [ref_file]
        return merge_patches(patches, data_seq, tile_strides,
                             [input_files], None, out_files, ref_files)


    def predict_online(self, input_images, norm_channel, norm_shifts,
                       tile_shape, tile_strides, batch_size, gpu_mem_size=None):
        """
        Make online predictions on a test/real data set on memory.

        Parameters
        ----------
        input_images : list of 3D numpy.ndarray of float
            List of input images. Each corresponds to one channel of the input.

        Refer to predict() for the definitions of other parameters.

        Returns
        -------
        3D numpy.ndarray of float
            U-Net output.

        """
        data_seq = VI_Sequence(batch_size, self.model_io_shape, tile_shape,
                               None, None, [input_images],
                               norm_channel, norm_shifts,
                               tiled=True, tile_strides=tile_strides)
        patches = _predict_sub(self.models[0], data_seq, gpu_mem_size, 1)
        return merge_patches(patches, data_seq, tile_strides,
                             None, None, None, None)
