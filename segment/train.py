import keras

from .data import get_training_data
from .sequence import VI_Sequence
from .model import get_model


def train_model(input_dir_list, target_dir, model_dir,
                seed, validation_ratio,
                patch_shape, num_darts, batch_size, epochs):
    """
    Train the U-Net for cell segmentation.

    Parameters
    ----------
    input_dir_list : list of pathlib.Path
        List of directory paths containing input files. Each directory path
        corresponds to one channel of the input.
    target_dir : pathlib.Path
        Directory path containing target files.
    model_dir : pathlib.Path
        Directory path in which the trained model will be saved.
    seed : integer
        Seed for randomized splitting into traning and validation data.
    validation_ratio : integer
        What fraction of the inputs are used for validation. If there are
        N inputs, N/validation_ratio of them will be used for validation,
        while the rest will be used for training.
    patch_shape : tuple (height, width) of integer
        Size of patches to be extracted from images. Training will be
        performed on a patch-wise manner.
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
    
    data = get_training_data(input_dir_list, target_dir,
                             seed, validation_ratio)
    train_input_paths = data[0]
    train_target_paths = data[1]
    valid_input_paths = data[2]
    valid_target_paths = data[3]

    train_seq = VI_Sequence(batch_size, patch_shape,
                            train_input_paths, train_target_paths,
                            num_darts=num_darts, shuffle=True)
    
    valid_seq = VI_Sequence(batch_size, patch_shape,
                            valid_input_paths, valid_target_paths,
                            num_darts=num_darts)
        
    # Free up RAM in case the model definition has been executed multiple times
    keras.backend.clear_session()
    num_channels = len(train_input_paths[0])
    model = get_model(patch_shape, num_channels)
    model.summary()    
    model.compile(optimizer="rmsprop", loss='binary_crossentropy')
    
    callbacks = [
        keras.callbacks.ModelCheckpoint(model_dir.joinpath('weight.h5'),
                                        save_best_only=True)
    ]
    
    model.fit(train_seq, validation_data=valid_seq,
              epochs=epochs, callbacks=callbacks)
    
    model.save(model_dir.joinpath('model.h5'))
