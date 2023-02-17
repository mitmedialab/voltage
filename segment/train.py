import keras

from .data import get_training_data
from .sequence import VI_Sequence
from .model import get_model
from .loss import weighted_bce, dice_loss, bce_dice_loss, iou_loss


def train_model(input_dir_list, target_dir, model_dir, log_file,
                seed, validation_ratio,
                norm_channel, norm_shifts,
                model_io_shape, num_darts, batch_size, epochs):
    """
    Train the U-Net for cell segmentation.

    Parameters
    ----------
    input_dir_list : list of pathlib.Path
        List of directory paths containing input files. Each directory path
        corresponds to one channel of the input.
    target_dir : pathlib.Path
        Directory path containing target files.
    model_dir : string or pathlib.Path
        Directory path to which the trained models will be saved.
    log_file : string or pathlib.Path
        File path to which the training log will be saved.
    seed : integer
        Seed for randomized splitting into traning and validation data.
    validation_ratio : integer
        What fraction of the inputs are used for validation. If there are
        N inputs, N/validation_ratio of them will be used for validation,
        while the rest will be used for training.
    model_io_shape : tuple (height, width) of integer
        The U-Net model's input/output shape. Patches of this size will be
        extracted from input/target images and used for training.
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

    train_seq = VI_Sequence(batch_size, model_io_shape, model_io_shape,
                            train_input_paths, train_target_paths,
                            norm_channel, norm_shifts,
                            num_darts=num_darts, shuffle=True)

    valid_seq = VI_Sequence(batch_size, model_io_shape, model_io_shape,
                            valid_input_paths, valid_target_paths,
                            norm_channel, norm_shifts,
                            num_darts=num_darts)

    # Free up RAM in case the model definition has been executed multiple times
    keras.backend.clear_session()
    num_channels = len(train_input_paths[0])

    model = get_model(model_io_shape, num_channels, 3, 32, 0.5,
                      'conv_transpose', False)
    model.summary()
    #opt = keras.optimizers.Adadelta() # slow with default values
    #opt = keras.optimizers.Adam() # good
    #opt = keras.optimizers.Adamax() # so-so
    #opt = keras.optimizers.Nadam() # so-so
    opt = keras.optimizers.RMSprop() # good
    #opt = keras.optimizers.RMSprop(learning_rate=0.01)
    model.compile(optimizer=opt, loss='binary_crossentropy')
    #model.compile(optimizer=opt, loss=weighted_bce)
    #model.compile(optimizer=opt, loss=dice_loss)
    #model.compile(optimizer=opt, loss=bce_dice_loss)
    #model.compile(optimizer=opt, loss=iou_loss)

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
