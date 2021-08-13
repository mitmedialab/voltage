import keras

from .data import get_training_data
from .sequence import VI_Sequence
from .model import get_model


def train_model(input_dir_list, target_dir, model_dir,
                seed, validation_ratio,
                patch_shape, num_darts, batch_size, epochs):
    
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
        keras.callbacks.ModelCheckpoint(model_dir + '/weight.h5',
                                        save_best_only=True)
    ]
    
    model.fit(train_seq, validation_data=valid_seq,
              epochs=epochs, callbacks=callbacks)
    
    model.save(model_dir + '/model.h5')
