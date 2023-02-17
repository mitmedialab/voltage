from keras import layers, models
from .loss import weighted_bce, dice_loss, bce_dice_loss, iou_loss


def _conv(inputs, num_filters):
    x = layers.Conv2D(num_filters, 3, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(num_filters, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x


def _down(inputs, num_filters, dropout_rate, centered):
    c = _conv(inputs, num_filters)
    if(centered):
        pool_size = 3
    else:
        pool_size = 2
    x = layers.MaxPooling2D(pool_size, strides=2, padding='same')(c)
    x = layers.Dropout(dropout_rate)(x)
    return x, c


def _up(inputs, copy, num_filters, dropout_rate, method, centered):
    if(method == 'conv_transpose'):
        if(centered):
            kernel_size = 3
        else:
            kernel_size = 2
        x = layers.Conv2DTranspose(num_filters, kernel_size,
                                   strides=2, padding='same')(inputs)
    else:
        x = layers.UpSampling2D(2)(inputs)

    x = layers.concatenate([copy, x], axis=3)
    x = layers.Dropout(dropout_rate)(x)
    x = _conv(x, num_filters)
    return x


def get_model(img_size, num_channels,
              num_stages, num_filters, dropout_rate,
              upsample_method, centered):

    inputs = layers.Input(shape=img_size + (num_channels,))

    x = inputs
    conv_list = []
    n = num_filters
    for i in range(num_stages):
        x, c = _down(x, n, dropout_rate, centered)
        conv_list.append(c)
        n *= 2

    x = _conv(x, n)

    for i in range(num_stages):
        n /= 2
        x = _up(x, conv_list[-i-1], n, dropout_rate, upsample_method, centered)

    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(x)

    model = models.Model(inputs, outputs)
    return model


def load_model(model_file):
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
    loss_dict = {'weighted_bce': weighted_bce,
                 'dice_loss': dice_loss,
                 'bce_dice_loss': bce_dice_loss,
                 'iou_loss': iou_loss}
    model = models.load_model(model_file, custom_objects=loss_dict)
    # model.input_shape = (None, height, width, num_channels), where the first
    # element represents the batch size, which is undefined at this point
    io_shape = model.input_shape[1:3]
    return model, io_shape
