from keras import layers, models


def conv(inputs, num_filters):
    x = layers.Conv2D(num_filters, 3, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(num_filters, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x


def down(inputs, num_filters, dropout_rate, centered):
    c = conv(inputs, num_filters)
    if(centered):
        pool_size = 3
    else:
        pool_size = 2
    x = layers.MaxPooling2D(pool_size, strides=2, padding='same')(c)
    x = layers.Dropout(dropout_rate)(x)
    return x, c


def up(inputs, copy, num_filters, dropout_rate, method, centered):
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
    x = conv(x, num_filters)
    return x


def get_model(img_size, num_channels,
              num_stages, num_filters, dropout_rate,
              upsample_method, centered):

    inputs = layers.Input(shape=img_size + (num_channels,))

    x = inputs
    conv_list = []
    n = num_filters
    for i in range(num_stages):
        x, c = down(x, n, dropout_rate, centered)
        conv_list.append(c)
        n *= 2

    x = conv(x, n)

    for i in range(num_stages):
        n /= 2
        x = up(x, conv_list[-i-1], n, dropout_rate, upsample_method, centered)

    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(x)

    model = models.Model(inputs, outputs)
    return model
