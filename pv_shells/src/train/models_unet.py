# src/train/models_unet.py
from keras import layers, Model, Input

def _block(x, filters, dilation=1, dropout=0.0):
    x = layers.Conv2D(filters, 3, padding="same", dilation_rate=dilation, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    if dropout:
        x = layers.Dropout(dropout)(x)
    x = layers.Conv2D(filters, 3, padding="same", dilation_rate=dilation, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x

def unet_pv(patch_shape=(96, 512, 1), base_filters=32, depth=4, dilation_rate=1, dropout=0.0):
    inputs = Input(shape=patch_shape)
    skips = []
    x = inputs
    f = base_filters

    # Encoder
    for d in range(depth):
        x = _block(x, f, dilation=dilation_rate, dropout=dropout if d>0 else 0.0)
        skips.append(x)
        x = layers.MaxPooling2D(pool_size=2)(x)
        f *= 2

    # Bottleneck
    x = _block(x, f, dilation=dilation_rate, dropout=dropout)

    # Decoder
    for d in reversed(range(depth)):
        f //= 2
        x = layers.Conv2DTranspose(f, 2, strides=2, padding="same")(x)
        x = layers.Concatenate()([x, skips[d]])
        x = _block(x, f, dilation=1, dropout=dropout if d>0 else 0.0)

    # 1-channel mask with sigmoid
    out = layers.Conv2D(1, 1, activation="sigmoid")(x)
    return Model(inputs, out, name="unet_pv")