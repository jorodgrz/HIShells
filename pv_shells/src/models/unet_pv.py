# src/models/unet_pv.py
from __future__ import annotations
import tensorflow as tf
from tensorflow.keras import layers as L, Model

def conv_block(x, f, dilation=1, dropout=0.0):
    x = L.Conv2D(f, 3, padding="same", dilation_rate=dilation, use_bias=False)(x)
    x = L.BatchNormalization()(x)
    x = L.Activation("relu")(x)
    if dropout and dropout > 0:
        x = L.SpatialDropout2D(dropout)(x)
    x = L.Conv2D(f, 3, padding="same", dilation_rate=dilation, use_bias=False)(x)
    x = L.BatchNormalization()(x)
    x = L.Activation("relu")(x)
    return x

def build_unet(input_shape, base_filters=32, depth=4, dilation_rate=1, dropout=0.0):
    inputs = L.Input(shape=input_shape)  # (V, S, 1)
    skips = []
    x = inputs
    f = base_filters

    # encoder
    for d in range(depth):
        x = conv_block(x, f, dilation=(dilation_rate if d>0 else 1), dropout=dropout if d>0 else 0.0)
        skips.append(x)
        x = L.MaxPooling2D(pool_size=2)(x)
        f *= 2

    # bottleneck
    x = conv_block(x, f, dilation=dilation_rate, dropout=dropout)

    # decoder
    for d in reversed(range(depth)):
        f //= 2
        x = L.UpSampling2D(size=2, interpolation="bilinear")(x)
        x = L.Concatenate()([x, skips[d]])
        x = conv_block(x, f, dilation=1, dropout=dropout if d>0 else 0.0)

    outputs = L.Conv2D(1, 1, activation="sigmoid")(x)
    return Model(inputs, outputs, name="unet_pv")

if __name__ == "__main__":
    m = build_unet((96, 512, 1))
    m.summary()