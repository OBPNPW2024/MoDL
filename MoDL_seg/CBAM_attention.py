

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model


# Channel attention mechanism
def channel_attention(inputs, ratio=0.25):
    channel = inputs.shape[-1]
    x_max = layers.GlobalMaxPooling2D()(inputs)
    x_avg = layers.GlobalAveragePooling2D()(inputs)
    x_max = layers.Reshape([1, 1, -1])(x_max)
    x_avg = layers.Reshape([1, 1, -1])(x_avg)
    x_max = layers.Dense(channel * ratio)(x_max)
    x_avg = layers.Dense(channel * ratio)(x_avg)
    x_max = layers.Activation('relu')(x_max)
    x_avg = layers.Activation('relu')(x_avg)
    x_max = layers.Dense(channel)(x_max)
    x_avg = layers.Dense(channel)(x_avg)
    x = layers.Add()([x_max, x_avg])
    x = tf.nn.sigmoid(x)
    x = layers.Multiply()([inputs, x])
    return x


# Spatial attention mechanism
def spatial_attention(inputs):
    x_max = tf.reduce_max(inputs, axis=3, keepdims=True)
    x_avg = tf.reduce_mean(inputs, axis=3, keepdims=True)
    x = layers.concatenate([x_max, x_avg])
    x = layers.Conv2D(filters=1, kernel_size=(1, 1), strides=1, padding='same')(x)
    x = tf.nn.sigmoid(x)
    x = layers.Multiply()([inputs, x])
    return x


# CBAM_attention implement
def cbam_attention(inputs):
    x = channel_attention(inputs)
    x = spatial_attention(x)
    return x
