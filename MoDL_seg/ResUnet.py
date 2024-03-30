from tensorflow.keras.activations import relu
from tensorflow.keras.layers import Input, Conv2D, Activation, MaxPooling2D, Add
import tensorflow as tf


# ResUnet-Conv_block-DownSampling
def conv_block(input_x, kn1, kn2, kn3, side_kn):
    # Main pathway
    x = Conv2D(filters=kn1, kernel_size=(1, 1))(input_x)
    x = Activation(relu)(x)
    x = Conv2D(filters=kn2, kernel_size=(3, 3), padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Activation(relu)(x)
    x = Conv2D(filters=kn3, kernel_size=(1, 1))(x)
    x = Activation(relu)(x)
    # Residual connection
    y = Conv2D(filters=side_kn, kernel_size=(1, 1))(input_x)
    y = MaxPooling2D(pool_size=(2, 2))(y)
    y = Activation(relu)(y)

    output = tf.keras.layers.add([x, y])
    output = Activation(relu)(output)
    return output


# ResUnet-Conv_block-UpSampling
def conv_block1(input_x, kn1, kn2, kn3, side_kn):
    # Main pathway
    x = Conv2D(filters=kn1, kernel_size=(1, 1))(input_x)
    x = Activation(relu)(x)
    x = Conv2D(filters=kn2, kernel_size=(3, 3), padding='same')(x)
    x = Activation(relu)(x)
    x = Conv2D(filters=kn3, kernel_size=(1, 1))(x)
    x = Activation(relu)(x)
    # Residual connection
    y = Conv2D(filters=side_kn, kernel_size=(1, 1))(input_x)
    y = Activation(relu)(y)

    output = tf.keras.layers.add([x, y])
    output = Activation(relu)(output)
    return output


# ResUnet-Identity_block
def identity_block(input_x, kn1, kn2, kn3):
    # Main pathway
    x = Conv2D(filters=kn1, kernel_size=(1, 1))(input_x)
    x = Activation(relu)(x)
    x = Conv2D(filters=kn2, kernel_size=(3, 3), padding='same')(x)
    x = Activation(relu)(x)
    x = Conv2D(filters=kn3, kernel_size=(1, 1))(x)
    x = Activation(relu)(x)

    output = tf.keras.layers.add([x, input_x])
    output = Activation(relu)(output)
    return output
