import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Concatenate, \
    Activation, Conv2DTranspose
from tensorflow.keras import backend as keras
from data import *
from ResUnet import *
from CBAM_attention import *


class MyUnet(object):
    def __init__(self, img_rows=512, img_cols=512):
        self.img_rows = img_rows
        self.img_cols = img_cols

    def load_data(self):
        mydata = dataProcess(self.img_rows, self.img_cols)
        train, mask_train = mydata.load_train_data()
        return train, mask_train

    def get_unet(self):
        inputs = Input((self.img_rows, self.img_cols, 1))
        # DownSampling
        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
        conv1 = Dropout(0.5)(conv1)
        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
        conv2 = Dropout(0.5)(conv2)
        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
        conv2 = conv_block(conv2, 32, 32, 128, 128)
        conv2 = identity_block(conv2, 32, 32, 128)
        conv2 = identity_block(conv2, 32, 32, 128)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
        conv3 = Dropout(0.5)(conv3)
        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
        conv3 = conv_block(conv3, 64, 64, 256, 256)
        conv3 = identity_block(conv3, 64, 64, 256)
        conv3 = identity_block(conv3, 64, 64, 256)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
        conv4 = Dropout(0.5)(conv4)
        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
        conv4 = conv_block(conv4, 128, 128, 512, 512)
        conv4 = identity_block(conv4, 128, 128, 512)
        conv4 = identity_block(conv4, 128, 128, 512)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
        conv5 = Dropout(0.5)(conv5)
        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
        conv5 = conv_block(conv5, 256, 256, 1024, 1024)
        conv5 = identity_block(conv5, 256, 256, 1024)
        conv5 = identity_block(conv5, 256, 256, 1024)
        drop5 = Dropout(0.5)(conv5)

        # UpSampling
        up6 = Conv2DTranspose(512, 2, activation='relu', padding='same',
                              kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(drop5))
        up6 = Dropout(0.5)(up6)
        up6 = Conv2DTranspose(512, 2, activation='relu', padding='same',
                              kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(up6))
        merge6 = Concatenate(axis=3)([conv4, up6])
        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
        conv6 = Dropout(0.5)(conv6)
        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
        conv6 = conv_block1(conv6, 128, 128, 512, 512)
        conv6 = identity_block(conv6, 128, 128, 512)
        conv6 = identity_block(conv6, 128, 128, 512)

        up7 = Conv2DTranspose(256, 2, activation='relu', padding='same',
                              kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv6))
        up7 = Dropout(0.5)(up7)
        up7 = Conv2DTranspose(256, 2, activation='relu', padding='same',
                              kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(up7))
        merge7 = Concatenate(axis=3)([conv3, up7])
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
        conv7 = Dropout(0.5)(conv7)
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
        conv7 = conv_block1(conv7, 64, 64, 256, 256)
        conv7 = identity_block(conv7, 64, 64, 256)
        conv7 = identity_block(conv7, 64, 64, 256)

        up8 = Conv2DTranspose(128, 2, activation='relu', padding='same',
                              kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv7))
        up8 = Dropout(0.5)(up8)
        up8 = Conv2DTranspose(128, 2, activation='relu', padding='same',
                              kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(up8))
        merge8 = Concatenate(axis=3)([conv2, up8])
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
        conv8 = Dropout(0.5)(conv8)
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
        conv8 = conv_block1(conv8, 32, 32, 128, 128)
        conv8 = identity_block(conv8, 32, 32, 128)
        conv8 = identity_block(conv8, 32, 32, 128)

        up9 = Conv2DTranspose(64, 2, activation='relu', padding='same',
                              kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv8))
        up9 = Dropout(0.5)(up9)
        up9 = Conv2DTranspose(64, 2, activation='relu', padding='same',
                              kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(up9))
        conv1 = cbam_attention(conv1)
        merge9 = Concatenate(axis=3)([conv1, up9])
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
        conv9 = Dropout(0.5)(conv9)
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        conv9 = conv_block1(conv9, 16, 16, 64, 64)
        conv9 = identity_block(conv9, 16, 16, 64)
        conv9 = identity_block(conv9, 16, 16, 64)

        # Output layer
        conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

        model = Model(inputs=inputs, outputs=conv10)
        model.compile(optimizer=Adam(learning_rate=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-8),
                      loss='binary_crossentropy', metrics=['accuracy'])
        model.summary()
        return model

    def train(self):
        # Train the model
        train, mask_train = self.load_data()
        model = self.get_unet()
        model_checkpoint = ModelCheckpoint('./model/U-RNet+.hdf5', monitor='loss', verbose=1, save_best_only=True)
        history = model.fit(train, mask_train, batch_size=2, epochs=500, verbose=1, validation_split=0.2,
                            shuffle=True, callbacks=[model_checkpoint])

        # Plot accuracy and loss curves
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(len(acc))

        plt.plot(epochs, acc, 'b', label='training accuracy')
        plt.plot(epochs, val_acc, ':r', label='validation accuracy')
        plt.title('Accuracy')
        plt.savefig('./model/Accuracy.png')
        plt.legend()
        plt.figure()

        plt.plot(epochs, loss, 'b', label='training loss')
        plt.plot(epochs, val_loss, ':r', label='validation loss')
        plt.title('Loss')
        plt.savefig('./model/Loss.png')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    myunet = MyUnet()
    myunet.train()
