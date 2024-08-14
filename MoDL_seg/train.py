import numpy as np
from tensorflow.keras.models import *
from tensorflow.python.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Concatenate, Activation, Conv2DTranspose, Add
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras import backend as keras
from data_load import *
import matplotlib.pyplot as plt
import datetime
from PIL import Image
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'




class myUnet(object):
   def __init__(self, img_rows = 512, img_cols = 512):
      self.img_rows = img_rows
      self.img_cols = img_cols

   def load_data(self):
      mydata = DataProcess(self.img_rows, self.img_cols)
      imgs_train, imgs_mask_train = mydata.load_train_data()
      return imgs_train, imgs_mask_train

   def get_unet(self):
      inputs = Input((self.img_rows, self.img_cols,1))



      conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
      print("conv1 shape:", conv1.shape)
      conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
      print ("conv1 shape:",conv1.shape)
      pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
      print ("pool1 shape:",pool1.shape)


      conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
      print("conv2 shape:", conv2.shape)
      conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
      print ("conv2 shape:",conv2.shape)
      pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
      print ("pool2 shape:",pool2.shape)


      conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
      print("conv3 shape:", conv3.shape)
      conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
      print ("conv3 shape:",conv3.shape)
      pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
      print ("pool3 shape:",pool3.shape)


      conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
      print("conv4 shape:", conv4.shape)
      conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
      print("conv4 shape:", conv4.shape)
      drop4 = Dropout(0.5)(conv4)
      pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
      print("pool4 shape:", pool4.shape)

      conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
      print("conv5 shape:", conv5.shape)
      conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
      print("conv5 shape:", conv5.shape)
      drop5 = Dropout(0.5)(conv5)


      up6 = Conv2DTranspose(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
      print("up6 shape:", up6.shape)
      merge6 = Concatenate(axis=3)([drop4, up6])
      print("merge6 shape:", merge6.shape)
      conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
      print("conv6 shape:", conv6.shape)
      conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
      print("conv6 shape:", conv6.shape)

      up7 = Conv2DTranspose(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
      print("up7 shape:", up7.shape)
      merge7 = Concatenate(axis=3)([conv3, up7])
      print("merge7 shape:", merge7.shape)
      conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
      print("conv7 shape:", conv7.shape)
      conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
      print("conv7 shape:", conv7.shape)


      up8 = Conv2DTranspose(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
      print("up8 shape:", up8.shape)
      merge8 = Concatenate(axis=3)([conv2, up8])
      print("merge8 shape:", merge8.shape)
      conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
      print("conv8 shape:", conv8.shape)
      conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
      print("conv8 shape:", conv8.shape)


      up9 = Conv2DTranspose(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
      print("up9 shape:", up9.shape)
      merge9 = Concatenate(axis=3)([conv1, up9])
      print("merge9 shape:", merge9.shape)
      conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
      print("conv9 shape:", conv9.shape)
      conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
      print("conv9 shape:", conv9.shape)
      conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
      print("conv9 shape:", conv9.shape)
      conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)
      print("conv10 shape:", conv10.shape)

      model = Model(inputs = inputs, outputs = conv10)
      model.compile(optimizer = Adam(learning_rate=1e-5), loss = 'binary_crossentropy', metrics = ['accuracy'])
      return model


   def train(self):
      print("loading data")
      
      imgs_train = np.load("../npydata/imgs_train.npy")
      imgs_mask_train = np.load("../npydata/imgs_mask_train.npy")
      imgs_train = imgs_train.astype('float32')
      imgs_mask_train = imgs_mask_train.astype('float32')
      imgs_train /= 255
      mean = imgs_train.mean(axis=0)
      imgs_train -= mean
      imgs_mask_train /= 255
      imgs_mask_train[imgs_mask_train > 0.5] = 1
      imgs_mask_train[imgs_mask_train <= 0.5] = 0

      print("loading data done")
      model = self.get_unet()
      print("got unet")
      model_checkpoint = ModelCheckpoint('../model/U-RNet+.hdf5', monitor='loss',verbose=1, save_best_only=True)


      starttrain = datetime.datetime.now()
      print('Fitting model...')
      history = model.fit(imgs_train, imgs_mask_train, batch_size=2, epochs=50, verbose=1, validation_split=0.2,
                     shuffle=True, callbacks=[model_checkpoint])
      endtrain = datetime.datetime.now()
      print('train time: %s seconds' % (endtrain - starttrain))


      acc = history.history['accuracy']
      acc0 = np.array(acc)
      val_acc = history.history['val_accuracy']
      loss = history.history['loss']
      loss0 = np.array(loss)
      val_loss = history.history['val_loss']
      epochs = range(len(acc))



      plt.plot(epochs, acc, 'b', label='training accuracy')
      plt.plot(epochs, val_acc, ':r', label='validation accuracy')
      plt.title('Accuracy')
      plt.savefig('../model/Accuracy.png')
      plt.legend()
      plt.figure()


      plt.plot(epochs, loss, 'b', label='training loss')
      plt.plot(epochs, val_loss, ':r', label='validation loss')
      plt.title('Loss')
      plt.savefig('../model/Loss.png')
      plt.legend()
      plt.show()
      with open('../model/unet.txt', 'wt') as ft:
         ft.write('loss: %.6s ' % (loss0[-1]))
         ft.write('\n')
         ft.write('accuracy: %.6s ' % (acc0[-1]))
         ft.write('\n')



if __name__ == '__main__':

   myunet = myUnet()
   myunet.train()
