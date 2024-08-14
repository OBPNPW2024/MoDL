import glob
import numpy as np
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array, load_img


class DataProcess(object):

    # Initialize the DataProcess class
    def __init__(self, out_rows, out_cols,
                 data_path="../deform/train",
                 label_path="../deform/label",
                 test_path="../test",
                 npy_path="../npydata",
                 img_type="tif"
                 ):
        self.out_rows = out_rows
        self.out_cols = out_cols
        self.data_path = data_path
        self.label_path = label_path
        self.img_type = img_type
        self.test_path = test_path
        self.npy_path = npy_path

    # create_train_data
    def create_train_data(self):
        i = 0
        print('-' * 30)
        print('Creating training images...')
        print('-' * 30)
        imgs = glob.glob(self.data_path + "/*." + self.img_type)
        print(len(imgs))

        imgdatas = np.ndarray((len(imgs), self.out_rows, self.out_cols, 1), dtype=np.uint8)
        imglabels = np.ndarray((len(imgs), self.out_rows, self.out_cols, 1), dtype=np.uint8)
        for imgname in imgs:

            midname = imgname[imgname.rindex("\\") + 1:]
            img = load_img(self.data_path + "/" + midname, color_mode='grayscale')
            label = load_img(self.label_path + "/" + midname, color_mode='grayscale')
            img = img_to_array(img)
            label = img_to_array(label)


            imgdatas[i] = img
            imglabels[i] = label
            if i % 100 == 0:
                print('Done: {0}/{1} images'.format(i, len(imgs)))
            i += 1
        print('loading done')
        np.save(self.npy_path + '/imgs_train.npy', imgdatas)
        np.save(self.npy_path + '/imgs_mask_train.npy', imglabels)

    # load_train_data
    def load_train_data(self):
        train = np.load(self.npy_path + "/imgs_train.npy")
        mask_train = np.load(self.npy_path + "/imgs_mask_train.npy")
        # Convert images and masks to float32 and normalize it
        train = train.astype('float32')
        mask_train = mask_train.astype('float32')
        train /= 255
        mean = train.mean(axis=0)
        train -= mean
        mask_train /= 255
        mask_train[mask_train > 0.5] = 1
        mask_train[mask_train <= 0.5] = 0


if __name__ == "__main__":
    mydata = DataProcess(512, 512)
    mydata.create_train_data()
