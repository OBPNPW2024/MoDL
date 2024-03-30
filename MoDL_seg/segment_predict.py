import cv2
import numpy as np
import os
from glob import glob
from pathlib import Path
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img


def clear_files(folders):
    for root, dirs, files in os.walk(folders):
        for file in files:
            file_path = os.path.join(root, file)
            os.remove(file_path)


# We initially cropped 1 original stack (2048 × 2048 pixel2 resolution) into 16 patches (512 × 512 pixel2 resolution)
# in three ways, later reassembling these segmentation patches to reconstruct a complete image.
def crop_images(input_val):
    if input_val == "44":
        crop_boxes = [[0, 0, 1 / 4, 1 / 4], [1 / 4, 0, 1 / 2, 1 / 4], [1 / 2, 0, 3 / 4, 1 / 4], [3 / 4, 0, 1, 1 / 4],
                      [0, 1 / 4, 1 / 4, 1 / 2], [1 / 4, 1 / 4, 1 / 2, 1 / 2], [1 / 2, 1 / 4, 3 / 4, 1 / 2],
                      [3 / 4, 1 / 4, 1, 1 / 2],
                      [0, 1 / 2, 1 / 4, 3 / 4], [1 / 4, 1 / 2, 1 / 2, 3 / 4], [1 / 2, 1 / 2, 3 / 4, 3 / 4],
                      [3 / 4, 1 / 2, 1, 3 / 4],
                      [0, 3 / 4, 1 / 4, 1], [1 / 4, 3 / 4, 1 / 2, 1], [1 / 2, 3 / 4, 3 / 4, 1], [3 / 4, 3 / 4, 1, 1]]

    elif input_val == "43":
        crop_boxes = [[0, 1 / 8, 1 / 4, 3 / 8], [1 / 4, 1 / 8, 1 / 2, 3 / 8], [1 / 2, 1 / 8, 3 / 4, 3 / 8],
                      [3 / 4, 1 / 8, 1, 3 / 8], [0, 3 / 8, 1 / 4, 5 / 8], [1 / 4, 3 / 8, 1 / 2, 5 / 8],
                      [1 / 2, 3 / 8, 3 / 4, 5 / 8], [3 / 4, 3 / 8, 1, 5 / 8], [0, 5 / 8, 1 / 4, 7 / 8],
                      [1 / 4, 5 / 8, 1 / 2, 7 / 8], [1 / 2, 5 / 8, 3 / 4, 7 / 8], [3 / 4, 5 / 8, 1, 7 / 8]]

    elif input_val == "34":
        crop_boxes = [[1 / 8, 0, 3 / 8, 1 / 4], [3 / 8, 0, 5 / 8, 1 / 4], [5 / 8, 0, 7 / 8, 1 / 4],
                      [1 / 8, 1 / 4, 3 / 8, 1 / 2], [3 / 8, 1 / 4, 5 / 8, 1 / 2], [5 / 8, 1 / 4, 7 / 8, 1 / 2],
                      [1 / 8, 1 / 2, 3 / 8, 3 / 4], [3 / 8, 1 / 2, 5 / 8, 3 / 4], [5 / 8, 1 / 2, 7 / 8, 3 / 4],
                      [1 / 8, 3 / 4, 3 / 8, 1], [3 / 8, 3 / 4, 5 / 8, 1], [5 / 8, 3 / 4, 7 / 8, 1]]

    raw = glob(os.path.join(img_paths, "*"))
    for i, file in enumerate(raw):
        img = Image.open(file)

        for j, crop_box in enumerate(crop_boxes):
            img_cropped = img.crop([crop_box[0] * input_pixel, crop_box[1] * input_pixel, crop_box[2] * input_pixel,
                                    crop_box[3] * input_pixel]).convert("L")
            img_cropped.save(os.path.join(path_save, f"{i}({j+1})" + ".tif"))


def rename_images(path):
    files = glob(os.path.join(path, "*"))
    for i, file in enumerate(files):
        count = str(i).zfill(4)
        new_filename = os.path.join(path, f"{count}.tif")
        os.rename(file, new_filename)


def test(test_path):
    # Create test data
    i = 0
    imgs = glob(test_path + "*")
    imgdatas = np.ndarray((len(imgs), output_pixel, output_pixel, 1), dtype=np.uint8)
    for imgname in imgs:
        midname = imgname[imgname.rindex("\\") + 1:]
        img = load_img(test_path + midname, color_mode='grayscale')
        img = img_to_array(img)
        imgdatas[i] = img
        i += 1
    np.save(npy_path + 'imgs_test.npy', imgdatas)

    # Convert images to float32 and normalize it
    imgs_test = imgdatas.astype('float32')
    imgs_test /= 255
    mean = imgs_test.mean(axis=0)
    imgs_test -= mean

    # Load the trained model and predict
    model = load_model('../model/U-RNet+.hdf5')
    imgs_mask_test = model.predict(imgs_test, batch_size=1, verbose=1)
    np.save(npy_path + 'imgs_mask_test.npy', imgs_mask_test)
    imgs_mask_test[imgs_mask_test > 0.7] = 1
    imgs_mask_test[imgs_mask_test <= 0.7] = 0
    for i in range(imgs_mask_test.shape[0]):
        img = imgs_mask_test[i]
        img = array_to_img(img)
        img.save(test_save + "bw/%d.tif" % i)
    return model


def color_enhance(img):
    w, h, channel = img.shape
    img_color = np.zeros((w, h, 3), dtype=np.uint8)

    for row in range(w):
        for col in range(h):
            i_gray = img[row][col]
            i_gray1 = np.mean(i_gray)

            # Enhance the pixel color based on its grayscale intensity
            if np.any(i_gray < 64):
                img_color[row][col] = [255, i_gray1 * 4, 0]
            elif np.any(i_gray < 128):
                img_color[row][col] = [0, (128 - i_gray1) * 4 - 1, 255]
            elif np.any(i_gray < 192):
                img_color[row][col] = [0, (i_gray1 - 128) * 4, 255]
            else:
                img_color[row][col] = [0, (192 - i_gray1) * 4 - 1, 255]

    b = np.count_nonzero(img_color[:, :, 0])
    g = np.count_nonzero(img_color[:, :, 1])
    r = np.count_nonzero(img_color[:, :, 2])
    return img_color, b, g, r


def process_pseudo(path, pseudo):
    file_list = glob(os.path.join(path, '*'))
    for infile in file_list:
        i = Path(infile).stem
        test_img = cv2.imread(infile, 1)
        pseudo_img, b, g, r = color_enhance(test_img)
        bw_img = cv2.imread(test_save + f'bw/{int(i)}.tif')

        # Merge the color-enhanced image with the binary mask
        merge = cv2.bitwise_and(pseudo_img, bw_img)
        cv2.imwrite(pseudo + f'pseudo{int(i)}.tif', merge)


def stitch(image_dir, output_dir, positions, order, num_positions):
    image_files = sorted(glob(os.path.join(image_dir, '*')), key=os.path.getctime)
    images = [Image.open(image_file) for image_file in image_files]
    n = len(image_files) // num_positions
    for i in range(n):
        new_image = Image.new('RGB', (2048, 2048), 'black')

        # Paste patches onto the new image according to the specified positions and order
        for j, pos_index in enumerate(order):
            new_image.paste(images[pos_index + num_positions * i], positions[j])
        new_image.save(os.path.join(output_dir, str(i) + '.tif'))


def merge_images(input_path, output_path):
    num_raw = len(glob('../testraw/*'))
    for i in range(num_raw):
        stitch44 = cv2.imread('../results/results_44/' + input_path + '/' + str(i) + '.tif')
        stitch34 = cv2.imread('../results/results_34/' + input_path + '/' + str(i) + '.tif')
        stitch43 = cv2.imread('../results/results_43/' + input_path + '/' + str(i) + '.tif')

        stitch44 = cv2.bitwise_not(stitch44)
        stitch34 = cv2.bitwise_not(stitch34)
        stitch43 = cv2.bitwise_not(stitch43)

        merge1 = cv2.bitwise_and(stitch44, stitch34)
        merge2 = cv2.bitwise_and(merge1, stitch43)
        merge2 = cv2.bitwise_not(merge2)
        cv2.imwrite(output_path + '/' + str(i) + '.tif', merge2)


if __name__ == "__main__":
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        except RuntimeError as e:
            print(e)
    else:
        tf.config.set_visible_devices([], 'GPU')

    # Segment
    img_paths = "../testraw/"
    input_pixel = 2048
    output_pixel = 512

    test_folders = ["44", "43", "34"]
    for folder in test_folders:
        path_save = f"../test/test_{folder}/"
        clear_files(path_save)
        npy_path = f"../npydata/npydata_{folder}/"
        clear_files(npy_path)
        test_save = f"../results/results_{folder}/"
        clear_files(test_save)
        pseudo_save = f"../results/results_{folder}/pseudo/"
        clear_files(pseudo_save)

        crop_images(folder)
        rename_images(path_save)
        test(path_save)
        process_pseudo(path_save, pseudo_save)

    # Paste patches onto the new image according to the specified positions and order
    final_results = '../final_results/'
    clear_files(final_results)
    p_values = ["bw", "pseudo"]
    positions_44 = [(0, 0), (512, 0), (1024, 0), (1536, 0),
                    (0, 512), (512, 512), (1024, 512), (1536, 512),
                    (0, 1024), (512, 1024), (1024, 1024), (1536, 1024),
                    (0, 1536), (512, 1536), (1024, 1536), (1536, 1536)]
    order_44 = [0, 8, 9, 10, 11, 12, 13, 14, 15, 1, 2, 3, 4, 5, 6, 7]

    positions_43 = [(0, 256), (512, 256), (1024, 256), (1536, 256),
                    (0, 768), (512, 768), (1024, 768), (1536, 768),
                    (0, 1280), (512, 1280), (1024, 1280), (1536, 1280)]
    order_43 = [0, 4, 5, 6, 7, 8, 9, 10, 11, 1, 2, 3]

    positions_34 = [(256, 0), (768, 0), (1280, 0),
                    (256, 512), (768, 512), (1280, 512),
                    (256, 1024), (768, 1024), (1280, 1024),
                    (256, 1536), (768, 1536), (1280, 1536)]
    order_34 = [0, 4, 5, 6, 7, 8, 9, 10, 11, 1, 2, 3]

    for p in p_values:
        stitch(f"../results/results_44/{p}/", f"../results/results_44/{p}_stitch/", positions_44, order_44, 16)
        stitch(f"../results/results_43/{p}/", f"../results/results_43/{p}_stitch/", positions_43, order_43, 12)
        stitch(f"../results/results_34/{p}/", f"../results/results_34/{p}_stitch/", positions_34, order_34, 12)
        merge_images(f'{p}_stitch', f'../final_results/{p}/')
