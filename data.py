from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import glob
import skimage.io as io
import skimage.transform as trans
from keras import backend as K
import tensorflow as tf

label_1 = [0, 0, 0]

label_2 = [128, 0, 0]
label_3 = [0, 128, 0]
label_4 = [128, 128, 0]
label_5 = [0, 0, 128]
label_6 = [128, 0, 128]
label_7 = [0, 128, 128]
label_8 = [128, 128, 128]
label_9 = [64, 0, 0]
label_10 = [192, 0, 0]
label_11 = [64, 128, 0]

label_12 = [192, 128, 0]
label_13 = [64, 0, 128]
label_14 = [192, 0, 128]
label_15 = [64, 128, 128]
label_16 = [192, 128, 128]
label_17 = [0, 64, 0]
label_18 = [128, 64, 0]
label_19 = [0, 192, 0]
label_20 = [128, 192, 0]

COLOR_DICT = np.array(
    [label_1, label_2, label_2, label_4, label_12, label_6, label_8, label_8, label_9, label_10, label_11, label_2,
     label_12, label_14, label_15, label_16, label_17, label_18, label_19, label_20])

IMAGE_SIZE = 256


def adjustData(img_1, img_2, img_3, label, flag_multi_class, num_class):
    if (flag_multi_class):
        img_1 = img_1 / 255.
        img_2 = img_2 / 255.
        img_3 = img_3 / 255.
        label = label[:, :, :, 0] if (len(label.shape) == 4) else label[:, :, 0]
        new_label = np.zeros(label.shape + (num_class,))
        for i in range(num_class):
            new_label[label == i, i] = 1
        label = new_label
    elif (np.max(img_1) > 1):
        img_1 = img_1 / 255.
        img_2 = img_2 / 255.
        img_3 = img_3 / 255.
        label = label / 255.
        label[label > 0.5] = 1
        label[label <= 0.5] = 0
    return (img_1, img_2, img_3, label)


def trainGenerator(batch_size, aug_dict, train_path, image_folder_1, image_folder_2, image_folder_3, label_folder,
                   image_color_mode='grayscale',
                   label_color_mode='grayscale', image_save_prefix='image', label_save_prefix='label',
                   flag_multi_class=True, num_class=20, save_to_dir=None, target_size=(IMAGE_SIZE, IMAGE_SIZE), seed=1):
    image_1_datagen = ImageDataGenerator(**aug_dict)
    image_2_datagen = ImageDataGenerator(**aug_dict)
    image_3_datagen = ImageDataGenerator(**aug_dict)
    label_datagen = ImageDataGenerator(**aug_dict)
    image_1_generator = image_1_datagen.flow_from_directory(
        train_path,
        classes=[image_folder_1],
        class_mode=None,
        color_mode=image_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=image_save_prefix,
        seed=seed
    )
    image_2_generator = image_2_datagen.flow_from_directory(
        train_path,
        classes=[image_folder_2],
        class_mode=None,
        color_mode=image_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=image_save_prefix,
        seed=seed
    )
    image_3_generator = image_3_datagen.flow_from_directory(
        train_path,
        classes=[image_folder_3],
        class_mode=None,
        color_mode=image_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=image_save_prefix,
        seed=seed
    )
    label_generator = label_datagen.flow_from_directory(
        train_path,
        classes=[label_folder],
        class_mode=None,
        color_mode=label_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=label_save_prefix,
        seed=seed
    )
    train_generator = zip(image_1_generator, image_2_generator, image_3_generator, label_generator)
    for img_1, img_2, img_3, label in train_generator:
        img_1, img_2, img_3, label = adjustData(img_1, img_2, img_3, label, flag_multi_class, num_class)
        yield [img_1, img_2, img_3], label


def getFileNum(test_path):
    for root, dirs, files in os.walk(test_path):
        lens = len(files)
        return lens


def testGenerator(test_path_1, test_path_2, test_path_3, target_size=(IMAGE_SIZE, IMAGE_SIZE), flag_multi_class=True,
                  as_gray=True):
    num_image = getFileNum(test_path_1)
    for i in range(num_image):
        img_1 = io.imread(os.path.join(test_path_1, "%d.png" % i), as_gray=as_gray)
        img_1 = img_1 / 255
        img_1 = trans.resize(img_1, target_size)
        img_1 = np.reshape(img_1, img_1.shape + (1,)) if (not flag_multi_class) else img_1
        img_1 = np.reshape(img_1, (1,) + img_1.shape)

        img_2 = io.imread(os.path.join(test_path_2, "%d.png" % i), as_gray=as_gray)
        img_2 = img_2 / 255
        img_2 = trans.resize(img_2, target_size)
        img_2 = np.reshape(img_2, img_2.shape + (1,)) if (not flag_multi_class) else img_2
        img_2 = np.reshape(img_2, (1,) + img_2.shape)

        img_3 = io.imread(os.path.join(test_path_3, "%d.png" % i), as_gray=as_gray)
        img_3 = img_3 / 255
        img_3 = trans.resize(img_3, target_size)
        img_3 = np.reshape(img_3, img_3.shape + (1,)) if (not flag_multi_class) else img_3
        img_3 = np.reshape(img_3, (1,) + img_3.shape)

        yield [img_1, img_2, img_3]


def saveResult(save_path, npyfile, flag_multi_class=True):
    for i, item in enumerate(npyfile):
        if flag_multi_class:
            img = item
            img_out = np.zeros(img[:, :, 0].shape + (3,))
            for row in range(img.shape[0]):
                for col in range(img.shape[1]):
                    index_of_class = np.argmax(img[row, col])
                    img_out[row, col] = COLOR_DICT[index_of_class]
                    #img_out[row, col] = index_of_class
            img = img_out.astype(np.uint8)
            io.imsave(os.path.join(save_path, '%s_predict.png' % i), img)
        else:
            img = item[:, :, 0]
            img[img > 0.5] = 1
            img[img <= 0.5] = 0
            img = img * 255.
            io.imsave(os.path.join(save_path, '%s_predict.png' % i), img)
