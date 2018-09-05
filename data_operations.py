from scipy.misc import imread, imresize
import glob
import os
import numpy as np
import tensorflow as tf
import scipy.spatial.distance as dist
import scipy.io as sio

def read_data(images_dir_train,labels_dir_train,images_dir_val,labels_dir_val,images_dir_test,labels_dir_test, images_dir_train_extra, labels_dir_train_extra, extraData, realData, budgetFake, budgetReal):
    if (realData):
        os.chdir(images_dir_train)
        list_of_images_train = sorted(glob.glob('*.png'))
        os.chdir(labels_dir_train)
        list_of_labels_train = sorted(glob.glob('*.png'))

        np_imgs_train = np.zeros((len(list_of_images_train),48,80,3))
        np_labels_train = np.zeros((len(list_of_labels_train),48,80, 3))

        for i in range(len(list_of_images_train)):
            current_image = imread(images_dir_train+list_of_images_train[i])
            # current_label = sio.loadmat(labels_dir_train+list_of_labels_train[i])[str(list_of_labels_train[i][:4])]
            current_label = imread(labels_dir_train+list_of_labels_train[i])
            np_imgs_train[i,...] = current_image
            np_labels_train[i,...] = current_label

    if(extraData):
        os.chdir(images_dir_train_extra)
        list_of_images_train_extra = sorted(glob.glob('*.png'))
        os.chdir(labels_dir_train_extra)
        list_of_labels_train_extra = sorted(glob.glob('*.png'))
        np_imgs_train_extra = np.zeros((len(list_of_images_train_extra),48,80,3))
        np_labels_train_extra = np.zeros((len(list_of_labels_train_extra),48,80, 3))

        for i in range(len(list_of_images_train_extra)):
            current_image = imread(images_dir_train_extra + list_of_images_train_extra[i])
            current_label = imread(labels_dir_train_extra + list_of_labels_train_extra[i])
            np_imgs_train_extra[i,...] = current_image
            np_labels_train_extra[i,...] = current_label

        # tr_range = np.arange(np_imgs_train_extra.shape[0])
        # np.random.shuffle(tr_range)
        # np_imgs_train_extra = np_imgs_train_extra[tr_range[:budget]]
        # np_labels_train_extra = np_labels_train_extra[tr_range[:budget]]


    os.chdir(images_dir_val)
    list_of_images_val = sorted(glob.glob('*.png'))
    os.chdir(labels_dir_val)
    list_of_labels_val = sorted(glob.glob('*.png'))


    os.chdir(images_dir_test)
    list_of_images_test = sorted(glob.glob('*.png'))
    os.chdir(labels_dir_test)
    list_of_labels_test = sorted(glob.glob('*.png'))

    np_imgs_val = np.zeros((len(list_of_images_val),48,80,3))
    np_labels_val = np.zeros((len(list_of_labels_val),48,80, 3))

    np_imgs_test = np.zeros((len(list_of_images_test),48,80,3))
    np_labels_test = np.zeros((len(list_of_labels_test),48,80, 3))


    for i in range(len(list_of_images_val)):
        current_image = imread(images_dir_val+list_of_images_val[i])
        # current_label = sio.loadmat(labels_dir_val+list_of_labels_val[i])[str(list_of_labels_val[i][:4])]
        current_label = imread(labels_dir_val+list_of_labels_val[i])
        np_imgs_val[i,...] = current_image
        np_labels_val[i,...] = current_label

    for i in range(len(list_of_images_test)):
        current_image = imread(images_dir_test+list_of_images_test[i])
        # current_label = sio.loadmat(labels_dir_test+list_of_labels_test[i])[str(list_of_labels_test[i][:4])]
        current_label = imread(labels_dir_test+list_of_labels_test[i])
        np_imgs_test[i,...] = current_image
        np_labels_test[i,...] = current_label


    trX = np_imgs_train if realData else None
    trY = np_labels_train if realData else None

    valX = np_imgs_val
    valY = np_labels_val

    teX = np_imgs_test
    teY = np_labels_test

    if(extraData):
        return trX,valX,teX,trY,valY,teY, np_imgs_train_extra, np_labels_train_extra
    else:
        return trX,valX,teX,trY,valY,teY, None, None


    # tr_range = np.arange(trX.shape[0])
    # np.random.shuffle(tr_range)
    # trX = trX[tr_range]
    # trY = trY[tr_range]
    #
    # val_range = np.arange(valX.shape[0])
    # np.random.shuffle(val_range)
    # valX = valX[val_range]
    # valY = valY[val_range]
    #
    # te_range = np.arange(teX.shape[0])
    # np.random.shuffle(te_range)
    # teX = teX[te_range]
    # teY = teY[te_range]