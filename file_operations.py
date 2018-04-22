from vgg16 import vgg16
import tensorflow as tf
import numpy as np
from scipy.misc import imread, imresize
from imagenet_classes import class_names
import scipy.io as sio
import glob
import os



def read_experiment_images(folder):
    sess = tf.Session()
    imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
    vgg = vgg16(imgs, 'vgg16_weights.npz', sess)

    os.chdir(folder)

    relevant_files = glob.glob("*.png")

    num_images = len(relevant_files);

    cnt = 0;

    np_imgs = np.zeros((num_images,224,224,3))

    for file in relevant_files:
        img1 = imread(folder+file, mode='RGB')  # was laska.png
        img1 = imresize(img1, (224, 224))
        np_imgs[cnt,:,:] = img1;
        cnt += 1
        print cnt


    #representation = sess.run(vgg.fc2l, feed_dict={vgg.imgs: np_imgs})


    #representations[cnt,:] = representation;

    representations = sess.run(vgg.fc2l, feed_dict={vgg.imgs: np_imgs})

    return representations,relevant_files


#read_experiment_images("/Users/aliselmanaydin/Desktop/masks/picture_vocab/")


# quantify the sum of the distances to the closest
def temporal_uniformity_shell(S,):
    if len(X) > 0:
        min_dist = distMat[:, X].min(axis=1)
        min_dist[min_dist > norm] = norm
        return min_dist.mean() / norm
    else:
        return 1