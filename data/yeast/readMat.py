import cv2
import numpy as np
import scipy.io as sio
import os
import glob

seg_mat = np.full((48,80,3), 0)

rootDir = "/Users/anuragarora/Desktop/ppts/CSE523/segnet/data/yeast/temp/"
rootDir1 = "/Users/anuragarora/Desktop/ppts/CSE523/segnet/data/yeast/temp1/"

os.chdir(rootDir1)
list_of_labels_train = sorted(glob.glob('*.png'))


mat = sio.loadmat(rootDir + list_of_labels_train[0][:4] + ".mat")[str(list_of_labels_train[0][:4])]

print(mat)
