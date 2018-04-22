import cv2
import numpy as np
import glob
import os
import scipy.io as sio

def imageRead():
    rootDir = "/Users/anuragarora/Desktop/ppts/CSE523/segnet/data/yeast/cluster1segm/"
    os.chdir(rootDir)
    list_of_seg = sorted(glob.glob('*.png'))

    segmeFolder = "/Users/anuragarora/Desktop/ppts/CSE523/segnet/data/yeast/cluster1segmMat/"

    for count in range(len(list_of_seg)):
        segImage = cv2.imread(rootDir + list_of_seg[count])
        seg_mat = np.full((48,80,3), 0)
        for j in range(48):
            for k in range(80):
                for i in range(3):
                    if segImage[j][k][i] < 127:
                        seg_mat[j][k][i] = 0
                    else:
                        seg_mat[j][k][i] = 255

        sio.savemat(segmeFolder + str(list_of_seg[count])[:4], {str(list_of_seg[count])[:4]: seg_mat})
        # cv2.imwrite(segmeFolder + list_of_seg[count], segImage)
    
if __name__ == '__main__':
    imageRead()


