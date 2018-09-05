import cv2
import numpy as np
import glob
import os


def imageRead():
    rootDir = "/Users/anuragarora/Desktop/ppts/CSE523/segnet/data/yeast/"
    
    os.chdir(rootDir + "clustersegm0/")
    list_of_seg = sorted(glob.glob('*.png'))

    segmeFolder = rootDir +"clustersegm0Corrected/"

    for count in range(len(list_of_seg)):
        segImage = cv2.imread(rootDir + "clustersegm0/" + list_of_seg[count])

        for j in range(48):
            for k in range(80):
                for i in range(3):
                    if segImage[j][k][i] < 127:
                        segImage[j][k][i] = 0
                    else:
                        segImage[j][k][i] = 255
        
        cv2.imwrite(segmeFolder + list_of_seg[count], segImage)
    
if __name__ == '__main__':
    imageRead()


