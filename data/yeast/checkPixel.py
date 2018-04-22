import cv2
import numpy as np
import scipy.io as sio

seg_mat = np.full((48,80,3), 0)

rootDir = "/Users/anuragarora/Desktop/ppts/CSE523/segnet/data/yeast/temp/"

img = cv2.imread(rootDir + "2651.png")
mat = sio.loadmat(rootDir +"2651.mat")["2651"]

f1 = open(rootDir + "mat.txt",'w', 0)
f2 = open(rootDir + "img.txt",'w', 0)
for i in range(48):
	for j in range(80):
		f1.write(str(img[i][j][0]) + str(img[i][j][1]) + str(img[i][j][2]) + "\n")
		f2.write(str(mat[i][j][0]) + str(mat[i][j][1]) + str(mat[i][j][2]) + "\n")

f1.close()
f2.close()