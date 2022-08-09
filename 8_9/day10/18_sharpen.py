'''图像的锐化
增大像素与像素之间的差异值
'''
import cv2
import numpy as np

img = cv2.imread('../data/lena.jpg',0)
cv2.imshow('img',img)

#锐化算子1
sharpen1 = np.array([[-1,-1,-1],
                     [-1,9,-1],
                     [-1,-1,-1]])
res1 = cv2.filter2D(img,
                    -1,#图像的深度（通道数）-1与原始图像一致
                    sharpen1)
cv2.imshow('res1',res1)
#锐化算子2
sharpen2 = np.array([[0,-1,0],
                     [-1,8,-1],
                     [0,1,0]]) / 4.0
res2 = cv2.filter2D(img,-1,sharpen2)
cv2.imshow('res2',res2)


cv2.waitKey()
cv2.destroyAllWindows()


