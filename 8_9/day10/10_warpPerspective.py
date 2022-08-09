'''
透视变换
'''
import cv2
import numpy as np

img = cv2.imread('../data/pers.png')
cv2.imshow('img', img)

h,w = img.shape[:2]
src = np.float32([[58, 2], [167, 9], [8, 196], [126, 196]])
dst = np.float32([[16, 2], [167, 8], [8, 196], [169, 196]])
M = cv2.getPerspectiveTransform(src,dst)
# 透视变换
res = cv2.warpPerspective(img,
                          M,
                          (w, h))

cv2.imshow('res',res)

cv2.waitKey()
cv2.destroyAllWindows()
