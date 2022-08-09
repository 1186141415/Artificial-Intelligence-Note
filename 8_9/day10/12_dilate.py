'''图像的膨胀'''
import cv2
import numpy as np

img = cv2.imread('../data/9.png')
cv2.imshow('img',img)

kernel = np.ones((3,3),np.uint8)
res = cv2.dilate(img,kernel,iterations=4)
cv2.imshow('res',res)


cv2.waitKey()
cv2.destroyAllWindows()
