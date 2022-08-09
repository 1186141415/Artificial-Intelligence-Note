'''图像的腐蚀'''
import cv2
import numpy as np

img = cv2.imread('../data/5.png')
cv2.imshow('img',img)

#腐蚀核
kernel = np.ones((3,3),np.uint8)
res = cv2.erode(img,kernel,iterations=3)
cv2.imshow('res',res)

cv2.waitKey()
cv2.destroyAllWindows()