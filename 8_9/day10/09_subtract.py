'''
图像相减，找到差异
'''
import cv2

img_3 = cv2.imread('../data/3.png',1)
img_4 = cv2.imread('../data/4.png',1)

res = cv2.subtract(img_4,img_3)
cv2.imshow('img3',img_3)
cv2.imshow('img4',img_4)
cv2.imshow('res',res)

cv2.waitKey()
cv2.destroyAllWindows()