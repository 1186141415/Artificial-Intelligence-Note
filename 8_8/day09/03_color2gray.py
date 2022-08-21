'''
彩色图像转为灰度图像（不可逆）
转换色彩空间
彩色色彩空间：默认为BGR   灰度：GRAY
'''
import cv2

img = cv2.imread('../data/dog2.png') #0:灰度  1：彩色
cv2.imshow('img',img)
#BGR-->GRAY
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow('gray',gray)

cv2.waitKey()
cv2.destroyAllWindows()









