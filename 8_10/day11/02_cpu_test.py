'''
查找轮廓练习
'''
import cv2
import numpy as np

img = cv2.imread('../data/CPU3.png')
cv2.imshow('img',img)
#灰度化
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#二值化
t,binary = cv2.threshold(gray,160,255,cv2.THRESH_BINARY)
cv2.imshow('binary',binary)
#查找轮廓
image,cnts,hie = cv2.findContours(binary,
                                  cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_NONE)
# print(len(cnts))
img_cnt = cv2.drawContours(img,
                           cnts,
                           -1,
                           (0,0,255),
                           2)
cv2.imshow('res',img_cnt)

cv2.waitKey()
cv2.destroyAllWindows()



