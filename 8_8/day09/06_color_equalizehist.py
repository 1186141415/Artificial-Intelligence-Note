'''
彩色图像亮度通道直方图均衡化
'''
import cv2

img = cv2.imread('../data/sunrise.jpg')
cv2.imshow('img',img)
#BGR-->YUV
yuv = cv2.cvtColor(img,cv2.COLOR_BGR2YUV)
#均衡化
yuv[...,0] = cv2.equalizeHist(yuv[...,0])

res = cv2.cvtColor(yuv,cv2.COLOR_YUV2BGR)
cv2.imshow('res',res)

cv2.waitKey()
cv2.destroyAllWindows()