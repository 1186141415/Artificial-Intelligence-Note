'''对彩色图像的通道进行操作'''
import cv2

img = cv2.imread('../data/opencv2.png') #BGR
cv2.imshow('img',img)

b = img[:,:,0]
cv2.imshow('b',b)

img[:,:,0] = 0
cv2.imshow('b0',img)

img[:,:,1] = 0
cv2.imshow('b0-g0',img)

cv2.waitKey()
cv2.destroyAllWindows()
