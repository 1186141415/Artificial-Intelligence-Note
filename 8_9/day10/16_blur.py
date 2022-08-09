'''
图像的模糊
'''
import cv2

img = cv2.imread('../data/salt.jpg')
cv2.imshow('img',img)

#均值滤波
blur = cv2.blur(img,(5,5))
cv2.imshow('blur',blur)
#高斯滤波
gaussian = cv2.GaussianBlur(img,(5,5),3)
cv2.imshow('gaussian',gaussian)
#中值滤波
median_blur = cv2.medianBlur(img,5)
cv2.imshow('median',median_blur)
cv2.imwrite('median.jpg',median_blur)


cv2.waitKey()
cv2.destroyAllWindows()