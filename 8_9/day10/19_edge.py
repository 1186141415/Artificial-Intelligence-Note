'''
边沿检测
'''
import cv2

img = cv2.imread('../data/lily.png',0)
cv2.imshow('img',img)

#Sobel
sobel = cv2.Sobel(img,
                  cv2.CV_64F, #图像深度
                  dx=1, #0,1
                  dy=1,
                  ksize=5)
cv2.imshow('sobel',sobel)

#Laplacian
lap = cv2.Laplacian(img,
                    cv2.CV_64F)
cv2.imshow('lap',lap)

#Canny
canny = cv2.Canny(img,
                  50,#滞后阈值
                  360)#模糊度
cv2.imshow('canny',canny)



cv2.waitKey()
cv2.destroyAllWindows()