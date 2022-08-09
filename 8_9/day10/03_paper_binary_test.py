'''将paper.jpg进行二值化处理
   纸变为白色，背景变为黑色'''
import cv2

img = cv2.imread('../data/paper.jpg',0)

res = cv2.resize(img,(200,200))
cv2.imshow('img',res )

t,binary = cv2.threshold(res,200,255,cv2.THRESH_BINARY)
cv2.imshow('binary',binary)

cv2.waitKey()
cv2.destroyAllWindows()