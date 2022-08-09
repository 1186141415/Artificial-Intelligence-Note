'''
二值化和反二值化
'''
import cv2

img = cv2.imread('../data/lena.jpg')
cv2.imshow('img', img)
# 灰度化
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('gray', gray)
# 二值化
t, res = cv2.threshold(gray,
                       100,
                       255,
                       cv2.THRESH_BINARY)
cv2.imshow('res',res)

#反二值化
t,res_inv = cv2.threshold(gray,
                          100,
                          255,
                          cv2.THRESH_BINARY_INV)
cv2.imshow('res_inv',res_inv)

cv2.waitKey()
cv2.destroyAllWindows()
