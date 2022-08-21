'''
拟合轮廓的多边形
'''
import cv2

img = cv2.imread('../data/cloud.png')
#灰度化
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#二值化
t,binary = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
cv2.imshow('binary',binary)
#查找轮廓
image,cnts,hie = cv2.findContours(binary,
                                  cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_NONE)
# print(len(cnts))
# 根据轮廓坐标，生成多边形坐标点集
#精度1
adp1 = img.copy()
eps = 0.005 * cv2.arcLength(cnts[0],True)
points = cv2.approxPolyDP(cnts[0], #轮廓坐标
                          eps, #精度
                          True)#是否闭合
# print(points.shape)
adp1 = cv2.drawContours(adp1,
                        [points],
                        -1,
                        (0,0,255),
                        2)
cv2.imshow('adp1_0.005',adp1)
#精度2
adp2 = img.copy()
eps2 = 0.01 * cv2.arcLength(cnts[0],True)
points2 = cv2.approxPolyDP(cnts[0],eps2,True)
adp2 = cv2.drawContours(adp2,[points2],0,(0,0,255),2)
cv2.imshow('adp2_0.01',adp2)

cv2.waitKey()
cv2.destroyAllWindows()
