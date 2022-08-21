'''
拟合轮廓的最小外接圆
'''
import cv2
import numpy as np

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

#根据轮廓坐标，生成圆形数据参数（圆心和半径）
center,radius = cv2.minEnclosingCircle(cnts[0])
#圆心和半径必须是整数
center = (int(center[0]),int(center[1]))
radius = int(radius)
print(f'圆心:{center},半径是:{radius}')
#画圆 cv2.circle
img_cnt = cv2.circle(img,center,radius,(0,0,255),2)
cv2.imshow('res',img_cnt)


cv2.waitKey()
cv2.destroyAllWindows()


