'''
度盘区域瑕疵检测
'''
import cv2
import numpy as np
# 1.加载图像
img = cv2.imread('../data/CPU3.png')
cv2.imshow('img',img)
# 2.灰度化
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow('gray',gray)
# 3.二值化
t,binary = cv2.threshold(gray,160,255,cv2.THRESH_BINARY)
cv2.imshow('binary',binary)
# 4.查找度盘区域轮廓
image,cnts,hie = cv2.findContours(binary,
                                  cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_NONE)
# print(len(cnts))
# 5.生成数组mask, zeros_like(binary)
mask = np.zeros_like(binary)
# cv2.imshow('mask',mask)
# 6.使用实心化填充，将度盘轮廓画在mask上   白色(255,0,0)
img_fill = cv2.drawContours(mask,cnts,-1,(255,0,0),-1)
cv2.imshow('img_fill',img_fill)
# 7.使用二值化图像，与mask做减法
img_sub = cv2.subtract(img_fill,binary)
cv2.imshow('sub',img_sub)
# 8.使用闭运算，将瑕疵点收缩在一起
kernel = np.ones((5,5),np.uint8)
close = cv2.morphologyEx(img_sub,cv2.MORPH_CLOSE,kernel,iterations=2)
cv2.imshow('close',close)
# 9.查找瑕疵的轮廓
image,cnts,hie = cv2.findContours(close,
                                  cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_NONE)
# print(len(cnts))
if len(cnts) > 0:
    cnts = sorted(cnts,
                  key=cv2.contourArea,
                  reverse=True)
# 10.拟合瑕疵的最小外接圆
center,radius = cv2.minEnclosingCircle(cnts[0])
center = (int(center[0]),int(center[1]))
radius = int(radius)
# 11.将瑕疵的最小外接圆画在原图上
res = cv2.circle(img,center,radius,(0,0,255),2)
cv2.imshow('res',res)

#根据业务指标，判断是否为瑕疵
area = cv2.contourArea(cnts[0])
if area > 10:
    print('有瑕疵,瑕疵面积为:{}'.format(area))
else:
    print('好产品')

cv2.waitKey()
cv2.destroyAllWindows()

