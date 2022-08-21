'''
图像矫正，透视变换
求变换之前和变换之后的坐标
'''
import numpy as np
import cv2
import math

img = cv2.imread('../data/paper.jpg')
cv2.imshow('img', img)
# 灰度化
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imshow('gray', gray)
# 二值化
# t,binary = cv2.threshold(gray,190,255,cv2.THRESH_BINARY)
# cv2.imshow('binary',binary)
# 边沿检测
# sobel = cv2.Sobel(gray,cv2.CV_64F,1,1,ksize=5)
# cv2.imshow('sobel',sobel)
# lap = cv2.Laplacian(gray,cv2.CV_64F,ksize=5)
# cv2.imshow('Lap',lap)

# Gaussian模糊
blured = cv2.GaussianBlur(gray, (5, 5), 0)
# 闭运算
kernel = np.ones((3, 3), np.uint8)
close = cv2.morphologyEx(blured, cv2.MORPH_CLOSE, kernel)

# Canny
canny = cv2.Canny(close, 30, 120)
# cv2.imshow('canny', canny)

# 查找轮廓
image, cnts, hie = cv2.findContours(canny,
                                    cv2.RETR_EXTERNAL,  # 只检测外层轮廓
                                    cv2.CHAIN_APPROX_SIMPLE)  # 保存终点坐标
# for i in range(len(cnts)):
#     print(cnts[i].shape)
# 绘制轮廓
# cv2.drawContours(gray,cnts,1,(0,0,0),2)
# cv2.imshow('gray_cnt',gray)


# 计算轮廓面积，排序
doccnt = None
if len(cnts) > 0:
    cnts = sorted(cnts,
                  key=cv2.contourArea,  # 排序依据：面积
                  reverse=True)  # 降序排序
    #拿到面积最大的四边形
    for cnt in cnts:
        eps = 0.02 * cv2.arcLength(cnt,True)
        approx = cv2.approxPolyDP(cnt,eps,True)
        if len(approx) == 4:
            doccnt = approx
            break

#绘制四个角的顶点(左上角，左下角，右下角，右上角)
points = []
for peak in doccnt:
    peak = peak[0]
    # cv2.circle(gray,tuple(peak),10,(0,0,0),2)
    points.append(peak)

# cv2.imshow('point_circle',gray)

#变换之前的坐标点
src = np.array(points,dtype='float32')

# 求宽度，和高度
h = int(math.sqrt((src[0][0] - src[1][0])**2 + (src[0][1] - src[1][1])**2))
w = int(math.sqrt((src[0][0] - src[3][0])**2 + (src[0][1] - src[3][1])**2))
# print(f'h:{h},w:{w}')

#生成变换之后的坐标点
dst = np.float32([[0,0],[0,h],[w,h],[w,0]])
#生成透视变换矩阵
M = cv2.getPerspectiveTransform(src,dst)
#执行透视变换
res = cv2.warpPerspective(img,M,(w,h))
cv2.imshow('res',res)

cv2.waitKey()
cv2.destroyAllWindows()


