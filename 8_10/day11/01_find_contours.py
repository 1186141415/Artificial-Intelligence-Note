'''
查找并绘制轮廓
'''
import cv2

img = cv2.imread('../data/3.png')
cv2.imshow('img', img)
# 灰度图像
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 二值化
t, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
cv2.imshow('binary', binary)
# 查找轮廓
image, cnts, hie = cv2.findContours(binary,
                                    cv2.RETR_EXTERNAL,  # 只检测外层轮廓
                                    cv2.CHAIN_APPROX_NONE)  # 保存所有坐标点

# print(type(cnts))
# print(len(cnts))
# print(cnts[0])
# for i in range(len(cnts)):
#     print(cnts[i].shape)
# print(hie)
# 绘制轮廓
img_cnt = cv2.drawContours(img,  # 要绘制的图像
                           cnts,  # 绘制的坐标点
                           -1,  # 绘制所有轮廓
                           (0, 0, 255),  # 颜色 BGR
                           2)  # 线条粗细 px
cv2.imshow('res',img_cnt)


cv2.waitKey()
cv2.destroyAllWindows()

