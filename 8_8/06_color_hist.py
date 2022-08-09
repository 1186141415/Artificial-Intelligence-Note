# 彩色亮度直方图均衡化
import cv2

img = cv2.imread('sunrise.jpg')

cv2.imshow('img', img)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# 均衡化
hsv[:, :, -1] = cv2.equalizeHist(hsv[:, :, -1])

bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

cv2.imshow('img2', bgr)

cv2.waitKey()
cv2.destroyAllWindows()
