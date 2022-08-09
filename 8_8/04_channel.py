import cv2

img = cv2.imread('opencv2.png')
cv2.imshow('img', img)

b = img[:, :, 0]
cv2.imshow("b", b)

img[:, :, 0] = 0  # 图三
cv2.imshow('3', img)

img[:, :, 1] = 0  # 图四
cv2.imshow('4', img)

cv2.waitKey()  # 等待用户按键反馈
cv2.destroyAllWindows()  # 销毁所有创建的窗口

print(cv2.waitKey())
