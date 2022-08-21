'''
读取摄像头，并播放
'''
import cv2

# 读取摄像头对象
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, img = cap.read()  # 捕获一帧图像
    cv2.imshow('img', img)
    key = cv2.waitKey(1)
    if key == 27:
        break
cap.release()  # 释放掉摄像头对象
cv2.destroyAllWindows()
