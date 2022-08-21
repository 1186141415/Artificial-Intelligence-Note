import numpy as np
import cv2

cap = cv2.VideoCapture("./output.avi")  # 打开视频文件
while cap.isOpened():
    ret, frame = cap.read()  # 读取帧
    cv2.imshow("frame", frame)  # 显示
    c = cv2.waitKey(25)
    if c == 27:  # ESC键
        break

cap.release()  # 释放视频设备
cv2.destroyAllWindows()