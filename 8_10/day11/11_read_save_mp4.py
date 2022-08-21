import numpy as np
import cv2

""" 编解码4字标记值说明
cv2.VideoWriter_fourcc（'I','4','2','0'）表示未压缩的YUV颜色编码格式，色度子采样为4:2:0。
    该编码格式具有较好的兼容性，但产生的文件较大，文件扩展名为.avi。
cv2.VideoWriter_fourcc（'P','I','M','I'）表示 MPEG-1编码类型，生成的文件的扩展名为.avi。
cv2.VideoWriter_fourcc（'X','V','I','D'）表示MPEG-4编码类型。如果希望得到的视频大小为平均值，可以选用这个参数组合。
    该组合生成的文件的扩展名为.avi。
cv2.VideoWriter_fourcc（'T','H','E','O'）表示Ogg Vorbis编码类型，文件的扩展名为.ogv。
cv2.VideoWriter_fourcc（'F','L','V','I'）表示Flash视频，生成的文件的扩展名为.flv。
"""
cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc("I", "4", "2", "0")  # 编解码4字标记值
out = cv2.VideoWriter("C:\\Users\\BJTT\\Desktop\\0810\\output.avi",  # 文件名
                      fourcc,  # 编解码类型
                      20,  # fps(帧速度)
                      (640, 480))  # 视频分辨率

while cap.isOpened():
    ret, frame = cap.read()  # 读取帧
    if ret == True:
        out.write(frame)  # 写入帧
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) == 27:  # ESC键
            break
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()