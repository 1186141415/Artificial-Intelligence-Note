# 01_capsules_detection.py
# 利用图像技术实现胶囊瑕疵检测
import cv2
import numpy
import os


def empty_detection(img_path, fn, im, im_gray):
    """
    检测传入的图像是否为空胶囊
    :param img_path: 图片完整路径
    :param fn: 图片名称
    :param im: 三通道彩色图像
    :param im_gray: 单通道灰度图像
    :return: 空胶囊返回True, 非空返回False
    """
    # 模糊处理
    im_blur = cv2.GaussianBlur(im_gray, (3, 3), 0)
    # 二值化
    t, im_bin = cv2.threshold(im_blur,
                              210, 255,
                              cv2.THRESH_BINARY)
    cv2.imshow("im_bin", im_bin)
    # 提取轮廓
    img, cnts, hie = cv2.findContours(
        im_bin, # 输入图像
        cv2.RETR_CCOMP, # 两层轮廓
        cv2.CHAIN_APPROX_NONE) # 存储所有点的坐标

    # 对轮廓进行筛选(按周长)
    new_cnts = [] # 存放筛选后的轮廓
    for c in cnts: # 遍历每个轮廓
        cir_len = cv2.arcLength(c, True) # 计算轮廓周长
        if cir_len >= 1000:
            new_cnts.append(c) # 存入筛选后的列表

    # 绘制轮廓
    im_cnt = cv2.drawContours(im, # 在原三通道图像上绘制
                              new_cnts, # 轮廓数据
                              -1, # 绘制所有轮廓
                              (0,0,255), 2) # 颜色、粗细
    cv2.imshow("im_cnt", im_cnt)

    if len(new_cnts) == 1: # 空胶囊
        print("空胶囊:", fn)
        new_path = os.path.join("capsules/empty", fn)#新路径
        os.rename(img_path, new_path) # 移动文件
        print("移动文件成功:%s ---> %s" % (img_path, new_path))
        return True
    else: # 非空
        return False

def bub_detection(img_path, fn, im, im_gray):# 判断气泡
    # 模糊处理
    im_blur = cv2.GaussianBlur(im_gray, (3, 3), 0)
    # 边沿检测
    im_canny = cv2.Canny(im_blur, 60, 240)
    cv2.imshow("im_canny", im_canny)
    # 轮廓检测
    img, cnts, hie = cv2.findContours(
        im_canny,  # 输入图像
        cv2.RETR_CCOMP,  # 两层轮廓
        cv2.CHAIN_APPROX_NONE)  # 存储所有点的坐标

    # 轮廓过滤
    new_cnts = []
    for i in range(len(cnts)):
        area = cv2.contourArea(cnts[i]) # 计算第i个轮廓的面积
        cir_len = cv2.arcLength(cnts[i], True) #第i个轮廓的周长

        if area >= 10000 or cir_len >= 900 or area < 5:
            continue

        if hie[0][i][3] != -1: # 有父轮廓，保留
            new_cnts.append(cnts[i])

    im_cnt = cv2.drawContours(im,  # 在原三通道图像上绘制
                              new_cnts,  # 轮廓数据
                              -1,  # 绘制所有轮廓
                              (0, 0, 255), 2)  # 颜色、粗细
    cv2.imshow("im_cnt", im_cnt)

    if len(new_cnts) > 0: # 有气泡
        print("气泡:", fn)
        new_path = os.path.join("capsules/bub/", fn)#新路径
        os.rename(img_path, new_path) # 移动文件
        print("移动文件成功:%s ---> %s" % (img_path, new_path))
        return True
    else: # 没气泡
        return False

if __name__ == "__main__":
    # 读取每张图像
    img_dir = "capsules/"  # 图像所在目录
    img_files = os.listdir(img_dir)  # 列出目录下所有内容

    # 循环判断每张图像有没有瑕疵
    for fn in img_files:  # 遍历
        img_path = os.path.join(img_dir, fn)  # 图片完整路径

        if os.path.isdir(img_path):  # 目录
            continue

        im = cv2.imread(img_path)  # 读取图像
        im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)  # 转灰度图
        cv2.imshow("im", im)
        cv2.imshow("im_gray", im_gray)

        # 判断是否为空
        is_empty = False
        #is_empty = empty_detection(img_path, fn, im, im_gray)

        # 判断是否有气泡
        is_bub = False
        if not is_empty: # 非空，判断有没有气泡
            is_bub = bub_detection(img_path, fn, im, im_gray)

        # 判断大小头
        # bal_detectoin()

        cv2.waitKey()
        cv2.destroyAllWindows()
