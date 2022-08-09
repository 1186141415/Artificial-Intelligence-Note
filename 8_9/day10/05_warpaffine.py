'''
图像的仿射变换:平移，旋转
'''
import cv2
import numpy as np

def translate(img,x,y):
    '''
    平移
    :param img: 平移的图像
    :param x: 水平方向平移的像素值
    :param y: 垂直方向平移的像素值
    :return: 返回平移结果
    '''
    h,w = img.shape[:2]
    #平移平移矩阵
    M = np.float32([[1,0,x],
                    [0,1,y]])
    res = cv2.warpAffine(img,
                         M,
                         (w,h))
    return res

def rotate(img,angle,center=None):
    h, w = img.shape[:2]
    if center is None:
        center = (w/2,h/2)

    #生成旋转矩阵
    M = cv2.getRotationMatrix2D(center,angle,scale=1.0)
    res = cv2.warpAffine(img,
                         M,
                         (w,h))
    return res


if __name__ == '__main__':
    img = cv2.imread('../data/lena.jpg')
    cv2.imshow('img',img)
    # 平移
    x_50 = translate(img,50,-60)
    cv2.imshow('x_50',x_50)
    #旋转(默认逆时针)
    r_45 = rotate(img,-45)
    cv2.imshow('r45',r_45)

    cv2.waitKey()
    cv2.destroyAllWindows()


