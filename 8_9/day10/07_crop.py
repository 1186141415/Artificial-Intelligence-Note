'''
图像的裁剪
随机裁剪，中心裁剪
'''
import cv2
import numpy as np

def random_crop(img,w,h):
    '''
    随机裁剪
    :param img: 待裁剪图像
    :param w: 要切多宽
    :param h: 要切多高
    :return: 返回裁剪之后的图像
    '''
    start_x = np.random.randint(0,img.shape[1]-w)
    start_y = np.random.randint(0,img.shape[0]-h)

    new_img = img[start_y:start_y+h,start_x:start_x+w]
    return new_img

def center_crop(img,w,h):
    start_x = int(img.shape[1] / 2) - int(w / 2)
    start_y = int(img.shape[0] / 2) - int(h / 2)
    new_img = img[start_y:start_y+h,start_x:start_x+w]
    return new_img

if __name__ == '__main__':
    img = cv2.imread('../data/banana_1.png')
    cv2.imshow('img',img)
    #随机裁剪
    res = random_crop(img,200,200)
    cv2.imshow('random',res)
    #中心裁剪
    center = center_crop(img,200,200)
    cv2.imshow('center',center)

    cv2.waitKey()
    cv2.destroyAllWindows()





