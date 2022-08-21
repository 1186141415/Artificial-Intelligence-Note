'''
图像的卷积示例
'''
from scipy import misc  # 加载图像
import scipy.ndimage as sn  # 加载图像
from scipy import signal  # 卷积
import numpy as np
import matplotlib.pyplot as plt

img = misc.imread('../data/zebra.png', flatten=True)
# print(img.shape)

# 卷积核
flt_x = np.array([[-1, 0, 1],
                  [-2, 0, 2],
                  [-1, 0, 1]])
flt_y = np.array([[1, 2, 1],
                  [0, 0, 0],
                  [-1, -2, -1]])
# 卷积
res_x = signal.convolve2d(img,  # 原始图像
                          flt_x,  # 卷积核
                          mode='same',  # 卷积方式
                          boundary='symm')  # 边沿处理方式

res_y = signal.convolve2d(img,  # 原始图像
                          flt_y,  # 卷积核
                          mode='same',  # 卷积方式
                          boundary='symm')  # 边沿处理方式

plt.figure('Conv')
plt.subplot(1,3,1)
plt.imshow(img,cmap='gray')
plt.xticks([])
plt.yticks([])
#--------------
plt.subplot(1,3,2)
plt.imshow(res_x.astype('int32'),cmap='gray')
plt.xticks([])
plt.yticks([])
#--------------
plt.subplot(1,3,3)
plt.imshow(res_y.astype('int32'),cmap='gray')
plt.xticks([])
plt.yticks([])

plt.show()





