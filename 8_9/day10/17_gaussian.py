'''
自己构建高斯分布卷积核，实现卷积
'''
import cv2
import numpy as np

img = cv2.imread('../data/salt.jpg',0)
cv2.imshow('img',img)

gaussian_blur = np.array([
    [1, 4, 7, 4, 1],
    [4, 16, 26, 16, 4],
    [7, 26, 41, 26, 7],
    [4, 16, 26, 16, 4],
    [1, 4, 7, 4, 1]], np.float32) / 273

res = cv2.filter2D(img,-1,gaussian_blur)
cv2.imshow('res',res)

cv2.waitKey()
cv2.destroyAllWindows()
