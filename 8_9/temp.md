@[toc](目录)
# 机器视觉基础
```python
'''
读取图像，显示图像，保存图像
'''
import cv2
import matplotlib.pyplot as plt

#加载图像
img = cv2.imread('../data/lena.jpg')
# print(img.shape) #(300, 300, 3) 3是三通道BGR彩色图片
# print(type(img)) #数据是class 'numpy.ndarray
cv2.imshow('img',img)
# cv2.imshow('img2',img)

#保存图像
cv2.imwrite('lena_new.jpg',img)

#主动进入阻塞等待，等待用户按下某个按键，停止阻塞
cv2.waitKey()
cv2.destroyAllWindows()
```

```python
'''
彩色图像转为灰度图像（不可逆）
转换色彩空间
彩色色彩空间：默认为BGR   灰度：GRAY
'''
import cv2

img = cv2.imread('../data/dog2.png') #0:灰度  1：彩色
cv2.imshow('img',img)
#BGR-->GRAY
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow('gray',gray)

cv2.waitKey()
cv2.destroyAllWindows()
```

```python
'''对彩色图像的通道进行操作'''
import cv2

img = cv2.imread('../data/opencv2.png') #BGR
cv2.imshow('img',img)

b = img[:,:,0] #提出来蓝色，这样蓝色就成了单通道颜色，就是灰度图
cv2.imshow('b',b)

img[:,:,0] = 0
cv2.imshow('b0',img)

img[:,:,1] = 0
cv2.imshow('b0-g0',img)

cv2.waitKey()
cv2.destroyAllWindows()
```

```python
'''
灰度图像直方图均衡化
'''
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('../data/sunrise.jpg', 0)
cv2.imshow('img', img)

# 直方图均衡化
res = cv2.equalizeHist(img)
cv2.imshow('res', res)

plt.figure('Hist')
plt.subplot(2, 1, 1)
plt.hist(img.ravel(),
         bins=256,
         range=[0, 256])

plt.subplot(2, 1, 2)
plt.hist(res.ravel(),
         bins=256,
         range=[0, 256])

plt.show()

cv2.waitKey()
cv2.destroyAllWindows()
```

```python
'''
彩色图像亮度通道直方图均衡化
'''
import cv2

img = cv2.imread('../data/sunrise.jpg')
cv2.imshow('img',img)
#BGR-->YUV
yuv = cv2.cvtColor(img,cv2.COLOR_BGR2YUV)
#亮度均衡化 y是亮度
yuv[...,0] = cv2.equalizeHist(yuv[...,0])

res = cv2.cvtColor(yuv,cv2.COLOR_YUV2BGR)
cv2.imshow('res',res)

cv2.waitKey()
cv2.destroyAllWindows()
```