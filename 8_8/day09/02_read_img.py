'''
读取图像，显示图像，保存图像
'''
import cv2

#加载图像
img = cv2.imread('../data/lena.jpg')
# print(img.shape)
# print(type(img))
cv2.imshow('img',img)
# cv2.imshow('img2',img)

#保存图像
cv2.imwrite('lena_new.jpg',img)

#主动进入阻塞等待，等待用户按下某个按键，停止阻塞
cv2.waitKey()
cv2.destroyAllWindows()




