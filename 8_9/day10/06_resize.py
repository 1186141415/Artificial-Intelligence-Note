'''
图像的缩放
缩小
放大：最近邻插值法，双线性插值法
'''
import cv2

img = cv2.imread('../data/Linus.png')
cv2.imshow('img',img)

h,w = img.shape[:2]
print(f'h:{h},w:{w}')

#缩小
dst_size = (int(w/2),int(h/2))
resized = cv2.resize(img,dst_size)
cv2.imshow('resized',resized)

#放大
dst_size = (200,300)
resized = cv2.resize(img,dst_size,
                     interpolation=cv2.INTER_NEAREST)
cv2.imshow('NEAREST',resized)

dst_size = (200,300)
resized = cv2.resize(img,dst_size,
                     interpolation=cv2.INTER_LINEAR)
cv2.imshow('LINEAR',resized)


cv2.waitKey()
cv2.destroyAllWindows()





