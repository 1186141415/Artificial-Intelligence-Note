'''
图像的相加
'''
import cv2

lena = cv2.imread('../data/lena.jpg',0)
lily = cv2.imread('../data/lily_square.png',0)
cv2.imshow('lena',lena)
cv2.imshow('lily',lily)

add = cv2.add(lena,lily)
cv2.imshow('add',add)

#按照权重进行相加
res = cv2.addWeighted(lena,0.5,
                      lily,0.5,
                      0) #亮度调节量
cv2.imshow('res',res)

cv2.waitKey()
cv2.destroyAllWindows()




