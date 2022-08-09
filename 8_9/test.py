import cv2

im = cv2.imread("../data/lena.jpg", 0)
cv2.imshow("orig", im)
cv2.waitKey()
cv2.destroyAllWindows()
