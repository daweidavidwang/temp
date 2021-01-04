import cv2 as cv
import numpy as np

img = np.ones((224,224,3),np.uint8)*0

points_list = [(54,54)]

point_size = 1
point_color = (225, 225, 225) # BGR
thickness = 4 #  0 、4、8

for point in points_list:
    cv.circle(img, point, point_size, point_color, thickness)

img_gray = cv.cvtColor(img,cv.COLOR_RGB2GRAY)
img_new = cv.resize(img_gray,(112,112))

cv.imshow("img",img_new)

cv.waitKey()    
cv.destroyAllWindows() 