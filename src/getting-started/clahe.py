import cv2     
import matplotlib.pyplot as plt 

gridsize = 8

orig = cv2.imread('under.jpg')

lab = cv2.cvtColor(orig, cv2.COLOR_BGR2LAB)
lab_planes = cv2.split(lab)
clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(gridsize,gridsize))
lab_planes[0] = clahe.apply(lab_planes[0])
lab = cv2.merge(lab_planes)
bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

cv2.imshow("image", orig)
cv2.imshow("lab", bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()