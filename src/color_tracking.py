import cv2 
import numpy as np
from utils import whitebalance, clahe

#device = cv2.VideoCapture(0)	
device = cv2.VideoCapture('video/gate3.mp4')

def clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(3,3))
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return bgr

while True:
    ret, frame = device.read()
    frame_corrected = whitebalance(frame, 2)
    frame_corrected = clahe(frame_corrected)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv_c = cv2.cvtColor(frame_corrected, cv2.COLOR_BGR2HSV)

    lower = np.array([5, 50, 50])
    upper = np.array([15, 255, 255])

    mask_c = cv2.inRange(hsv_c, lower, upper)
    mask = cv2.inRange(hsv, lower, upper)
    out = np.hstack((frame, frame_corrected))
    masks = np.hstack((mask, mask_c))

    cv2.imshow('Results', out)
    cv2.imshow("Masked", masks)
       
    # escape key
    if cv2.waitKey(1) == 27:    
        break

device.release()
cv2.destroyAllWindows()