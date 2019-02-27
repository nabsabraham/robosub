import cv2 
import numpy as np
from utils import whitebalance, clahe, getContours, drawContours

#device = cv2.VideoCapture(0)	
device = cv2.VideoCapture('video/gate2.mp4')

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

    contours = getContours(mask)
    if len(contours) > 0: 
        drawContours(frame, contours)

    cv2.imshow('Original', frame)
   
    #cv2.imshow('Results', out)
    #cv2.imshow("Masked", masks)
       
    # escape key
    if cv2.waitKey(1) == 27:    
        break

device.release()
cv2.destroyAllWindows()