# whitebalancing: https://stackoverflow.com/questions/46390779/automatic-white-balancing-with-grayworld-assumption
import cv2 
import numpy as np
import matplotlib.pyplot as plt 

def white_balance(img):
    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    return result

def ancuti_wb(img):
    img = img.astype(float)
    img = img/255
    avg_r = np.average(img[:,:,2])
    avg_g = np.average(img[:,:,1])
    Ig, Ir = img[:,:,1], img[:,:,2]
    img[:,:,2] = Ir + 2*(avg_g - avg_r)* (1-Ir)*Ig
    return img*255

img = cv2.imread('images/underwater.jpg')
output = white_balance(img)
output2 = ancuti_wb(img)

cv2.imshow("original", img)
final = np.hstack((output, output2.astype(np.uint8)))
cv2.imshow("final",final )

cv2.waitKey(0)
cv2.destroyAllWindows()