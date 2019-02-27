import cv2
import numpy as np
import matplotlib.pyplot as plt 

img = cv2.imread('images/flower.jpg')
lower = np.array([0,0,0], dtype="uint8")
upper = np.array([50,56,100], dtype="uint8")

mask = cv2.inRange(img, lower, upper)
plt.imshow(mask)
plt.show()