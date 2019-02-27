'''
Color Balance and Fusion for Underwater Image Enhancement
I haven't implemented the full algorithm - just kinda fudged the fusion part 
gamma correction from: https://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction/

output image in the range 0-1
'''

import cv2 
import numpy as np
import matplotlib.pyplot as plt 

def ancuti_wb(img, scale):
    img = img.astype(float)
    img = img/255
    avg_r = np.average(img[:,:,2])
    avg_g = np.average(img[:,:,1])
    Ig, Ir = img[:,:,1], img[:,:,2]
    img[:,:,2] = Ir + scale*(avg_g - avg_r)* (1-Ir)*Ig
    return img

def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)])
    out= cv2.LUT(image, table)
    out = out.astype(float)/np.max(out)
    return out

def unsharp_mask(img, k):
    '''
    img : floating point
    k : gaussian kernel for smoothing
    '''
    smooth = cv2.GaussianBlur(img, (k,k), 0)
    sharp = img - smooth*img
    sharp_norm = sharp/np.max(sharp)
    return (img+sharp_norm)/2

def wb_corrected(img, scale):
    wb = ancuti_wb(img, scale)
    wb_int = (wb*255).astype("uint8")
    sharp = unsharp_mask(wb, 5)
    gamma = adjust_gamma(wb_int,gamma=0.5)
    out = gamma + sharp
    out = (out.astype(float)/np.max(out))
    return np.uint8(out*255)

if __name__=="__main__":
        
    img = cv2.imread('images/underwater.jpg')
    wb = wb_corrected(img)
    print(np.max(wb))
    cv2.imshow('orig', img)
    cv2.imshow("final",wb )

    cv2.waitKey(0)
    cv2.destroyAllWindows()