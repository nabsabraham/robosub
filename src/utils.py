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

def clahe(img, gridsize=8)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(gridsize,gridsize))
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return bgr

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
    # img : floating point
    # k : gaussian kernel for smoothing

    smooth = cv2.GaussianBlur(img, (k,k), 0)
    sharp = img - smooth*img
    sharp_norm = sharp/np.max(sharp)
    return (img+sharp_norm)/2

def whitebalance(img, scale=2):
    # white balancing method from transactions paper
    wb = ancuti_wb(img, scale)
    wb_int = (wb*255).astype("uint8")
    sharp = unsharp_mask(wb, 5)
    gamma = adjust_gamma(wb_int,gamma=0.5)
    out = gamma + sharp
    out = (out.astype(float)/np.max(out))
    return np.uint8(out*255)