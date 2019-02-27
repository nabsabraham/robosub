#!/usr/bin/env python
# https://gist.github.com/jkarimi91/d393688c4d4cdb9251e3f939f138876e
'''
proof of concept - the code can read an image from the web or dir and run recognition on it 
no ROS as of yet
'''

import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import sys
import torch
import torchvision.models as models
import matplotlib.pyplot as plt
from torchvision import transforms, utils
from PIL import Image as PILImage
from PIL import ImageFont, ImageDraw
from torch.autograd import Variable
from torchsummary import summary
import requests
import io
import time
import numpy as np

# get the imagenet labels that vgg was trained on
LABELS_URL = 'https://s3.amazonaws.com/outcome-blog/imagenet/labels.json' 
response = requests.get(LABELS_URL)                                 #make an HTTP request and store the response
labels = {int(key):value for key, value in response.json().items()} #create a dict with all the labels and their number encoding
#print(labels)

IMG_URL = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg"
response = requests.get(IMG_URL)
img = PILImage.open(io.BytesIO(response.content))  #read the bytes and store as image 

imsize=224
transformations = transforms.Compose([transforms.Resize(imsize), 
                                    transforms.ToTensor(), 
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

def process_img(img):
    img = transformations(img)
    img = Variable(img, requires_grad=True)         #PyTorch expects model inputs to be variables (I think this changed in the new torch)
    img = img.unsqueeze(0)                          #add an extra axis at axis=0
    return img

'''
#get image from dir
img = PILImage.open('images/flower.jpg')
'''
orig_img = img
start = time.time()
img = process_img(img)
vgg = models.vgg16(pretrained=True)
vgg.eval()
log_probs = vgg(img)
#probs = torch.exp(log_probs)
top_prob, top_class = log_probs.topk(1, dim=1)
top_class = top_class.data.numpy()

end = time.time()
label = labels[int(top_class)]
print('Time:', end-start)


cv_image = cv2.cvtColor(np.array(orig_img), cv2.COLOR_RGB2BGR)
font = cv2.FONT_HERSHEY_SIMPLEX

cv2.putText(cv_image, str(label),(10,350), font, 1,(255,0,0),2,cv2.LINE_AA)
cv2.imshow("image", cv_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
