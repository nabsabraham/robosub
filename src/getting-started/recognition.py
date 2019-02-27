#!/usr/bin/env python
from __future__ import print_function

import roslib
#roslib.load_manifest('my_package')
import sys
import rospy
import cv2
import requests
from std_msgs.msg import String
from sensor_msgs.msg import Image
from PIL import Image as PILImage
from cv_bridge import CvBridge, CvBridgeError
from torchvision import models, transforms, utils
from torch.autograd import Variable

class image_converter:

    def __init__(self):
        self.label_pub = rospy.Publisher("image_label",String, queue_size=1000)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/usb_cam/image_raw", Image, self.callback)
        self.vgg = models.vgg16(pretrained=True)
        self.label_url = 'https://s3.amazonaws.com/outcome-blog/imagenet/labels.json'
        self.response = requests.get(self.label_url)                                 
        self.labels = {int(key):value for key, value in self.response.json().items()}
        self.transformations = transformations = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224), 
                                    transforms.ToTensor(), 
                                    transforms.Normalize(   mean=[0.485, 0.456, 0.406],
                                                            std=[0.229, 0.224, 0.225])] )
    def process_img(self, cv_image):
        pil_img = PILImage.fromarray(cv_image)
        img = self.transformations(pil_img)
        img = Variable(img, requires_grad=True)         
        img = img.unsqueeze(0)                          
        preds = self.vgg(img)
        top_prob, top_class = preds.topk(1, dim=1)
        top_class = top_class.data.numpy()
        label = self.labels[int(top_class)]
        return str(label)

    def callback(self,data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        class_label = self.process_img(cv_image)
        cv2.imshow("Image window", cv_image)
        cv2.waitKey(5)
        try:
            rospy.loginfo(class_label)
            self.label_pub.publish(class_label)
            rospy.sleep(1)
            
        except CvBridgeError as e:
            print(e)

def main(args):
    print('in the main')
    ic = image_converter()
    rospy.init_node('image_converter', anonymous=False)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)