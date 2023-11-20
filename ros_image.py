#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

def image_callback(msg):
    bridge = CvBridge()
    cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
    cv2.imwrite('rect_color_image.jpg', cv_image)
    rospy.signal_shutdown("Image saved")

def save_one_image():
    rospy.init_node('save_one_image')
    rospy.Subscriber('/camera/image_rect_color', Image, image_callback)
    rospy.spin()

if __name__ == '__main__':
    save_one_image()