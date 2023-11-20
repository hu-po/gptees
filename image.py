#!/usr/bin/env python3
# encoding: utf-8
import argparse
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

argparser = argparse.ArgumentParser()
argparser.add_argument("--command", type=str, default="image.jpg")
args = argparser.parse_args()

def image_callback(msg):
    bridge = CvBridge()
    cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
    cv2.imwrite(args.command, cv_image)
    rospy.signal_shutdown("Image saved")

def save_one_image():
    rospy.init_node('save_one_image')
    rospy.Subscriber('/camera/image_rect_color', Image, image_callback)
    rospy.spin()

if __name__ == '__main__':
    save_one_image()