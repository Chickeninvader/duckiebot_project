#!/usr/bin/env python3

import os
import rospy
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage

import cv2
from cv_bridge import CvBridge

class CameraReaderNode(DTROS):

    def __init__(self, node_name):
        # initialize the DTROS parent class
        super(CameraReaderNode, self).__init__(node_name=node_name, node_type=NodeType.VISUALIZATION)
        
        # static parameters
        self._vehicle_name = os.environ['VEHICLE_NAME']
        self._camera_topic = f"/{self._vehicle_name}/camera_node/image/compressed"
        
        # bridge between OpenCV and ROS
        self._bridge = CvBridge()
        
        # create window
        self._window = "camera-reader"
        cv2.namedWindow(self._window, cv2.WINDOW_AUTOSIZE)
        
        # initialize variables for FPS calculation
        self._last_time = rospy.get_time()
        self._frame_count = 0
        self._fps = 0
        
        # construct subscriber
        self.sub = rospy.Subscriber(self._camera_topic, CompressedImage, self.callback)

    def callback(self, msg):
        # update the frame count
        self._frame_count += 1
        current_time = rospy.get_time()
        time_diff = current_time - self._last_time

        # calculate FPS every second
        if time_diff >= 1.0:
            self._fps = self._frame_count / time_diff
            self._frame_count = 0
            self._last_time = current_time

        # convert JPEG bytes to CV image
        image = self._bridge.compressed_imgmsg_to_cv2(msg)

        # overlay FPS on the image
        cv2.putText(
            image, 
            f"FPS: {self._fps:.2f}", 
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, 
            (255, 255, 255), 
            2
        )

        # display frame
        cv2.imshow(self._window, image)
        cv2.waitKey(1)

if __name__ == '__main__':
    # create the node
    node = CameraReaderNode(node_name='camera_reader_node')
    # keep spinning
    rospy.spin()
