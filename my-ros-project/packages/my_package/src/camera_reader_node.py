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
        self._vehicle_name = os.environ.get('VEHICLE_NAME', 'default_vehicle')
        self._camera_topic = f"/agent/object_detection_node/image/compressed"
        # bridge between OpenCV and ROS
        self._bridge = CvBridge()
        # create OpenCV window
        self._window = "camera-reader"
        cv2.namedWindow(self._window, cv2.WINDOW_AUTOSIZE)
        # construct subscriber
        self.sub = rospy.Subscriber(self._camera_topic, CompressedImage, self.callback)

    def callback(self, msg):
        try:
            # convert JPEG bytes to CV image
            image = self._bridge.compressed_imgmsg_to_cv2(msg)
            # display frame using OpenCV
            cv2.imshow(self._window, image)
            cv2.waitKey(1)  # Necessary to refresh the OpenCV window
        except Exception as e:
            rospy.logerr(f"Error converting or displaying image: {e}")

    def on_shutdown(self):
        # Close OpenCV windows on shutdown
        cv2.destroyAllWindows()

if __name__ == '__main__':
    # create the node
    node = CameraReaderNode(node_name='camera_reader_node')
    # register shutdown hook
    rospy.on_shutdown(node.on_shutdown)
    # keep spinning
    rospy.spin()
