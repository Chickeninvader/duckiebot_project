#!/usr/bin/env python3

import os
import rospy
import cv2
import torch
import numpy as np
from cv_bridge import CvBridge
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage
from duckietown_msgs.msg import WheelsCmdStamped
from robomimic.utils import file_utils as FileUtils
from robomimic.utils import torch_utils as TorchUtils

class InferenceNode(DTROS):
    def __init__(self, node_name):
        super(InferenceNode, self).__init__(node_name=node_name, node_type=NodeType.PERCEPTION)

        # Load environment variables
        self.vehicle_name = os.getenv("VEHICLE_NAME", "default_vehicle")
        self.camera_topic = f"/{self.vehicle_name}/camera_node/image/compressed"
        self.wheels_topic = f"/{self.vehicle_name}/wheels_driver_node/wheels_cmd"

        # Load the policy
        self.ckpt_path = str(os.getcwd()) + "/packages/robomimic_control/weights/training_run_20250303/20250303025924/models/model_epoch_30.pth"
        self.device = TorchUtils.get_torch_device(try_to_use_cuda=True)
        self.policy, _ = FileUtils.policy_from_checkpoint(ckpt_path=self.ckpt_path, device=self.device, verbose=True)

        # Initialize OpenCV bridge
        self.bridge = CvBridge()

        # ROS Subscribers & Publishers
        self.sub_camera = rospy.Subscriber(self.camera_topic, CompressedImage, self.process_image, queue_size=1)
        self.pub_wheels = rospy.Publisher(self.wheels_topic, WheelsCmdStamped, queue_size=1)

        self.log("InferenceNode initialized.")

    def process_image(self, msg):
        """Processes an image from the camera, runs inference, and sends wheel commands."""
        try:
            # Convert image message to OpenCV format
            np_arr = np.frombuffer(msg.data, np.uint8)
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            # Preprocess image for inference
            obs_dict = self.prepare_observation(image)

            # Get action from policy
            action = self.get_policy_action(obs_dict)

            # Publish wheel command
            self.publish_wheel_command(action)

        except Exception as e:
            self.log(f"Error in process_image: {e}", type="error")

    def prepare_observation(self, image):
        """Prepares an image for inference by converting it to a tensor."""
        image = cv2.resize(image, (640, 480))  # Resize to model input size
        image = np.transpose(image, (2, 0, 1))  # Convert HWC to CHW
        image = image / 255.0  # Normalize
        image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).to(self.device)
        return {"observation": image_tensor}

    def get_policy_action(self, obs_dict):
        """Runs inference and returns the action."""
        with torch.no_grad():
            action = self.policy.policy.get_action(obs_dict)
        return action.cpu().numpy().flatten()

    def publish_wheel_command(self, action):
        """Publishes the action as wheel commands."""
        wheel_cmd = WheelsCmdStamped()
        wheel_cmd.vel_left = float(action[0])  
        wheel_cmd.vel_right = float(action[1]) 
        self.log(f"left: {wheel_cmd.vel_left} right: {wheel_cmd.vel_right}")
        self.pub_wheels.publish(wheel_cmd)

    def on_shutdown(self):
        """Stops the robot on shutdown."""
        stop_cmd = WheelsCmdStamped(vel_left=0, vel_right=0)
        self.pub_wheels.publish(stop_cmd)

if __name__ == "__main__":
    node = InferenceNode(node_name="inference_node")
    rospy.spin()

