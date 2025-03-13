#!/usr/bin/env python3

import os
import rospy
import cv2
import torch
import numpy as np
from cv_bridge import CvBridge
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage
from duckietown_msgs.msg import Twist2DStamped
from robomimic.utils import file_utils as FileUtils
from robomimic.utils import torch_utils as TorchUtils

class InferenceNode(DTROS):
    def __init__(self, node_name):
        super(InferenceNode, self).__init__(node_name=node_name, node_type=NodeType.PERCEPTION)

        # Load environment variables
        self.vehicle_name = os.getenv("VEHICLE_NAME", "default_vehicle")
        self.camera_topic = f"/{self.vehicle_name}/camera_node/image/compressed"
        self.cmd_topic = f"/{self.vehicle_name}/car_cmd_switch_node/cmd"

        # Load the policy
        self.ckpt_path = str(os.getcwd()) + "/packages/robomimic_control/weights/20250312150115/models/model_epoch_20.pth"
        self.device = TorchUtils.get_torch_device(try_to_use_cuda=True)
        self.policy, _ = FileUtils.policy_from_checkpoint(ckpt_path=self.ckpt_path, device=self.device, verbose=True)

        # Initialize OpenCV bridge
        self.bridge = CvBridge()

        # ROS Subscribers & Publishers
        self.sub_camera = rospy.Subscriber(self.camera_topic, CompressedImage, self.process_image, queue_size=1)
        self.pub_cmd = rospy.Publisher(self.cmd_topic, Twist2DStamped, queue_size=1)

        self.log("InferenceNode initialized.")

    def process_image(self, msg):
        """Processes an image from the camera, runs inference, and sends velocity commands."""
        try:
            # Convert image message to OpenCV format
            np_arr = np.frombuffer(msg.data, np.uint8)
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            # Preprocess image for inference
            obs_dict = self.prepare_observation(image)

            # Get action from policy
            action = self.get_policy_action(obs_dict)

            # Publish velocity command
            self.publish_velocity_command(action)

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

    def publish_velocity_command(self, action):
        """Publishes the action as a velocity command."""
        cmd_msg = Twist2DStamped()
        cmd_msg.v = float(action[0])  # Linear velocity
        cmd_msg.omega = float(action[1])  # Angular velocity
        self.log(f"Velocity: {cmd_msg.v}, Omega: {cmd_msg.omega}")
        self.pub_cmd.publish(cmd_msg)

    def on_shutdown(self):
        """Stops the robot on shutdown."""
        stop_cmd = Twist2DStamped(v=0, omega=0)
        self.pub_cmd.publish(stop_cmd)

if __name__ == "__main__":
    node = InferenceNode(node_name="inference_node")
    rospy.spin()
