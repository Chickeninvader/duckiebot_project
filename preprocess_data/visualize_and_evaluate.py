import os
import json

from cv_bridge import CvBridge
import cv2
import torch
import rosbag
import datetime
import numpy as np
import imageio
import matplotlib.pyplot as plt
from collections import deque
from robomimic.utils import file_utils as FileUtils
from robomimic.utils import torch_utils as TorchUtils

# Configuration
bag_file = "record/lab_record/human_control_light_50/chicinvabot_2025-03-12-18-50-27.bag"  # Set your input bag file here
base_path = "../robomimic_project/bc_trained_models/test/20250312150115/"
ckpt_path = os.path.join(base_path, "models/model_epoch_20.pth")
video_output_path = os.path.join(base_path, "videos")

# ROS topics
image_topic = "/chicinvabot/camera_node/image/compressed"
cmd_topic = "/chicinvabot/car_cmd_switch_node/cmd"

# Load inference model
device = TorchUtils.get_torch_device(try_to_use_cuda=True)
policy, _ = FileUtils.policy_from_checkpoint(ckpt_path=ckpt_path, device=device, verbose=True)

# Process ROS bag
bag = rosbag.Bag(bag_file)
bridge = CvBridge()
timestamps, ground_truth_actions, predicted_actions = [], [], []
action_queue = deque()
latest_action = [0.0, 0.0]  # vel, omega
i = 0

for topic, msg, t in bag.read_messages(topics=[image_topic, cmd_topic]):
    timestamp = t.to_sec()

    if topic == cmd_topic:
        latest_action = [round(msg.v, 3), round(msg.omega, 3)]  # Use vel and omega
        action_queue.append((timestamp, latest_action))

    elif topic == image_topic:
        while action_queue and action_queue[0][0] < timestamp - 0.1:
            action_queue.popleft()

        action = latest_action if action_queue else [0.0, 0.0]
        timestamps.append(timestamp)
        ground_truth_actions.append(action)

        # Prepare input for inference
        frame = bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
        frame = cv2.resize(frame, (640, 480)).astype(np.float32) / 255.0
        frame = np.transpose(frame, (2, 0, 1))  # (C, H, W)

        obs_dict = {"observation": torch.tensor(frame).unsqueeze(0).to(device)}
        pred_action = policy.policy.get_action(obs_dict)[0].cpu().detach().numpy()
        predicted_actions.append(pred_action.tolist())

        i += 1
        print(f"Processed frame {i}")

bag.close()

# Convert to NumPy
timestamps = np.array(timestamps)
ground_truth_actions = np.array(ground_truth_actions)  # (N, 2) -> (vel, omega)
predicted_actions = np.array(predicted_actions)  # (N, 2) -> (vel, omega)


# Compute trajectory from velocity and angular velocity
def compute_trajectory(actions, timestamps):
    x, y, theta = 0, 0, 0  # Initial position
    trajectory = [(x, y)]

    for i in range(1, len(actions)):
        dt = timestamps[i] - timestamps[i - 1]
        v, omega = actions[i]

        # Update state
        x += v * np.cos(theta) * dt
        y += v * np.sin(theta) * dt
        theta += omega * dt
        trajectory.append((x, y))

    return np.array(trajectory)


# Compute both trajectories
gt_trajectory = compute_trajectory(ground_truth_actions, timestamps)
pred_trajectory = compute_trajectory(predicted_actions, timestamps)

# Plot the trajectories
plt.figure(figsize=(8, 6))
plt.plot(gt_trajectory[:, 0], gt_trajectory[:, 1], label="Ground Truth", linestyle="dashed", color="blue")
plt.plot(pred_trajectory[:, 0], pred_trajectory[:, 1], label="Prediction", linestyle="solid", color="red")
plt.scatter(gt_trajectory[0, 0], gt_trajectory[0, 1], marker="o", color="green", label="Start Position")
plt.xlabel("X Position (m)")
plt.ylabel("Y Position (m)")
plt.legend()
plt.title("2D Trajectory Comparison: Ground Truth vs Prediction")
plt.grid()
plt.savefig(os.path.join(video_output_path, "trajectory_comparison.png"))
plt.show()

print(f"Trajectory visualization saved to {video_output_path}")
