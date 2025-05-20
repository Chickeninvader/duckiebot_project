import os
import cv2
import torch
import rosbag
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from cv_bridge import CvBridge
from robomimic.utils import file_utils as FileUtils
from robomimic.utils import torch_utils as TorchUtils

# Configuration
bag_file = "old_record/lab_record/human_control_light_50/chicinvabot_2025-03-12-18-50-27.bag"

# Define the folder containing all base paths
base_folder = "../robomimic_project/bc_trained_models"

# Get all valid subdirectories that contain 'models/model_epoch_20.pth'
base_paths = []
for subdir in os.listdir(base_folder):
    subdir_path = os.path.join(base_folder, subdir)
    if os.path.isdir(subdir_path):
        # Look for deeper subdirectories inside this subdir
        for deeper_subdir in os.listdir(subdir_path):
            deeper_subdir_path = os.path.join(subdir_path, deeper_subdir)
            if os.path.isdir(deeper_subdir_path):
                ckpt_path = os.path.join(deeper_subdir_path, "models", "model_epoch_20.pth")
                if os.path.exists(ckpt_path):
                    base_paths.append(deeper_subdir_path)

# Ensure at least one valid path is found
if not base_paths:
    raise FileNotFoundError("No valid model checkpoint directories found.")

video_output_path = os.path.join(base_paths[0], "videos")
OMEGA_MAX = 8.0
GAIN = 1.0

# ROS topics
image_topic = "/chicinvabot/camera_node/image/compressed"
cmd_topic = "/chicinvabot/car_cmd_switch_node/cmd"

# Load inference models
models = {}
device = TorchUtils.get_torch_device(try_to_use_cuda=True)
for base_path in base_paths:
    ckpt_path = os.path.join(base_path, "models/model_epoch_20.pth")
    models[base_path], _ = FileUtils.policy_from_checkpoint(ckpt_path=ckpt_path, device=device, verbose=True)

# Process ROS bag
bag = rosbag.Bag(bag_file)
bridge = CvBridge()
timestamps, ground_truth_actions = [], []
predicted_actions_all = {base_path: [] for base_path in base_paths}
action_queue = deque()
latest_action = [0.0, 0.0]  # vel, omega
i = 0

for topic, msg, t in bag.read_messages(topics=[image_topic, cmd_topic]):
    timestamp = t.to_sec()

    if topic == cmd_topic:
        latest_action = [round(msg.v, 3), round(msg.omega, 3)]
        action_queue.append((timestamp, latest_action))

    elif topic == image_topic:
        while action_queue and action_queue[0][0] < timestamp - 0.1:
            action_queue.popleft()

        action = latest_action if action_queue else [0.0, 0.0]
        timestamps.append(timestamp)
        ground_truth_actions.append(action)

        # Prepare input for inference
        frame = bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
        frame_resized = cv2.resize(frame, (640, 480))
        frame_norm = frame_resized.astype(np.float32) / 255.0
        frame_norm = np.transpose(frame_norm, (2, 0, 1))
        obs_dict = {"observation": torch.tensor(frame_norm).unsqueeze(0).to(device)}

        # Run inference for each model
        for base_path, policy in models.items():
            pred_action = policy.policy.get_action(obs_dict)[0].cpu().detach().numpy()
            predicted_actions_all[base_path].append(pred_action.tolist())

        # Overlay text
        gt_text = f"GT: Vel={action[0]:.2f}, Omega={action[1]:.2f}"
        cv2.putText(frame_resized, gt_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        y_offset = 60
        for base_path in base_paths:
            pred_action = predicted_actions_all[base_path][-1]
            pred_text = f"{os.path.basename(os.path.dirname(base_path))}: Vel={(pred_action[0] * GAIN):.2f}, Omega={(pred_action[1] * OMEGA_MAX):.2f}"
            cv2.putText(frame_resized, pred_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 1)
            y_offset += 30

        # Display the frame
        cv2.imshow("Trajectory Comparison", frame_resized)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

        i += 1
        print(f"Processed frame {i}")

bag.close()
cv2.destroyAllWindows()

# Ensure video output directory exists
os.makedirs(video_output_path, exist_ok=True)

# Extract omega and velocity values for plotting
timestamps = np.array(timestamps)
gt_omegas = np.array([a[1] for a in ground_truth_actions])
gt_velocities = np.array([a[0] for a in ground_truth_actions])

plt.figure(figsize=(10, 5))
plt.plot(timestamps, gt_omegas, label="Ground Truth Omega", linestyle="dashed", color="blue")
for base_path in base_paths:
    pred_omegas = np.array([a[1] * OMEGA_MAX for a in predicted_actions_all[base_path]])
    plt.plot(timestamps, pred_omegas, label=f"Predicted Omega ({os.path.basename(os.path.dirname(base_path))})", linestyle="solid")
plt.xlabel("Time (s)")
plt.ylabel("Omega (rad/s)")
plt.legend()
plt.title("Omega Over Time: Ground Truth vs Prediction")
plt.grid()
plt.savefig(os.path.join(video_output_path, "omega_comparison.png"))
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(timestamps, gt_velocities, label="Ground Truth Velocity", linestyle="dashed", color="blue")
for base_path in base_paths:
    pred_velocities = np.array([a[0] * GAIN for a in predicted_actions_all[base_path]])
    plt.plot(timestamps, pred_velocities, label=f"Predicted Velocity ({os.path.basename(os.path.dirname(base_path))})", linestyle="solid")
plt.xlabel("Time (s)")
plt.ylabel("Velocity (m/s)")
plt.legend()
plt.title("Velocity Over Time: Ground Truth vs Prediction")
plt.grid()
plt.savefig(os.path.join(video_output_path, "velocity_comparison.png"))
plt.show()

print(f"Graphs saved to {video_output_path}")
