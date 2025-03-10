import rosbag
import numpy as np
import cv2
import h5py
import os
import json
import datetime
from cv_bridge import CvBridge
from collections import deque

# Define input folder containing ROS bag files and output HDF5 files
bag_folder = "sim_record"  # Change this to your actual folder
robot_name = "vchicinvabot"
output_dir = "record/converted_standard"
os.makedirs(output_dir, exist_ok=True)
hdf5_path_old = os.path.join(output_dir, "all_demos_old.hdf5")  # Old wheel topic
hdf5_path_new = os.path.join(output_dir, "all_demos_new.hdf5")  # New wheel topic

# Topics of interest
image_topic = f"/{robot_name}/camera_node/image/compressed"
old_cmd_topic = f"/{robot_name}/wheels_driver_node/wheels_cmd"
new_cmd_topic = f"/{robot_name}/car_cmd_switch_node/cmd"

# Initialize HDF5 files
for hdf5_path in [hdf5_path_old, hdf5_path_new]:
    with h5py.File(hdf5_path, "w") as f:
        f.create_group("data")

# Process each bag file
bag_files = sorted([f for f in os.listdir(bag_folder) if f.endswith(".bag")])

for idx, bag_file in enumerate(bag_files, start=1):
    bag_path = os.path.join(bag_folder, bag_file)
    print(f"Processing {bag_file} as demo_{idx}...")

    bag = rosbag.Bag(bag_path)
    bridge = CvBridge()

    # Data structures
    timestamps = []
    images = []
    actions_old = []
    actions_new = []

    latest_action_old = [0.0, 0.0]  # Default for old topic
    latest_action_new = [0.0, 0.0]  # Default for new topic (gamma, vel_wheel)
    action_queue_old = deque()
    action_queue_new = deque()

    for topic, msg, t in bag.read_messages(topics=[image_topic, old_cmd_topic, new_cmd_topic]):
        timestamp = t.to_sec()

        if topic == old_cmd_topic:
            latest_action_old = [round(msg.vel_left, 3), round(msg.vel_right, 3)]
            action_queue_old.append((timestamp, latest_action_old))

        elif topic == new_cmd_topic:
            latest_action_new = [round(msg.v, 3), round(msg.omega, 3)]
            action_queue_new.append((timestamp, latest_action_new))

        elif topic == image_topic:
            frame = bridge.compressed_imgmsg_to_cv2(msg, "bgr8")

            while action_queue_old and action_queue_old[0][0] < timestamp - 0.1:
                action_queue_old.popleft()

            while action_queue_new and action_queue_new[0][0] < timestamp - 0.1:
                action_queue_new.popleft()

            action_old = latest_action_old if action_queue_old else [0.0, 0.0]
            action_new = latest_action_new if action_queue_new else [0.0, 0.0]

            timestamps.append(timestamp)
            images.append(frame)
            actions_old.append(action_old)
            actions_new.append(action_new)

    bag.close()

    # Convert to NumPy arrays
    timestamps = np.array(timestamps, dtype=np.float64)
    images = np.array(images, dtype=np.uint8)
    actions_old = np.array(actions_old, dtype=np.float32)
    actions_new = np.array(actions_new, dtype=np.float32)

    num_samples = len(actions_old) - 1
    if num_samples <= 0:
        print(f"Skipping {bag_file} (no valid samples).")
        continue

    # Write to old topic HDF5 file
    with h5py.File(hdf5_path_old, "a") as f:
        grp = f["data"].create_group(f"demo_{idx}")
        grp.attrs["num_samples"] = num_samples
        grp.create_dataset("obs/observation", data=images[:-1])
        grp.create_dataset("next_obs/observation", data=images[1:])
        grp.create_dataset("actions", data=actions_old[:-1])
        grp.create_dataset("rewards", data=np.zeros(num_samples, dtype=np.float64))
        dones = np.zeros(num_samples, dtype=np.int64)
        dones[-1] = 1
        grp.create_dataset("dones", data=dones)

    # Write to new topic HDF5 file
    with h5py.File(hdf5_path_new, "a") as f:
        grp = f["data"].create_group(f"demo_{idx}")
        grp.attrs["num_samples"] = num_samples
        grp.create_dataset("obs/observation", data=images[:-1])
        grp.create_dataset("next_obs/observation", data=images[1:])
        grp.create_dataset("actions", data=actions_new[:-1])
        grp.create_dataset("rewards", data=np.zeros(num_samples, dtype=np.float64))
        dones = np.zeros(num_samples, dtype=np.int64)
        dones[-1] = 1
        grp.create_dataset("dones", data=dones)

print(f"Processing complete. Saved old topic data to {hdf5_path_old} and new topic data to {hdf5_path_new}")
