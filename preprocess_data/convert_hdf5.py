import rosbag
import numpy as np
import cv2
import h5py
import os
import json
import datetime
from cv_bridge import CvBridge
from collections import deque

# Define input folder containing ROS bag files and output HDF5 file
bag_folder = "/data/logs/home_record"  # Change this to your actual folder
output_dir = "converted_standard"
os.makedirs(output_dir, exist_ok=True)
hdf5_path = os.path.join(output_dir, "all_demos.hdf5")

# Topics of interest
image_topic = "/chicinvabot/camera_node/image/compressed"
cmd_topic = "/chicinvabot/wheels_driver_node/wheels_cmd"

# Initialize HDF5 file
with h5py.File(hdf5_path, "w") as f:
    # Create main data group
    grp = f.create_group("data")

    # Process each bag file in the folder
    bag_files = sorted([f for f in os.listdir(bag_folder) if f.endswith(".bag")])

    for idx, bag_file in enumerate(bag_files, start=1):
        bag_path = os.path.join(bag_folder, bag_file)
        print(f"Processing {bag_file} as demo_{idx}...")

        # ROS bag reader
        bag = rosbag.Bag(bag_path)
        bridge = CvBridge()

        # Storage lists
        timestamps = []
        images = []
        actions = []

        # Action buffer (deque for fast lookup)
        latest_action = [0.0, 0.0]  # Default action if no command exists
        action_queue = deque()

        for topic, msg, t in bag.read_messages(topics=[image_topic, cmd_topic]):
            timestamp = t.to_sec()

            if topic == cmd_topic:
                # Store the latest wheel command
                latest_action = [round(msg.vel_left, 3), round(msg.vel_right, 3)]
                action_queue.append((timestamp, latest_action))

            elif topic == image_topic:
                # Convert image message to OpenCV format
                frame = bridge.compressed_imgmsg_to_cv2(msg, "bgr8")

                # Find closest action timestamp (FIFO method)
                while action_queue and action_queue[0][0] < timestamp - 0.1:  # Allow small delay tolerance
                    action_queue.popleft()  # Remove outdated actions

                action = latest_action if action_queue else [0.0, 0.0]  # Default to zero if no recent action

                # Store data
                timestamps.append(timestamp)
                images.append(frame)
                actions.append(action)

        bag.close()

        # Convert to NumPy arrays
        timestamps = np.array(timestamps, dtype=np.float64)
        actions = np.array(actions, dtype=np.float32)
        images = np.array(images, dtype=np.uint8)  # Store images efficiently

        num_samples = len(actions) - 1  # Align with obs/next_obs

        if num_samples <= 0:
            print(f"Skipping {bag_file} (no valid samples).")
            continue

        # Create a new demonstration group for this bag
        demo_grp = grp.create_group(f"demo_{idx}")
        demo_grp.attrs["num_samples"] = num_samples

        # Assign observations
        obs_grp = demo_grp.create_group("obs")
        next_obs_grp = demo_grp.create_group("next_obs")

        obs_grp.create_dataset("observation", data=images[:-1])  # Remove last to align with next_obs
        next_obs_grp.create_dataset("observation", data=images[1:])  # Shifted by one step

        # Store states (image-based representation)
        demo_grp.create_dataset("states", data=images)  # States = raw observations

        # Store actions
        demo_grp.create_dataset("actions", data=actions[:-1])  # Remove last to align with obs

        # Store rewards (set to 0 for all steps)
        demo_grp.create_dataset("rewards", data=np.zeros(num_samples, dtype=np.float64))

        # Store dones (set last done to 1, others 0)
        dones = np.zeros(num_samples, dtype=np.int64)
        dones[-1] = 1  # Last timestep should be done
        demo_grp.create_dataset("dones", data=dones)

    # Add metadata attributes
    now = datetime.datetime.now()
    grp.attrs["date"] = f"{now.month}-{now.day}-{now.year}"
    grp.attrs["time"] = f"{now.hour}:{now.minute}:{now.second}"
    grp.attrs["repository_version"] = "1.0.0"  # Modify as needed
    grp.attrs["env"] = "robot_env"  # Modify based on environment

    # Example environment info
    env_info = {
        "controller": "velocity_control",
        "robot_type": "differential_drive",
        "additional_info": "Structured for Learning from Demonstration"
    }
    grp.attrs["env_info"] = json.dumps(env_info)

print(f"Processing complete. Saved all demos to {hdf5_path}")
