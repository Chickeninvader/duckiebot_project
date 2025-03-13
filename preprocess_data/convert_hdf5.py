import rosbag
import numpy as np
import cv2
import h5py
import os
import json
import datetime
from cv_bridge import CvBridge
from collections import deque

# Define input folders and output HDF5 file
base_folder = "record"
output_dir = "record/converted_standard"
os.makedirs(output_dir, exist_ok=True)

# User selection for processing mode
process_sim = False  # Set to False to exclude simulation data
process_lab = True  # Set to False to exclude real-world data
process_human = False  # Set to True for human, False for non-human, None for both

# Constant use for duckiebot
OMEGA_MAX = 8.0
GAIN = 1

# Search for bag files recursively
bag_files = []
record_types = {"sim_record": "vchicinvabot", "lab_record": "chicinvabot"}

for record_type, robot_name in record_types.items():
    if (record_type == "sim_record" and not process_sim) or (record_type == "lab_record" and not process_lab):
        continue
    for root, _, files in os.walk(os.path.join(base_folder, record_type)):
        is_human = "human_control" in root
        if process_human is not None and is_human != process_human:
            continue
        for file in files:
            if file.endswith(".bag"):
                bag_files.append((os.path.join(root, file), record_type, robot_name, is_human))

# Determine output filenames based on available records
record_used = set(record_type for _, record_type, _, _ in bag_files)
if not record_used:
    print("No valid bag files found based on selection. Exiting.")
    exit()

output_filename = "_".join(sorted(record_used))
if process_human is True:
    output_filename += "_human"
elif process_human is False:
    output_filename += "_nonhuman"
output_filename += ".hdf5"
hdf5_path = os.path.join(output_dir, output_filename)

# Topics of interest
image_topics = {"sim_record": "/vchicinvabot/camera_node/image/compressed", "lab_record": "/chicinvabot/camera_node/image/compressed"}
new_cmd_topics = {"sim_record": "/vchicinvabot/car_cmd_switch_node/cmd", "lab_record": "/chicinvabot/car_cmd_switch_node/cmd"}

# Initialize HDF5 file
with h5py.File(hdf5_path, "w") as f:
    f.create_group("data")

# Process each bag file
for idx, (bag_path, record_type, robot_name, is_human) in enumerate(bag_files, start=1):
    print(f"Processing {bag_path} as demo_{idx}...")
    bag = rosbag.Bag(bag_path)
    bridge = CvBridge()

    # Data structures
    timestamps = []
    images = []
    actions_new = []
    velocities = []
    omegas = []

    latest_action_new = [0.0, 0.0]
    action_queue_new = deque()

    image_topic = image_topics[record_type]
    new_cmd_topic = new_cmd_topics[record_type]

    for topic, msg, t in bag.read_messages(topics=[image_topic, new_cmd_topic]):
        timestamp = t.to_sec()

        if topic == new_cmd_topic:
            # Velocity and angle need to be normalized since robomimic kept the output between -1 and 1 (tanh func)
            latest_action_new = [round(msg.v, 3), round(msg.omega / OMEGA_MAX, 3)]
            action_queue_new.append((timestamp, latest_action_new))
            velocities.append(msg.v)
            omegas.append(msg.omega / OMEGA_MAX)

        elif topic == image_topic:
            frame = bridge.compressed_imgmsg_to_cv2(msg, "bgr8")

            while action_queue_new and action_queue_new[0][0] < timestamp - 0.1:
                action_queue_new.popleft()

            action_new = latest_action_new if action_queue_new else [0.0, 0.0]

            timestamps.append(timestamp)
            images.append(frame)
            actions_new.append(action_new)

    bag.close()

    # Convert to NumPy arrays
    timestamps = np.array(timestamps, dtype=np.float64)
    images = np.array(images, dtype=np.uint8)
    actions_new = np.array(actions_new, dtype=np.float32)
    velocities = np.array(velocities, dtype=np.float32)
    omegas = np.array(omegas, dtype=np.float32)

    num_samples = len(actions_new) - 1
    if num_samples <= 0:
        print(f"Skipping {bag_path} (no valid samples).")
        continue

    avg_velocity = np.mean(velocities) if velocities.size > 0 else 0.0
    std_velocity = np.std(velocities) if velocities.size > 0 else 0.0
    max_vel = np.max(velocities) if velocities.size > 0 else 0.0
    min_vel = np.min(velocities) if velocities.size > 0 else 0.0
    avg_omega = np.mean(omegas) if omegas.size > 0 else 0.0
    std_omega = np.std(omegas) if omegas.size > 0 else 0.0
    max_omega = np.max(omegas) if omegas.size > 0 else 0.0
    min_omega = np.min(omegas) if omegas.size > 0 else 0.0
    frame_rate = num_samples / (timestamps[-1] - timestamps[0]) if num_samples > 0 else 0.0

    print(f"Summary for {bag_path}:")
    print(f"  Timesteps: {num_samples}")
    print(f"  Average Velocity: {avg_velocity:.3f}")
    print(f"  Velocity Std Dev: {std_velocity:.3f}")
    print(f"  Cap in {min_vel} and {max_vel}")
    print(f"  Average Omega: {avg_omega:.3f}")
    print(f"  Omega Std Dev: {std_omega:.3f}")
    print(f"  Cap in {min_omega} and {max_omega}")
    print(f"  Frame Rate: {frame_rate:.3f} FPS")

    # Write to HDF5 file
    with h5py.File(hdf5_path, "a") as f:
        grp = f["data"].create_group(f"demo_{idx}")
        grp.attrs["num_samples"] = num_samples
        grp.attrs["frame_rate"] = frame_rate
        grp.attrs["human"] = int(is_human)
        grp.create_dataset("obs/observation", data=images[:-1])
        grp.create_dataset("actions", data=actions_new[:-1])
        grp.create_dataset("rewards", data=np.zeros(num_samples, dtype=np.float64))
        grp.create_dataset("dones", data=np.concatenate([np.zeros(num_samples - 1, dtype=np.int64), [1]]))

print(f"Processing complete. Saved data to {hdf5_path}")
