import rosbag
import numpy as np
import cv2
import h5py
import os
import json
from cv_bridge import CvBridge
from collections import deque


def compute_reward(d, phi, velocity, curvature, in_lane):
    """Compute reward based on lane position, heading, velocity, and whether in lane"""
    # Reference values
    d_ref = 0.0  # Ideal lateral offset
    phi_ref = 0.0  # Ideal heading angle

    # Tunable weights
    alpha = 1.0
    beta = 0.5
    gamma = 1.0
    lambda_ = 0.0  # High penalty for leaving the lane

    # Compute reward components
    r_d = -alpha * abs(d - d_ref)
    r_phi = -beta * abs(phi - phi_ref)
    r_v = gamma * velocity  # Reward movement
    r_lane = -lambda_ if not in_lane else 0  # High penalty for being out of lane

    # Total reward
    reward = r_d + r_phi + r_v + r_lane
    return reward


def process_bag_files(process_sim, process_lab, process_human):
    # Define input folders and output HDF5 file
    base_folder = "record"
    output_dir = "record/converted_standard"
    os.makedirs(output_dir, exist_ok=True)

    # Constants for duckiebot
    OMEGA_MAX = 8.0

    # Define dataset search criteria
    record_types = {"sim_record": "vchicinvabot", "lab_record": "chicinvabot"}
    bag_files = []

    # Find all bag files
    print("Searching for bag files...")
    for record_type, robot_name in record_types.items():
        if (record_type == "sim_record" and not process_sim) or (record_type == "lab_record" and not process_lab):
            continue

        record_path = os.path.join(base_folder, record_type)
        if not os.path.exists(record_path):
            continue

        for root, _, files in os.walk(record_path):
            is_human = "human_control" in root
            if process_human is not None and is_human != process_human:
                continue

            for file in files:
                if file.endswith(".bag"):
                    bag_files.append((os.path.join(root, file), record_type, robot_name, is_human))

    print(f"Found {len(bag_files)} bag files to process")

    if not bag_files:
        print("No valid bag files found based on selection.")
        return

    # Generate output filename
    output_filename = "sim" if process_sim else "lab"
    if process_human is True:
        output_filename += "_human"
    elif process_human is False:
        output_filename += "_nonhuman"
    else:
        output_filename += "_both"
    output_filename += ".hdf5"
    hdf5_path = os.path.join(output_dir, output_filename)

    # Topics of interest
    image_topics = {"sim_record": "/vchicinvabot/camera_node/image/compressed",
                    "lab_record": "/chicinvabot/camera_node/image/compressed"}
    cmd_topics = {"sim_record": "/vchicinvabot/car_cmd_switch_node/cmd",
                  "lab_record": "/chicinvabot/car_cmd_switch_node/cmd"}
    lane_pose_topics = {"sim_record": "/vchicinvabot/lane_filter_node/lane_pose",
                        "lab_record": "/chicinvabot/lane_filter_node/lane_pose"}

    # Initialize HDF5 file
    with h5py.File(hdf5_path, "w") as f:
        f.create_group("data")
    print(f"Successfully initialized {hdf5_path}")

    # Process each bag file
    successful_demos = 0
    stats = {"human_demos": 0, "non_human_demos": 0, "total_frames": 0}

    for idx, (bag_path, record_type, robot_name, is_human) in enumerate(bag_files, start=1):
        print(f"\nProcessing [{idx}/{len(bag_files)}] {bag_path} as demo_{successful_demos + 1}...")
        print(f"  Type: {record_type}, Human control: {is_human}")

        # Open bag file
        bag = rosbag.Bag(bag_path)
        bridge = CvBridge()

        # Get topics
        image_topic = image_topics[record_type]
        cmd_topic = cmd_topics[record_type]
        lane_pose_topic = lane_pose_topics[record_type]

        # Initialize data collection
        timestamps, images, actions = [], [], []
        lane_poses, in_lane_status = [], []
        action_queue = deque()
        latest_action = [0.0, 0.0]
        latest_lane_pose = {'d': 0.0, 'phi': 0.0, 'curvature': 0.0}
        latest_in_lane = True

        # Process bag messages
        for topic, msg, t in bag.read_messages(topics=[image_topic, cmd_topic, lane_pose_topic]):
            timestamp = t.to_sec()

            if topic == cmd_topic:
                velocity = round(msg.v, 3)
                omega = round(msg.omega / OMEGA_MAX, 3)
                latest_action = [velocity, omega]
                action_queue.append((timestamp, latest_action))

            elif topic == lane_pose_topic:
                try:
                    d = msg.d
                    phi = msg.phi
                    curvature = getattr(msg, 'curvature', 0.0)
                    latest_lane_pose = {'d': d, 'phi': phi, 'curvature': curvature}

                    # Determine if in lane (adjust threshold as needed)
                    latest_in_lane = abs(d) < 0.3
                except Exception as e:
                    print(f"  Error processing lane pose: {str(e)}")
                    continue

            elif topic == image_topic:
                try:
                    frame = bridge.compressed_imgmsg_to_cv2(msg, "bgr8")

                    # Clean up old actions
                    while action_queue and action_queue[0][0] < timestamp - 0.1:
                        action_queue.popleft()

                    # Use latest action or default
                    action = latest_action if action_queue else [0.0, 0.0]

                    # Store data
                    timestamps.append(timestamp)
                    images.append(frame)
                    actions.append(action)
                    lane_poses.append(latest_lane_pose)
                    in_lane_status.append(latest_in_lane)
                    stats["total_frames"] += 1
                except Exception as e:
                    print(f"  Error processing image: {str(e)}")
                    continue

        bag.close()

        # Check if we have any data
        if len(timestamps) <= 1:
            print(f"  Not enough data in {bag_path}, skipping")
            continue

        # Calculate frame rate
        num_samples = len(timestamps) - 1
        frame_rate = num_samples / (timestamps[-1] - timestamps[0]) if num_samples > 0 else 0.0
        print(f"  Frame rate: {frame_rate:.2f} Hz")

        # Calculate rewards for each step
        rewards = []
        for i in range(len(actions) - 1):
            velocity = actions[i][0]
            reward = compute_reward(lane_poses[i], velocity, in_lane_status[i])
            rewards.append(reward)

        # Convert to NumPy arrays
        timestamps = np.array(timestamps)
        images = np.array(images)
        actions = np.array(actions)
        rewards = np.array(rewards)

        # Update statistics
        successful_demos += 1
        if is_human:
            stats["human_demos"] += 1
        else:
            stats["non_human_demos"] += 1

        # Write to HDF5 file
        with h5py.File(hdf5_path, "a") as f:
            demo_name = f"demo_{successful_demos}"
            grp = f["data"].create_group(demo_name)
            grp.attrs.update({
                "num_samples": num_samples,
                "frame_rate": frame_rate,
                "human": int(is_human),
                "source_file": os.path.basename(bag_path)
            })
            grp.create_dataset("obs/observation", data=images[:-1])
            grp.create_dataset("actions", data=actions[:-1])
            grp.create_dataset("rewards", data=rewards)
            grp.create_dataset("dones", data=np.concatenate([np.zeros(num_samples - 1), [1]]))
            print(f"  Successfully wrote demo data to HDF5")

        print(f"  Completed processing {bag_path} as {demo_name}")

    # Print statistics for this configuration
    print("\n" + "=" * 50)
    print(f"STATISTICS FOR {hdf5_path}")
    print("=" * 50)
    print(f"Total demos processed: {successful_demos}")
    print(f"Human-controlled demos: {stats['human_demos']}")
    print(f"Non-human-controlled demos: {stats['non_human_demos']}")
    print(f"Total frames: {stats['total_frames']}")
    print("=" * 50 + "\n")

    # Save statistics to JSON
    stats_filename = os.path.splitext(output_filename)[0] + "_stats.json"
    stats_path = os.path.join(output_dir, stats_filename)
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=4)


# Run the function for all configurations
for sim in [True, False]:
    for human in [True, False, None]:
        process_bag_files(process_sim=sim, process_lab=not sim, process_human=human)
