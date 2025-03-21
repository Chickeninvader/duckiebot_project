import rosbag
import numpy as np
import cv2
import h5py
import os
import json
import datetime
import traceback
import sys
from cv_bridge import CvBridge
from collections import deque


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

    # Statistics variables
    stats = {
        "num_bag_files": 0,
        "num_demos": 0,
        "min_velocity": float('inf'),
        "max_velocity": float('-inf'),
        "min_omega": float('inf'),
        "max_omega": float('-inf'),
        "total_frames": 0,
        "human_demos": 0,
        "non_human_demos": 0,
        "errors": [],
        "skipped_files": []
    }

    # Find all bag files
    print("Searching for bag files...")
    for record_type, robot_name in record_types.items():
        if (record_type == "sim_record" and not process_sim) or (record_type == "lab_record" and not process_lab):
            continue

        record_path = os.path.join(base_folder, record_type)
        if not os.path.exists(record_path):
            print(f"Warning: Path {record_path} does not exist")
            continue

        for root, _, files in os.walk(record_path):
            is_human = "human_control" in root
            if process_human is not None and is_human != process_human:
                continue

            for file in files:
                if file.endswith(".bag"):
                    bag_files.append((os.path.join(root, file), record_type, robot_name, is_human))
                    stats["num_bag_files"] += 1

    print(f"Found {stats['num_bag_files']} bag files to process")

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

    # Initialize HDF5 file
    try:
        with h5py.File(hdf5_path, "w") as f:
            f.create_group("data")
        print(f"Successfully initialized {hdf5_path}")
    except Exception as e:
        print(f"Error initializing HDF5 file: {str(e)}")
        traceback.print_exc()
        return

    # Process each bag file
    successful_demos = 0
    for idx, (bag_path, record_type, robot_name, is_human) in enumerate(bag_files, start=1):
        try:

            print(f"\nProcessing [{idx}/{stats['num_bag_files']}] {bag_path} as demo_{successful_demos+1}...")
            print(f"  Type: {record_type}, Human control: {is_human}")

            # Verify file exists
            if not os.path.exists(bag_path):
                print(f"  Error: File {bag_path} does not exist")
                stats["errors"].append(f"File not found: {bag_path}")
                stats["skipped_files"].append(bag_path)
                continue

            # Check file size
            file_size_mb = os.path.getsize(bag_path) / (1024 * 1024)
            print(f"  File size: {file_size_mb:.2f} MB")
            if file_size_mb < 0.1:
                print(f"  Warning: File size very small ({file_size_mb:.2f} MB)")

            # Open bag file
            try:
                bag = rosbag.Bag(bag_path)
                print(f"  Successfully opened bag file")
            except Exception as e:
                print(f"  Error opening bag file: {str(e)}")
                stats["errors"].append(f"Failed to open {bag_path}: {str(e)}")
                stats["skipped_files"].append(bag_path)
                continue

            # Get bag info
            try:
                bag_info = bag.get_type_and_topic_info()
                topics = bag_info.topics
                print(f"  Available topics: {list(topics.keys())}")

                image_topic = image_topics[record_type]
                cmd_topic = cmd_topics[record_type]

                if image_topic not in topics:
                    print(f"  Error: Image topic {image_topic} not found in bag file")
                    stats["errors"].append(f"Missing image topic in {bag_path}")
                    stats["skipped_files"].append(bag_path)
                    bag.close()
                    continue

                if cmd_topic not in topics:
                    print(f"  Error: Command topic {cmd_topic} not found in bag file")
                    stats["errors"].append(f"Missing command topic in {bag_path}")
                    stats["skipped_files"].append(bag_path)
                    bag.close()
                    continue

                print(f"  Image topic messages: {topics[image_topic].message_count}")
                print(f"  Command topic messages: {topics[cmd_topic].message_count}")

                if topics[image_topic].message_count == 0:
                    print(f"  Error: No messages on image topic")
                    stats["errors"].append(f"No image messages in {bag_path}")
                    stats["skipped_files"].append(bag_path)
                    bag.close()
                    continue

                if topics[cmd_topic].message_count == 0:
                    print(f"  Error: No messages on command topic")
                    stats["errors"].append(f"No command messages in {bag_path}")
                    stats["skipped_files"].append(bag_path)
                    bag.close()
                    continue

            except Exception as e:
                print(f"  Error getting bag info: {str(e)}")
                stats["errors"].append(f"Failed to get info for {bag_path}: {str(e)}")
                stats["skipped_files"].append(bag_path)
                bag.close()
                continue

            bridge = CvBridge()
            timestamps, images, actions = [], [], []
            velocities, omegas = [], []
            action_queue = deque()
            latest_action = [0.0, 0.0]

            # Count messages for progress tracking
            total_messages = topics[image_topic].message_count + topics[cmd_topic].message_count
            processed_messages = 0

            try:
                for topic, msg, t in bag.read_messages(topics=[image_topic, cmd_topic]):
                    processed_messages += 1
                    if processed_messages % 100 == 0:
                        print(f"  Processing message {processed_messages}/{total_messages}")

                    timestamp = t.to_sec()

                    if topic == cmd_topic:
                        velocity = round(msg.v, 3)
                        omega = round(msg.omega / OMEGA_MAX, 3)
                        latest_action = [velocity, omega]
                        action_queue.append((timestamp, latest_action))
                        velocities.append(velocity)
                        omegas.append(omega)

                        # Update velocity and omega statistics
                        stats["min_velocity"] = min(stats["min_velocity"], velocity)
                        stats["max_velocity"] = max(stats["max_velocity"], velocity)
                        stats["min_omega"] = min(stats["min_omega"], omega)
                        stats["max_omega"] = max(stats["max_omega"], omega)

                    elif topic == image_topic:
                        try:
                            frame = bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
                            while action_queue and action_queue[0][0] < timestamp - 0.1:
                                action_queue.popleft()
                            action = latest_action if action_queue else [0.0, 0.0]

                            timestamps.append(timestamp)
                            images.append(frame)
                            actions.append(action)
                            stats["total_frames"] += 1
                        except Exception as e:
                            print(f"  Error processing image: {str(e)}")
                            continue
            except Exception as e:
                print(f"  Error reading messages: {str(e)}")
                stats["errors"].append(f"Error reading messages from {bag_path}: {str(e)}")
                bag.close()
                continue

            bag.close()
            print(f"  Successfully processed all messages")

            # Check if we have any data
            if not timestamps or not images or not actions:
                print(f"  Error: No valid data extracted from {bag_path}")
                stats["errors"].append(f"No valid data in {bag_path}")
                stats["skipped_files"].append(bag_path)
                continue

            print(f"  Extracted {len(timestamps)} timestamps, {len(images)} images, {len(actions)} actions")

            # Convert to NumPy arrays
            timestamps = np.array(timestamps)
            images = np.array(images)
            actions = np.array(actions)
            velocities = np.array(velocities) if velocities else np.array([])
            omegas = np.array(omegas) if omegas else np.array([])

            # Ensure we have consistent data
            if len(images) != len(actions) or len(images) != len(timestamps):
                print(f"  Error: Inconsistent data lengths - timestamps: {len(timestamps)}, images: {len(images)}, actions: {len(actions)}")
                stats["errors"].append(f"Inconsistent data in {bag_path}")
                stats["skipped_files"].append(bag_path)
                continue

            num_samples = len(actions) - 1
            if num_samples <= 0:
                print(f"  Error: Not enough samples in {bag_path}")
                stats["errors"].append(f"Not enough samples in {bag_path}")
                stats["skipped_files"].append(bag_path)
                continue

            frame_rate = num_samples / (timestamps[-1] - timestamps[0]) if num_samples > 0 else 0.0
            print(f"  Frame rate: {frame_rate:.2f} Hz")

            # Update demo statistics
            successful_demos += 1
            stats["num_demos"] += 1
            if is_human:
                stats["human_demos"] += 1
            else:
                stats["non_human_demos"] += 1

            # Write to HDF5 file
            try:
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
                    grp.create_dataset("rewards", data=np.zeros(num_samples))
                    grp.create_dataset("dones", data=np.concatenate([np.zeros(num_samples - 1), [1]]))
                    print(f"  Successfully wrote demo data to HDF5")
            except Exception as e:
                print(f"  Error writing to HDF5: {str(e)}")
                traceback.print_exc()
                stats["errors"].append(f"Failed to write {bag_path} to HDF5: {str(e)}")
                continue

            print(f"  Completed processing {bag_path} as {demo_name}")

        except Exception as e:
            print(f"  Unexpected error processing {bag_path}: {str(e)}")
            traceback.print_exc()
            stats["errors"].append(f"Unexpected error with {bag_path}: {str(e)}")
            stats["skipped_files"].append(bag_path)

    # Print statistics for this configuration
    print("\n" + "="*50)
    print(f"STATISTICS FOR {hdf5_path}")
    print("="*50)
    print(f"Number of bag files found: {stats['num_bag_files']}")
    print(f"Number of bag files processed: {stats['num_demos']}")
    print(f"Number of bag files skipped: {len(stats['skipped_files'])}")
    print(f"Number of errors encountered: {len(stats['errors'])}")
    print(f"Human-controlled demos: {stats['human_demos']}")
    print(f"Non-human-controlled demos: {stats['non_human_demos']}")
    print(f"Total frames: {stats['total_frames']}")

    if stats['min_velocity'] != float('inf'):
        print(f"Velocity range: {stats['min_velocity']:.3f} to {stats['max_velocity']:.3f}")
    else:
        print("No velocity data found")

    if stats['min_omega'] != float('inf'):
        print(f"Omega range (normalized): {stats['min_omega']:.3f} to {stats['max_omega']:.3f}")
    else:
        print("No omega data found")

    # Print first few errors if any
    if stats['errors']:
        print("\nFirst 5 errors:")
        for i, error in enumerate(stats['errors'][:5]):
            print(f"{i+1}. {error}")

    print("="*50 + "\n")

    # Save statistics to JSON
    stats_filename = os.path.splitext(output_filename)[0] + "_stats.json"
    stats_path = os.path.join(output_dir, stats_filename)
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=4)


def analyze_hdf5_file(hdf5_path):
    """
    Function to analyze an existing HDF5 file and print statistics
    """
    if not os.path.exists(hdf5_path):
        print(f"File not found: {hdf5_path}")
        return

    stats = {
        "num_demos": 0,
        "human_demos": 0,
        "non_human_demos": 0,
        "total_frames": 0,
        "min_velocity": float('inf'),
        "max_velocity": float('-inf'),
        "min_omega": float('inf'),
        "max_omega": float('-inf')
    }

    with h5py.File(hdf5_path, 'r') as f:
        data_group = f.get('data')
        if data_group is None:
            print(f"No 'data' group found in {hdf5_path}")
            return

        for demo_name in data_group.keys():
            demo = data_group[demo_name]
            stats["num_demos"] += 1

            # Get human attribute
            is_human = demo.attrs.get('human', 0)
            if is_human:
                stats["human_demos"] += 1
            else:
                stats["non_human_demos"] += 1

            # Get actions dataset
            if 'actions' in demo:
                actions = demo['actions'][()]
                stats["total_frames"] += len(actions)

                # Extract velocities and omegas
                velocities = actions[:, 0]
                omegas = actions[:, 1]

                # Update statistics
                if len(velocities) > 0:
                    stats["min_velocity"] = min(stats["min_velocity"], np.min(velocities))
                    stats["max_velocity"] = max(stats["max_velocity"], np.max(velocities))

                if len(omegas) > 0:
                    stats["min_omega"] = min(stats["min_omega"], np.min(omegas))
                    stats["max_omega"] = max(stats["max_omega"], np.max(omegas))

    # Print statistics
    print("\n" + "="*50)
    print(f"STATISTICS FOR {hdf5_path}")
    print("="*50)
    print(f"Number of demos: {stats['num_demos']}")
    print(f"Human-controlled demos: {stats['human_demos']}")
    print(f"Non-human-controlled demos: {stats['non_human_demos']}")
    print(f"Total frames: {stats['total_frames']}")

    if stats['min_velocity'] != float('inf'):
        print(f"Velocity range: {stats['min_velocity']:.3f} to {stats['max_velocity']:.3f}")
    else:
        print("No velocity data found")

    if stats['min_omega'] != float('inf'):
        print(f"Omega range (normalized): {stats['min_omega']:.3f} to {stats['max_omega']:.3f}")
    else:
        print("No omega data found")
    print("="*50 + "\n")


# Run the function for all configurations
for sim in [True, False]:
    for human in [True, False, None]:
        process_bag_files(process_sim=sim, process_lab=not sim, process_human=human)

# To analyze existing HDF5 files, you can use:
analyze_hdf5_file("record/converted_standard/sim_human.hdf5")
