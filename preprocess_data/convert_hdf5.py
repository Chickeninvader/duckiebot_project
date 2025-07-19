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


def process_obstacle_avoidance_bags(process_sim=True, process_lab=True):
    """
    Process obstacle avoidance bag files, filtering by FSM states to include only relevant autonomous phases.
    """
    # Define input folders and output HDF5 file
    base_folder = "record"
    output_dir = "record/converted_standard"
    os.makedirs(output_dir, exist_ok=True)

    # Constants for duckiebot
    OMEGA_MAX = 8.0

    # Define dataset search criteria - only obstacle_avoidance and lane_following folders
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
        "lane_following_demos": 0,
        "obstacle_avoidance_demos": 0,
        "errors": [],
        "skipped_files": [],
        "fsm_state_counts": {}
    }

    # FSM states to include in the dataset
    VALID_FSM_STATES = {
        "LANE_FOLLOWING",
        "SWITCH_LANE_LEFT",
        "OBSTACLE_AVOIDANCE",
        "SWITCH_LANE_RIGHT",
        "INTERSECTION_COORDINATION"
    }

    print("Searching for obstacle avoidance and lane following bag files...")
    for record_type, robot_name in record_types.items():
        if (record_type == "sim_record" and not process_sim) or (record_type == "lab_record" and not process_lab):
            continue

        record_path = os.path.join(base_folder, record_type)
        if not os.path.exists(record_path):
            print(f"Warning: Path {record_path} does not exist")
            continue

        # Look for both lane_following and obstacle_avoidance folders
        for demo_type in ["lane_following", "obstacle_avoidance"]:
            demo_path = os.path.join(record_path, demo_type)
            if not os.path.exists(demo_path):
                print(f"Warning: Path {demo_path} does not exist")
                continue

            for root, _, files in os.walk(demo_path):
                for file in files:
                    if file.endswith(".bag"):
                        bag_files.append((os.path.join(root, file), record_type, robot_name, demo_type))
                        stats["num_bag_files"] += 1

    print(f"Found {stats['num_bag_files']} bag files to process")

    if not bag_files:
        print("No valid bag files found based on selection.")
        return

    # Generate output filename
    output_filename = ""
    if process_sim and process_lab:
        output_filename = "sim_lab"
    elif process_sim:
        output_filename = "sim"
    elif process_lab:
        output_filename = "lab"
    output_filename += "_obstacle_avoidance.hdf5"
    hdf5_path = os.path.join(output_dir, output_filename)

    # Topics of interest
    image_topics = {"sim_record": "/vchicinvabot/camera_node/image/compressed",
                    "lab_record": "/chicinvabot/camera_node/image/compressed"}
    cmd_topics = {"sim_record": "/vchicinvabot/car_cmd_switch_node/cmd",
                  "lab_record": "/chicinvabot/car_cmd_switch_node/cmd"}
    fsm_topics = {"sim_record": "/vchicinvabot/fsm_node/mode",
                  "lab_record": "/chicinvabot/fsm_node/mode"}

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
    for idx, (bag_path, record_type, robot_name, demo_type) in enumerate(bag_files, start=1):
        try:
            print(f"\nProcessing [{idx}/{stats['num_bag_files']}] {bag_path} as demo_{successful_demos+1}...")
            print(f"  Type: {record_type}, Demo type: {demo_type}")

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
                fsm_topic = fsm_topics[record_type]

                required_topics = [image_topic, cmd_topic]
                # FSM topic is optional for lane_following demos
                if demo_type == "obstacle_avoidance":
                    required_topics.append(fsm_topic)

                for topic in required_topics:
                    if topic not in topics:
                        print(f"  Error: Required topic {topic} not found in bag file")
                        stats["errors"].append(f"Missing topic {topic} in {bag_path}")
                        stats["skipped_files"].append(bag_path)
                        bag.close()
                        continue

                if any(topic not in topics for topic in required_topics):
                    continue

                print(f"  Image topic messages: {topics[image_topic].message_count}")
                print(f"  Command topic messages: {topics[cmd_topic].message_count}")
                if fsm_topic in topics:
                    print(f"  FSM topic messages: {topics[fsm_topic].message_count}")

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
            fsm_state_queue = deque()
            latest_action = [0.0, 0.0]
            current_fsm_state = None

            # Read all topics for processing
            topics_to_read = [image_topic, cmd_topic]
            if fsm_topic in topics:
                topics_to_read.append(fsm_topic)

            # Count messages for progress tracking
            total_messages = sum(topics[topic].message_count for topic in topics_to_read if topic in topics)
            processed_messages = 0

            try:
                for topic, msg, t in bag.read_messages(topics=topics_to_read):
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

                    elif topic == fsm_topic:
                        # Handle FSM state message - could be different message types
                        if hasattr(msg, 'state'):
                            fsm_state = msg.state
                        elif hasattr(msg, 'data'):
                            fsm_state = msg.data
                        else:
                            fsm_state = str(msg)

                        current_fsm_state = fsm_state
                        fsm_state_queue.append((timestamp, fsm_state))

                        # Update FSM state statistics
                        if fsm_state not in stats["fsm_state_counts"]:
                            stats["fsm_state_counts"][fsm_state] = 0
                        stats["fsm_state_counts"][fsm_state] += 1

                    elif topic == image_topic:
                        try:
                            frame = bridge.compressed_imgmsg_to_cv2(msg, "bgr8")

                            # Clean up old actions
                            while action_queue and action_queue[0][0] < timestamp - 0.1:
                                action_queue.popleft()

                            # Clean up old FSM states
                            while fsm_state_queue and fsm_state_queue[0][0] < timestamp - 0.5:
                                fsm_state_queue.popleft()

                            action = latest_action if action_queue else [0.0, 0.0]

                            # For obstacle_avoidance demos, filter by FSM state
                            if demo_type == "obstacle_avoidance":
                                # Find the most recent FSM state
                                relevant_fsm_state = None
                                if fsm_state_queue:
                                    # Get the most recent FSM state
                                    _, relevant_fsm_state = fsm_state_queue[-1]
                                elif current_fsm_state:
                                    relevant_fsm_state = current_fsm_state

                                # Skip frames that are not in valid FSM states
                                if relevant_fsm_state not in VALID_FSM_STATES:
                                    continue

                            # Include this frame
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
            if demo_type == "lane_following":
                stats["lane_following_demos"] += 1
            elif demo_type == "obstacle_avoidance":
                stats["obstacle_avoidance_demos"] += 1

            # Write to HDF5 file
            try:
                with h5py.File(hdf5_path, "a") as f:
                    demo_name = f"demo_{successful_demos}"
                    grp = f["data"].create_group(demo_name)
                    grp.attrs.update({
                        "num_samples": num_samples,
                        "frame_rate": frame_rate,
                        "demo_type": demo_type,
                        "record_type": record_type,
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

    # Print statistics
    print("\n" + "="*60)
    print(f"STATISTICS FOR {hdf5_path}")
    print("="*60)
    print(f"Number of bag files found: {stats['num_bag_files']}")
    print(f"Number of bag files processed: {stats['num_demos']}")
    print(f"Number of bag files skipped: {len(stats['skipped_files'])}")
    print(f"Number of errors encountered: {len(stats['errors'])}")
    print(f"Lane following demos: {stats['lane_following_demos']}")
    print(f"Obstacle avoidance demos: {stats['obstacle_avoidance_demos']}")
    print(f"Total frames: {stats['total_frames']}")

    if stats['min_velocity'] != float('inf'):
        print(f"Velocity range: {stats['min_velocity']:.3f} to {stats['max_velocity']:.3f}")
    else:
        print("No velocity data found")

    if stats['min_omega'] != float('inf'):
        print(f"Omega range (normalized): {stats['min_omega']:.3f} to {stats['max_omega']:.3f}")
    else:
        print("No omega data found")

    # Print FSM state statistics
    if stats['fsm_state_counts']:
        print("\nFSM State Distribution:")
        for state, count in sorted(stats['fsm_state_counts'].items()):
            print(f"  {state}: {count} messages")

    # Print first few errors if any
    if stats['errors']:
        print(f"\nFirst 5 errors:")
        for i, error in enumerate(stats['errors'][:5]):
            print(f"{i+1}. {error}")

    print("="*60 + "\n")

    # Save statistics to JSON
    stats_filename = os.path.splitext(output_filename)[0] + "_stats.json"
    stats_path = os.path.join(output_dir, stats_filename)
    # Convert any numpy types to Python types for JSON serialization
    json_stats = {}
    for key, value in stats.items():
        if isinstance(value, (np.integer, np.floating)):
            json_stats[key] = value.item()
        else:
            json_stats[key] = value

    with open(stats_path, 'w') as f:
        json.dump(json_stats, f, indent=4)


def analyze_obstacle_avoidance_hdf5(hdf5_path):
    """
    Function to analyze an existing obstacle avoidance HDF5 file and print statistics
    """
    if not os.path.exists(hdf5_path):
        print(f"File not found: {hdf5_path}")
        return

    stats = {
        "num_demos": 0,
        "lane_following_demos": 0,
        "obstacle_avoidance_demos": 0,
        "total_frames": 0,
        "min_velocity": float('inf'),
        "max_velocity": float('-inf'),
        "min_omega": float('inf'),
        "max_omega": float('-inf'),
        "demo_types": {}
    }

    with h5py.File(hdf5_path, 'r') as f:
        data_group = f.get('data')
        if data_group is None:
            print(f"No 'data' group found in {hdf5_path}")
            return

        for demo_name in data_group.keys():
            demo = data_group[demo_name]
            stats["num_demos"] += 1

            # Get demo type attribute
            demo_type = demo.attrs.get('demo_type', 'unknown')
            if demo_type == 'lane_following':
                stats["lane_following_demos"] += 1
            elif demo_type == 'obstacle_avoidance':
                stats["obstacle_avoidance_demos"] += 1

            if demo_type not in stats["demo_types"]:
                stats["demo_types"][demo_type] = 0
            stats["demo_types"][demo_type] += 1

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
    print("\n" + "="*60)
    print(f"ANALYSIS FOR {hdf5_path}")
    print("="*60)
    print(f"Number of demos: {stats['num_demos']}")
    print(f"Lane following demos: {stats['lane_following_demos']}")
    print(f"Obstacle avoidance demos: {stats['obstacle_avoidance_demos']}")
    print(f"Total frames: {stats['total_frames']}")

    if stats['min_velocity'] != float('inf'):
        print(f"Velocity range: {stats['min_velocity']:.3f} to {stats['max_velocity']:.3f}")
    else:
        print("No velocity data found")

    if stats['min_omega'] != float('inf'):
        print(f"Omega range (normalized): {stats['min_omega']:.3f} to {stats['max_omega']:.3f}")
    else:
        print("No omega data found")

    if stats['demo_types']:
        print("\nDemo Type Distribution:")
        for demo_type, count in sorted(stats['demo_types'].items()):
            print(f"  {demo_type}: {count} demos")

    print("="*60 + "\n")


if __name__ == "__main__":
    # Process all configurations
    print("Processing simulation data...")
    process_obstacle_avoidance_bags(process_sim=True, process_lab=False)

    print("\nProcessing lab data...")
    process_obstacle_avoidance_bags(process_sim=False, process_lab=True)

    print("\nProcessing both sim and lab data...")
    process_obstacle_avoidance_bags(process_sim=True, process_lab=True)

    # Example usage for analysis:
    # analyze_obstacle_avoidance_hdf5("record/converted_standard/sim_obstacle_avoidance.hdf5")
    # analyze_obstacle_avoidance_hdf5("record/converted_standard/lab_obstacle_avoidance.hdf5")
    # analyze_obstacle_avoidance_hdf5("record/converted_standard/sim_lab_obstacle_avoidance.hdf5")
