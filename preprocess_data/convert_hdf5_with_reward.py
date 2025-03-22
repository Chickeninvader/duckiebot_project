import argparse
import csv
import os
import traceback
from collections import deque

import h5py
import numpy as np
import pandas as pd
import rosbag
from cv_bridge import CvBridge
from tqdm import tqdm


def compute_reward(d, phi, velocity, curvature, in_lane):
    """Compute reward based on lane position, heading, velocity, and lane status."""
    # Reference values
    d_ref = 0.0  # Ideal lateral offset
    phi_ref = 0.0  # Ideal heading angle

    # Tunable weights
    alpha = 1.0
    beta = 0.5
    gamma = 1.0
    lambda_ = 5.0  # Penalty for leaving the lane

    # Compute reward components
    r_d = -alpha * abs(d - d_ref)
    r_phi = -beta * abs(phi - phi_ref)
    r_v = gamma * velocity  # Reward movement
    r_lane = -lambda_ if not in_lane else 0  # Penalty for being out of lane

    # Total reward
    reward = r_d + r_phi + r_v + r_lane
    return reward


def process_bag_files(process_sim, process_lab, process_human, subset=True, debug=False):
    """Process ROS bag files and store observations, actions, and rewards in HDF5 format."""
    # Define input folders and output HDF5 file
    base_folder = "record"
    output_dir = "record/converted_standard"
    os.makedirs(output_dir, exist_ok=True)

    # Constants for duckiebot
    OMEGA_MAX = 8.0

    # Define dataset search criteria
    record_types = {"sim_record": "vchicinvabot",
                    "lab_record": "chicinvabot",
                    "lab_record_2": "chicinvabot",
                    "lab_record_3": "chicinvabot"}
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
        "avg_reward": 0.0,
        "total_demo_time": 0.0,
        "errors": [],
        "skipped_files": [],
        "demo_stats": {}  # Will store individual demo statistics

    }

    # Find all bag files
    print("Searching for bag files...")
    sim_bag_files = []
    lab_bag_files = []

    for record_type, robot_name in record_types.items():
        if (record_type == "sim_record" and not process_sim) or (
                record_type.startswith("lab_record") and not process_lab):
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
                    full_path = os.path.join(root, file)
                    if record_type == "sim_record":
                        sim_bag_files.append((full_path, record_type, robot_name, is_human))
                    else:  # lab_record, lab_record_2, lab_record_3
                        lab_bag_files.append((full_path, record_type, robot_name, is_human))

    # Handle debug and subset modes
    if debug:
        # In debug mode, only process one file from each type if available
        if sim_bag_files or lab_bag_files:
            bag_files = []
            if sim_bag_files:
                bag_files.append(sim_bag_files[0])
            if lab_bag_files:
                bag_files.append(lab_bag_files[0])
            print(f"Debug mode: Only processing {len(bag_files)} bag file(s)")
        else:
            print("No bag files found!")
            return
    elif subset and lab_bag_files:
        # Apply subset only to lab bag files (select half)
        import random
        random.seed(42)
        half_lab_size = max(1, len(lab_bag_files) // 2)
        selected_lab_files = random.sample(lab_bag_files, half_lab_size)

        # Combine with all sim files
        bag_files = sim_bag_files + selected_lab_files
        print(f"Subset mode: Processing all {len(sim_bag_files)} simulator files and {len(selected_lab_files)} lab files (half of {len(lab_bag_files)} total)")
    else:
        # Use all files
        bag_files = sim_bag_files + lab_bag_files
        print(f"Processing all {len(sim_bag_files)} simulator files and {len(lab_bag_files)} lab files")

    # Update the stats count to reflect the actual number of files to be processed
    stats["num_bag_files"] = len(bag_files)
    print(f"Total bag files to process: {stats['num_bag_files']}")
    # Generate output filename
    output_filename = ""
    if process_sim:
        output_filename += "sim_"
    if process_lab:
        output_filename += "lab_"
    if process_human is True:
        output_filename += "_human"
    elif process_human is False:
        output_filename += "_nonhuman"
    else:
        output_filename += "_both"

    if subset:
        output_filename += "_subset"

    if debug:
        output_filename += "_debug"

    output_filename += ".hdf5"
    hdf5_path = os.path.join(output_dir, output_filename)
    stats_path = os.path.join(output_dir, f"{output_filename[:-5]}_stats.csv")

    # Dataset name for statistics
    dataset_name = output_filename[:-5]  # Remove .hdf5 extension

    # Topics of interest for different record types
    image_topics = {
        "sim_record": "/vchicinvabot/camera_node/image/compressed",
        "lab_record": "/chicinvabot/camera_node/image/compressed",
        "lab_record_2": "/chicinvabot/camera_node/image/compressed",
        "lab_record_3": "/chicinvabot/camera_node/image/compressed"
    }
    cmd_topics = {
        "sim_record": "/vchicinvabot/car_cmd_switch_node/cmd",
        "lab_record": "/chicinvabot/car_cmd_switch_node/cmd",
        "lab_record_2": "/chicinvabot/car_cmd_switch_node/cmd",
        "lab_record_3": "/chicinvabot/car_cmd_switch_node/cmd"
    }
    lane_pose_topics = {
        "sim_record": "/vchicinvabot/lane_filter_node/lane_pose",
        "lab_record": "/chicinvabot/lane_filter_node/lane_pose",
        "lab_record_2": "/chicinvabot/lane_filter_node/lane_pose",
        "lab_record_3": "/chicinvabot/lane_filter_node/lane_pose"
    }

    with h5py.File(hdf5_path, "w") as f:
        f.create_group("data")

    # Process each bag file
    successful_demos = 0
    total_rewards = 0.0
    bag_pbar = tqdm(enumerate(bag_files, start=1), total=len(bag_files), desc="Processing bag files")

    for idx, (bag_path, record_type, robot_name, is_human) in bag_pbar:
        bag_name = os.path.basename(bag_path)
        bag_pbar.set_postfix({"file": bag_name, "type": record_type})

        try:
            # Open bag file
            try:
                bag = rosbag.Bag(bag_path)
            except Exception as e:
                stats["errors"].append(f"Failed to open {bag_path}: {str(e)}")
                stats["skipped_files"].append(bag_path)
                continue

            # Get bag info
            try:
                bag_info = bag.get_type_and_topic_info()
                topics = bag_info.topics

                image_topic = image_topics[record_type]
                cmd_topic = cmd_topics[record_type]
                lane_pose_topic = lane_pose_topics[record_type]

                if image_topic not in topics:
                    stats["errors"].append(f"Missing image topic in {bag_path}")
                    stats["skipped_files"].append(bag_path)
                    bag.close()
                    continue

                if cmd_topic not in topics:
                    stats["errors"].append(f"Missing command topic in {bag_path}")
                    stats["skipped_files"].append(bag_path)
                    bag.close()
                    continue

                # Lane pose topic may not exist in all files, so just warn if missing and continue
                if lane_pose_topic not in topics:
                    stats["errors"].append(f"Missing lane pose topic in {bag_path}")
                    stats["skipped_files"].append(bag_path)
                    bag.close()
                    continue

                if topics[image_topic].message_count == 0:
                    stats["errors"].append(f"No image messages in {bag_path}")
                    stats["skipped_files"].append(bag_path)
                    bag.close()
                    continue

                if topics[cmd_topic].message_count == 0:
                    stats["errors"].append(f"No command messages in {bag_path}")
                    stats["skipped_files"].append(bag_path)
                    bag.close()
                    continue

            except Exception as e:
                stats["errors"].append(f"Failed to get info for {bag_path}: {str(e)}")
                stats["skipped_files"].append(bag_path)
                bag.close()
                continue

            bridge = CvBridge()
            timestamps, images, actions = [], [], []
            velocities, omegas = [], []
            lane_poses, in_lane_status = [], []
            action_queue = deque()
            latest_action = [0.0, 0.0]
            latest_lane_pose = {"d": 0.0, "phi": 0.0, "curvature": 0.0, "in_lane": True}

            # Determine which topics to read
            topics_to_read = [image_topic, cmd_topic]
            if lane_pose_topic in topics:
                topics_to_read.append(lane_pose_topic)

            # Count messages for progress tracking
            total_messages = sum(topics[topic].message_count for topic in topics_to_read if topic in topics)

            try:
                msg_pbar = tqdm(total=total_messages, desc="Processing messages", leave=False)

                for topic, msg, t in bag.read_messages(topics=topics_to_read):
                    msg_pbar.update(1)

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

                    elif topic == lane_pose_topic:
                        try:
                            latest_lane_pose = {
                                "d": getattr(msg, "d", 0.0),
                                "phi": getattr(msg, "phi", 0.0),
                                "curvature": getattr(msg, "curvature", 0.0),
                                "in_lane": getattr(msg, "in_lane", True)  # Default to True if missing
                            }
                        except Exception as e:
                            continue

                    elif topic == image_topic:
                        try:
                            frame = bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
                            while action_queue and action_queue[0][0] < timestamp - 0.1:
                                action_queue.popleft()
                            action = latest_action if action_queue else [0.0, 0.0]

                            timestamps.append(timestamp)
                            images.append(frame)
                            actions.append(action)
                            lane_poses.append(latest_lane_pose.copy())
                            in_lane_status.append(latest_lane_pose["in_lane"])
                            stats["total_frames"] += 1
                        except Exception as e:
                            continue

                msg_pbar.close()
            except Exception as e:
                stats["errors"].append(f"Error reading messages from {bag_path}: {str(e)}")
                bag.close()
                continue

            bag.close()

            # Check if we have any data
            if not timestamps or not images or not actions:
                stats["errors"].append(f"No valid data in {bag_path}")
                stats["skipped_files"].append(bag_path)
                continue

            # Convert to NumPy arrays
            images_np = np.array(images)
            actions_np = np.array(actions)

            # Ensure we have consistent data
            if len(images_np) != len(actions_np) or len(images_np) != len(timestamps):
                stats["errors"].append(f"Inconsistent data in {bag_path}")
                stats["skipped_files"].append(bag_path)
                continue

            num_samples = len(images_np) - 1

            # Compute rewards
            rewards = []
            for i in range(len(actions_np) - 1):
                reward = compute_reward(
                    lane_poses[i]["d"],
                    lane_poses[i]["phi"],
                    actions_np[i][0],  # velocity
                    lane_poses[i]["curvature"],
                    lane_poses[i]["in_lane"]
                )
                rewards.append(reward)
            rewards_np = np.array(rewards)

            # Calculate demo statistics
            demo_time = timestamps[-1] - timestamps[0] if num_samples > 0 else 0.0
            stats["total_demo_time"] += demo_time

            # Update reward statistics
            total_demo_reward = np.sum(rewards_np)
            total_rewards += total_demo_reward

            # Calculate frame rate
            frame_rate = num_samples / demo_time if demo_time > 0 else 0.0

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
                    # Demo attributes
                    demo_attrs = {
                        "num_samples": num_samples,
                        "frame_rate": frame_rate,
                        "human": int(is_human),
                        "source_file": os.path.basename(bag_path),
                        "total_reward": total_demo_reward,
                        "demo_time": demo_time
                    }

                    # Store attributes in HDF5
                    grp.attrs.update(demo_attrs)

                    # Also store in stats dictionary for later reporting
                    stats["demo_stats"][demo_name] = {
                        "source_file": os.path.basename(bag_path),
                        "record_type": record_type,
                        "human_controlled": is_human,
                        "num_samples": num_samples,
                        "frame_rate": frame_rate,
                        "total_reward": total_demo_reward,
                        "demo_time": demo_time,
                        "avg_velocity": np.mean(velocities) if velocities else 0.0,
                        "avg_omega": np.mean(omegas) if omegas else 0.0
                    }

                    # Continue with creating datasets...
                    grp.create_dataset("obs/observation", data=images_np[:-1], compression="gzip")
                    grp.create_dataset("next_obs/observation", data=images_np[1:], compression="gzip")
                    grp.create_dataset("actions", data=actions_np[:-1])
                    grp.create_dataset("rewards", data=rewards_np)
                    grp.create_dataset("dones", data=np.concatenate([np.zeros(num_samples - 1, dtype=np.int32), [1]]))

            except Exception as e:
                traceback.print_exc()
                stats["errors"].append(f"Failed to write {bag_path} to HDF5: {str(e)}")
                continue

        except Exception as e:
            traceback.print_exc()
            stats["errors"].append(f"Unexpected error with {bag_path}: {str(e)}")
            stats["skipped_files"].append(bag_path)

    # Calculate average reward
    if successful_demos > 0:
        stats["avg_reward"] = total_rewards / successful_demos

    # Write statistics to CSV
    try:
        with open(stats_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Statistic', 'Value'])
            writer.writerow(['Number of bag files found', stats['num_bag_files']])
            writer.writerow(['Number of bag files processed', stats['num_demos']])
            writer.writerow(['Number of bag files skipped', len(stats['skipped_files'])])
            writer.writerow(['Number of errors encountered', len(stats['errors'])])
            writer.writerow(['Human-controlled demos', stats['human_demos']])
            writer.writerow(['Non-human-controlled demos', stats['non_human_demos']])
            writer.writerow(['Total frames', stats['total_frames']])
            writer.writerow(['Min velocity', stats['min_velocity']])
            writer.writerow(['Max velocity', stats['max_velocity']])
            writer.writerow(['Min omega', stats['min_omega']])
            writer.writerow(['Max omega', stats['max_omega']])
            writer.writerow(['Average reward', stats['avg_reward']])
            writer.writerow(['Total demo time (seconds)', stats['total_demo_time']])

            # Write errors
            writer.writerow([])
            writer.writerow(['Errors:'])
            for i, error in enumerate(stats['errors']):
                writer.writerow([f'Error {i + 1}', error])

            # Write skipped files
            writer.writerow([])
            writer.writerow(['Skipped Files:'])
            for i, skip_file in enumerate(stats['skipped_files']):
                writer.writerow([f'Skipped {i + 1}', skip_file])
    except Exception as e:
        print(f"Error writing statistics to CSV: {str(e)}")

    # Print summary
    print("\nProcessing Complete:")
    print(f"- Processed {stats['num_demos']} of {stats['num_bag_files']} bag files")
    print(f"- Created {successful_demos} demonstrations")
    print(f"- Output saved to {hdf5_path}")
    print(f"- Statistics saved to {stats_path}")

    # Return the statistics and dataset name
    return dataset_name, stats


# Add this function to your script to handle statistics collection
def save_statistics(output_dir, all_stats):
    """
    Save all collected statistics in organized CSV files.

    Args:
        output_dir: Directory to save statistics files
        all_stats: Dictionary with keys as dataset names and values as statistics dictionaries
    """
    # Make sure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Create a summary dataframe for general statistics across all datasets
    summary_data = []
    for dataset_name, stats in all_stats.items():
        summary_data.append({
            'Dataset': dataset_name,
            'Bag Files Found': stats['num_bag_files'],
            'Demos Processed': stats['num_demos'],
            'Files Skipped': len(stats['skipped_files']),
            'Errors': len(stats['errors']),
            'Human Demos': stats['human_demos'],
            'Non-human Demos': stats['non_human_demos'],
            'Total Frames': stats['total_frames'],
            'Min Velocity': stats['min_velocity'] if stats['min_velocity'] != float('inf') else 'N/A',
            'Max Velocity': stats['max_velocity'] if stats['max_velocity'] != float('-inf') else 'N/A',
            'Min Omega': stats['min_omega'] if stats['min_omega'] != float('inf') else 'N/A',
            'Max Omega': stats['max_omega'] if stats['max_omega'] != float('-inf') else 'N/A',
            'Avg Reward': stats['avg_reward'],
            'Total Demo Time (s)': stats['total_demo_time']
        })

    # Save the summary dataframe
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(output_dir, 'all_datasets_summary.csv'), index=False)
    print(f"Summary statistics saved to {os.path.join(output_dir, 'all_datasets_summary.csv')}")

    # Save individual demo statistics for all datasets
    all_demos_data = []
    for dataset_name, stats in all_stats.items():
        if 'demo_stats' in stats:
            for demo_name, demo_stats in stats['demo_stats'].items():
                demo_data = {'Dataset': dataset_name, 'Demo': demo_name}
                demo_data.update(demo_stats)
                all_demos_data.append(demo_data)

    if all_demos_data:
        demos_df = pd.DataFrame(all_demos_data)
        demos_df.to_csv(os.path.join(output_dir, 'all_demos_statistics.csv'), index=False)
        print(f"Detailed demo statistics saved to {os.path.join(output_dir, 'all_demos_statistics.csv')}")

    # Save errors and skipped files for each dataset
    error_data = []
    skipped_data = []

    for dataset_name, stats in all_stats.items():
        # Add errors
        for error in stats['errors']:
            error_data.append({
                'Dataset': dataset_name,
                'Error': error
            })

        # Add skipped files
        for skipped in stats['skipped_files']:
            skipped_data.append({
                'Dataset': dataset_name,
                'Skipped File': skipped
            })

    # Save errors
    if error_data:
        error_df = pd.DataFrame(error_data)
        error_df.to_csv(os.path.join(output_dir, 'all_errors.csv'), index=False)
        print(f"Error list saved to {os.path.join(output_dir, 'all_errors.csv')}")

    # Save skipped files
    if skipped_data:
        skipped_df = pd.DataFrame(skipped_data)
        skipped_df.to_csv(os.path.join(output_dir, 'all_skipped_files.csv'), index=False)
        print(f"Skipped files list saved to {os.path.join(output_dir, 'all_skipped_files.csv')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process bag files and convert to HDF5.')
    parser.add_argument('--debug', action='store_true', help='Debug mode - process only 1 bag file')
    args = parser.parse_args()

    debug_mode = args.debug

    print("=== Creating all dataset combinations ===")

    # Dictionary to collect statistics from all runs
    all_stats = {}

    # Scenario 1: Sim only (simulator data with all human + non-human demos)
    print("\n[1/4] Processing simulator data only...")
    dataset_name, stats = process_bag_files(
        process_sim=True,
        process_lab=False,
        process_human=None,  # Include both human and non-human demos
        subset=False,
        debug=debug_mode
    )
    all_stats[dataset_name] = stats

    # Scenario 2: Real only (lab data with all human + non-human demos)
    print("\n[2/4] Processing real/lab data only...")
    dataset_name, stats = process_bag_files(
        process_sim=False,
        process_lab=True,
        process_human=None,  # Include both human and non-human demos
        subset=False,
        debug=debug_mode
    )
    all_stats[dataset_name] = stats

    # Scenario 3: Sim + Real (all data with all human + non-human demos)
    print("\n[3/4] Processing simulator + real/lab data...")
    dataset_name, stats = process_bag_files(
        process_sim=True,
        process_lab=True,
        process_human=None,  # Include both human and non-human demos
        subset=False,
        debug=debug_mode
    )
    all_stats[dataset_name] = stats

    # Scenario 4: Sim + Subset Real (simulator + half of lab data with all human + non-human demos)
    print("\n[4/4] Processing simulator + subset of real/lab data...")
    dataset_name, stats = process_bag_files(
        process_sim=True,
        process_lab=True,
        process_human=None,  # Include both human and non-human demos
        subset=True,
        debug=debug_mode
    )
    all_stats[dataset_name] = stats

    # Save all statistics to organized CSV files
    output_dir = "record/converted_standard"
    save_statistics(output_dir, all_stats)
    print(all_stats)

    print("\n=== All dataset combinations created successfully ===")
    print(f"Statistics saved in {output_dir}")
