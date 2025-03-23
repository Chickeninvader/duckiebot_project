#!/usr/bin/env python3
import rosbag
import numpy as np
import os
import glob
import argparse
import csv
import sys
from datetime import datetime
from utils import compute_reward  # Assuming you have a compute_reward function in utils.py


def extract_bag_data(bag_path, bot_name):
    """Extract data from a bag file and return as a dictionary"""
    bag_name = os.path.basename(bag_path).split('.')[0]
    print(f"Processing {bag_path}...")

    # Dictionary to store data
    data = {
        'name': bag_name,
        'cmd_timestamps': [],
        'cmd_v': [],
        'cmd_omega': [],
        'lane_timestamps': [],
        'lane_d': [],
        'lane_phi': [],
        'lane_in_lane': [],
        'rewards': [],
        'terminated_early': False
    }

    # Topics with bot name
    cmd_topic = f'/{bot_name}/car_cmd_switch_node/cmd'
    lane_topic = f'/{bot_name}/lane_filter_node/lane_pose'

    # Open the bag file and extract data
    try:
        with rosbag.Bag(bag_path, 'r') as bag:
            # Extract car commands
            for topic, msg, t in bag.read_messages(topics=[cmd_topic]):
                data['cmd_timestamps'].append(t.to_sec())
                data['cmd_v'].append(msg.v)
                data['cmd_omega'].append(msg.omega)

            # Extract lane pose information
            for topic, msg, t in bag.read_messages(topics=[lane_topic]):
                data['lane_timestamps'].append(t.to_sec())
                data['lane_d'].append(msg.d)
                data['lane_phi'].append(msg.phi)

                # Check if in_lane attribute exists, otherwise use d threshold
                try:
                    in_lane = msg.in_lane
                except AttributeError:
                    in_lane = abs(msg.d) < 0.3  # Threshold based on lateral offset

                data['lane_in_lane'].append(1 if in_lane else 0)

                # Terminate processing if lane departure occurs
                if not in_lane:
                    print(f"Lane departure detected in {bag_name}, terminating processing")
                    data['terminated_early'] = True
                    break

    except Exception as e:
        print(f"Error processing {bag_path}: {e}")
        return None

    # Calculate rewards if we have both command and lane data
    if data['cmd_timestamps'] and data['lane_timestamps']:
        # Interpolate cmd_v to match lane_timestamps for reward calculation
        cmd_v_interp = np.interp(data['lane_timestamps'], data['cmd_timestamps'], data['cmd_v'])

        for i in range(len(data['lane_timestamps'])):
            reward = compute_reward(
                data['lane_d'][i],
                data['lane_phi'][i],
                cmd_v_interp[i],
                bool(data['lane_in_lane'][i])
            )
            data['rewards'].append(reward)

    # Normalize timestamps to start at 0
    if data['cmd_timestamps']:
        start_time = min(data['cmd_timestamps'])
        data['cmd_timestamps'] = [t - start_time for t in data['cmd_timestamps']]

    if data['lane_timestamps']:
        if not data['cmd_timestamps']:  # If no cmd_timestamps, set start_time from lane_timestamps
            start_time = min(data['lane_timestamps'])
        data['lane_timestamps'] = [t - start_time for t in data['lane_timestamps']]

    return data


def save_to_csv(data, output_dir, dataset_type):
    """Save the extracted data to CSV files"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create a timestamp to avoid overwriting files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save detailed data for each bag file
    for bag_data in data:
        if not bag_data:
            continue

        bag_name = bag_data['name']
        file_name = f"{dataset_type}_{bag_name}_{timestamp}.csv"
        file_path = os.path.join(output_dir, file_name)

        # Create a dataframe-like structure combining all timestamps
        all_data = []

        # Use lane timestamps as the base
        for i, t in enumerate(bag_data['lane_timestamps']):
            # Find closest cmd timestamp
            if bag_data['cmd_timestamps']:
                closest_cmd_idx = np.argmin(np.abs(np.array(bag_data['cmd_timestamps']) - t))
                cmd_v = bag_data['cmd_v'][closest_cmd_idx]
                cmd_omega = bag_data['cmd_omega'][closest_cmd_idx]
            else:
                cmd_v = None
                cmd_omega = None

            row = {
                'timestamp': t,
                'cmd_v': cmd_v,
                'cmd_omega': cmd_omega,
                'lane_d': bag_data['lane_d'][i],
                'lane_phi': bag_data['lane_phi'][i],
                'in_lane': bag_data['lane_in_lane'][i],
                'reward': bag_data['rewards'][i] if i < len(bag_data['rewards']) else None
            }
            all_data.append(row)

        # Write to CSV
        with open(file_path, 'w', newline='') as f:
            fieldnames = ['timestamp', 'cmd_v', 'cmd_omega', 'lane_d', 'lane_phi', 'in_lane', 'reward']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in all_data:
                writer.writerow(row)

        print(f"Saved data for {bag_name} to {file_path}")

    # Calculate statistics for summary
    num_demos = len([d for d in data if d is not None])
    total_frames = sum(len(d['lane_timestamps']) for d in data if d is not None)
    all_rewards = [r for d in data if d is not None for r in d['rewards']]
    avg_reward = np.mean(all_rewards) if all_rewards else 0
    total_time = sum(d['lane_timestamps'][-1] if d and d['lane_timestamps'] else 0 for d in data)

    # Save summary statistics
    summary_file_path = os.path.join(output_dir, f"{dataset_type}_summary_{timestamp}.csv")
    with open(summary_file_path, 'w', newline='') as f:
        fieldnames = ['Dataset', 'Bag Files Found', 'Demos Processed', 'Total Frames',
                      'Avg Reward', 'Total Demo Time (s)']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({
            'Dataset': dataset_type,
            'Bag Files Found': len(data),
            'Demos Processed': num_demos,
            'Total Frames': total_frames,
            'Avg Reward': avg_reward,
            'Total Demo Time (s)': total_time
        })

    print(f"Saved summary statistics to {summary_file_path}")

    return {
        'num_bag_files': len(data),
        'num_demos': num_demos,
        'skipped_files': [d['name'] for d in data if d is None],
        'errors': [],
        'total_frames': total_frames,
        'avg_reward': avg_reward,
        'total_demo_time': total_time
    }


def process_dataset(base_dir, environment, output_dir):
    """Process all bag files in a specific environment directory"""
    env_dir = os.path.join(base_dir, environment)
    if not os.path.exists(env_dir):
        print(f"Directory not found: {env_dir}")
        return None

    # Determine bot name based on environment
    bot_name = "chicinvabot" if environment == "real" else "vchicinvabot"

    # Find all bag files in the directory
    bag_files = glob.glob(os.path.join(env_dir, '*.bag'))

    if not bag_files:
        print(f"No bag files found in {env_dir}")
        return None

    print(f"Found {len(bag_files)} bag files in {environment} environment")

    # Process each bag file and collect data
    all_data = []
    for bag_file in bag_files:
        data = extract_bag_data(bag_file, bot_name)
        all_data.append(data)

    # Save data to CSV
    stats = save_to_csv(all_data, output_dir, environment)

    return stats


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Process ROS bag files and save data to CSV')
    parser.add_argument('--record_dir', default="record/evaluation", help='Base directory containing record/evaluation folders')
    parser.add_argument('--output_dir', default='record/evaluation', help='Directory to save CSV files (default: csv_output)')
    args = parser.parse_args()

    # Define the base record directory
    base_dir = args.record_dir

    # Process sim and real environments
    environments = ["sim", "real"]
    summary_data = []

    for env in environments:
        print(f"\nProcessing {env} environment...")
        stats = process_dataset(base_dir, env, args.output_dir)
        if stats:
            summary_data.append({
                'Dataset': env,
                'Bag Files Found': stats['num_bag_files'],
                'Demos Processed': stats['num_demos'],
                'Files Skipped': len(stats['skipped_files']),
                'Errors': len(stats['errors']),
                'Total Frames': stats['total_frames'],
                'Avg Reward': stats['avg_reward'],
                'Total Demo Time (s)': stats['total_demo_time']
            })

    # Print overall summary
    print("\nOverall Summary:")
    for summary in summary_data:
        print(f"Dataset: {summary['Dataset']}")
        print(f"  Bag Files Found: {summary['Bag Files Found']}")
        print(f"  Demos Processed: {summary['Demos Processed']}")
        print(f"  Files Skipped: {summary['Files Skipped']}")
        print(f"  Errors: {summary['Errors']}")
        print(f"  Total Frames: {summary['Total Frames']}")
        print(f"  Avg Reward: {summary['Avg Reward']:.4f}")
        print(f"  Total Demo Time: {summary['Total Demo Time (s)']:.2f} seconds")


if __name__ == "__main__":
    main()
