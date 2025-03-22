#!/usr/bin/env python3
import rosbag
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
import os
import glob
import argparse
from collections import deque


def compute_reward(d, phi, velocity, in_lane):
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
    if velocity == 0:
        return gamma  # reward for stopping at a red light or stop sign
    reward = r_d + r_phi + r_v + r_lane
    return reward


def extract_bag_data(bag_path):
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
        'rewards': []
    }

    # Open the bag file and extract data
    with rosbag.Bag(bag_path, 'r') as bag:
        # Extract car commands
        for topic, msg, t in bag.read_messages(topics=['/chicinvabot/car_cmd_switch_node/cmd']):
            data['cmd_timestamps'].append(t.to_sec())
            data['cmd_v'].append(msg.v)
            data['cmd_omega'].append(msg.omega)

        # Extract lane pose information
        for topic, msg, t in bag.read_messages(topics=['/chicinvabot/lane_filter_node/lane_pose']):
            data['lane_timestamps'].append(t.to_sec())
            data['lane_d'].append(msg.d)
            data['lane_phi'].append(msg.phi)

            # Check if in_lane attribute exists, otherwise use d threshold
            try:
                in_lane = msg.in_lane
            except AttributeError:
                in_lane = abs(msg.d) < 0.3  # Threshold based on lateral offset

            data['lane_in_lane'].append(1 if in_lane else 0)

            # Get curvature for reward calculation (not plotted)
            try:
                curvature = msg.curvature
            except AttributeError:
                curvature = 0.0

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
        else:
            start_time = min(start_time, min(data['lane_timestamps']))
        data['lane_timestamps'] = [t - start_time for t in data['lane_timestamps']]

    return data


def plot_combined_data(all_data, output_dir):
    """Create a combined plot with data from multiple bag files"""
    # Create a figure with subplots
    plt.figure(figsize=(15, 15))
    gs = GridSpec(6, 1, height_ratios=[1, 1, 1, 1, 1, 1])

    # Color cycle for different bag files
    colors = ['b', 'r', 'g', 'm', 'c', 'y', 'k', 'orange', 'purple', 'brown']

    # Plot linear velocity (v)
    ax1 = plt.subplot(gs[0])
    for i, data in enumerate(all_data):
        color = colors[i % len(colors)]
        ax1.plot(data['cmd_timestamps'], data['cmd_v'], color=color, linewidth=2,
                 label=data['name'])
    ax1.set_ylabel('Linear Velocity (v)')
    ax1.set_title('Car Commands - Linear Velocity')
    ax1.legend(loc='best', fontsize='small')
    ax1.grid(True)

    # Plot angular velocity (omega)
    ax2 = plt.subplot(gs[1])
    for i, data in enumerate(all_data):
        color = colors[i % len(colors)]
        ax2.plot(data['cmd_timestamps'], data['cmd_omega'], color=color, linewidth=2,
                 label=data['name'])
    ax2.set_ylabel('Angular Velocity (omega)')
    ax2.set_title('Car Commands - Angular Velocity')
    ax2.legend(loc='best', fontsize='small')
    ax2.grid(True)

    # Plot lateral offset (d)
    ax3 = plt.subplot(gs[2])
    for i, data in enumerate(all_data):
        color = colors[i % len(colors)]
        ax3.plot(data['lane_timestamps'], data['lane_d'], color=color, linewidth=2,
                 label=data['name'])
    ax3.set_ylabel('Lateral Offset (d)')
    ax3.set_title('Lane Position - Lateral Offset')
    ax3.legend(loc='best', fontsize='small')
    ax3.grid(True)

    # Plot heading error (phi)
    ax4 = plt.subplot(gs[3])
    for i, data in enumerate(all_data):
        color = colors[i % len(colors)]
        ax4.plot(data['lane_timestamps'], data['lane_phi'], color=color, linewidth=2,
                 label=data['name'])
    ax4.set_ylabel('Heading Error (phi)')
    ax4.set_title('Lane Position - Heading Error')
    ax4.legend(loc='best', fontsize='small')
    ax4.grid(True)

    # Plot rewards
    ax5 = plt.subplot(gs[4])
    for i, data in enumerate(all_data):
        color = colors[i % len(colors)]
        ax5.plot(data['lane_timestamps'], data['rewards'], color=color, linewidth=2,
                 label=data['name'])
    ax5.set_ylabel('Reward')
    ax5.set_title('Reward')
    ax5.legend(loc='best', fontsize='small')
    ax5.grid(True)

    # Plot in_lane status and error status
    ax6 = plt.subplot(gs[5])
    for i, data in enumerate(all_data):
        color = colors[i % len(colors)]
        # Plot in_lane as solid line
        ax6.plot(data['lane_timestamps'], data['lane_in_lane'], color=color, linewidth=2,
                 label=f"{data['name']} - In Lane")

    ax6.set_ylabel('Status (1=Error/In Lane, 0=Normal/Not in Lane)')
    ax6.set_title('Lane Status (solid=in_lane, dashed=error_status)')
    ax6.set_xlabel('Time (seconds)')
    ax6.legend(loc='best', fontsize='small')
    ax6.grid(True)

    # Add a main title for the entire figure
    plt.suptitle('Combined Duckiebot Data Analysis', fontsize=16)

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)  # Make room for suptitle

    # Save figure
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(os.path.join(output_dir, 'combined_analysis.png'), dpi=300)

    print(f"Generated combined plot in {output_dir}")

    # Also show the plot if running interactively
    plt.show()


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Process ROS bag files to generate combined plot')
    parser.add_argument('input_dir', help='Directory containing bag files')
    parser.add_argument('--output_dir', default='plots', help='Directory to save plots (default: plots)')
    parser.add_argument('--max_files', type=int, default=10, help='Maximum number of files to process (default: 10)')
    args = parser.parse_args()

    # Find all bag files in the directory
    bag_files = glob.glob(os.path.join(args.input_dir, '*.bag'))

    if not bag_files:
        print(f"No bag files found in {args.input_dir}")
        return

    # Limit number of files if needed
    if len(bag_files) > args.max_files:
        print(f"Found {len(bag_files)} bag files, limiting to {args.max_files} as specified")
        bag_files = bag_files[:args.max_files]
    else:
        print(f"Found {len(bag_files)} bag files")

    # Process each bag file and collect data
    all_data = []
    for bag_file in bag_files:
        data = extract_bag_data(bag_file)
        all_data.append(data)

    # Create combined plot
    plot_combined_data(all_data, args.output_dir)

    print(f"Processed {len(all_data)} bag files. Combined plot saved to {args.output_dir}")


if __name__ == "__main__":
    main()
