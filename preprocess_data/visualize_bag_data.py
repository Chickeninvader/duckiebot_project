from time import sleep

import cv2
import rosbag
import numpy as np
from cv_bridge import CvBridge
from collections import deque
import os
import argparse
from utils import compute_reward

def process_bag_file(bag_path, output_video=None, display=True, env_type="real"):
    """Process a bag file, display video with data, and optionally save to a file"""
    # Determine bot name based on environment type
    bot_name = "chicinvabot" if env_type == "real" else "vchicinvabot"

    # Define topics with bot name
    image_topic = f"/{bot_name}/camera_node/image/compressed"
    data_topic = f"/{bot_name}/car_cmd_switch_node/cmd"
    accept_topic = f"/{bot_name}/car_cmd_switch_node/cmd_executed"
    lane_pose_topic = f"/{bot_name}/lane_filter_node/lane_pose"

    print(f"Processing {bag_path} with bot name: {bot_name}")
    print(f"Using topics: \n  {image_topic}\n  {data_topic}\n  {accept_topic}\n  {lane_pose_topic}")

    # Open bag file
    bag = rosbag.Bag(bag_path)
    bridge = CvBridge()

    # Buffers for synchronizing messages
    data_buffer = deque(maxlen=20)
    accept_buffer = deque(maxlen=20)
    lane_pose_buffer = deque(maxlen=20)

    # Create output video writer if requested
    video_writer = None

    # Track frames for reward history
    reward_history = deque(maxlen=100)
    time_history = deque(maxlen=100)
    lane_d_history = deque(maxlen=100)
    lane_phi_history = deque(maxlen=100)

    # Get all timestamps from the bag to calculate progress
    total_timestamps = []
    for _, _, t in bag.read_messages():
        total_timestamps.append(t.to_sec())

    start_time = min(total_timestamps) if total_timestamps else 0
    end_time = max(total_timestamps) if total_timestamps else 0
    duration = end_time - start_time

    # Setup CSV for data logging
    csv_filename = os.path.splitext(bag_path)[0] + "_analysis.csv"
    csv_file = open(csv_filename, 'w')
    csv_file.write("timestamp,cmd_v,cmd_omega,executed_v,executed_omega,lane_d,lane_phi,in_lane,reward\n")

    # First pass to get video dimensions
    frame_shape = None
    for topic, msg, _ in bag.read_messages(topics=[image_topic]):
        if topic == image_topic:
            frame = bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
            frame_shape = frame.shape
            break

    # Setup video writer if dimensions are found
    if output_video and frame_shape:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_writer = cv2.VideoWriter(output_video, fourcc, 30.0,
                                       (frame_shape[1], frame_shape[0]))

    # Read messages
    current_progress = 0
    frames_processed = 0

    for topic, msg, t in bag.read_messages(topics=[image_topic, data_topic, accept_topic, lane_pose_topic]):
        sleep(0.01)  # Small delay to allow for processing
        timestamp = t.to_sec()
        progress = int(((timestamp - start_time) / duration) * 100) if duration > 0 else 0

        if progress > current_progress:
            current_progress = progress
            print(f"Progress: {current_progress}%")

        if topic == data_topic:
            data_buffer.append((timestamp, msg))

        elif topic == accept_topic:
            accept_buffer.append((timestamp, msg))

        elif topic == lane_pose_topic:
            lane_pose_buffer.append((timestamp, msg))

            # Check if lane departure occurred
            try:
                in_lane = msg.in_lane
            except AttributeError:
                in_lane = abs(msg.d) < 0.3

            if not in_lane:
                print(f"Lane departure detected at timestamp {timestamp:.3f}")

        elif topic == image_topic:
            frames_processed += 1
            frame = bridge.compressed_imgmsg_to_cv2(msg, "bgr8")

            # Find closest velocity message
            current_velocity = 0.0
            cmd_omega = 0.0
            if data_buffer:
                closest_time, closest_msg = min(data_buffer, key=lambda x: abs(x[0] - timestamp))
                time_diff = abs(closest_time - timestamp)
                if time_diff < 0.1:  # Only use if within 100ms
                    current_velocity = closest_msg.v
                    cmd_omega = closest_msg.omega

            # Find closest executed velocity message
            executed_v = 0.0
            executed_omega = 0.0
            if accept_buffer:
                closest_time, closest_msg = min(accept_buffer, key=lambda x: abs(x[0] - timestamp))
                time_diff = abs(closest_time - timestamp)
                if time_diff < 0.1:  # Only use if within 100ms
                    executed_v = closest_msg.v
                    executed_omega = closest_msg.omega

            # Find closest LanePose message
            lane_d = 0.0
            lane_phi = 0.0
            in_lane = True
            current_reward = None

            if lane_pose_buffer:
                closest_time, closest_lane_msg = min(lane_pose_buffer, key=lambda x: abs(x[0] - timestamp))
                time_diff = abs(closest_time - timestamp)

                if time_diff < 0.1:  # Only use if within 100ms
                    lane_d = closest_lane_msg.d
                    lane_phi = closest_lane_msg.phi

                    # Extract in_lane status
                    try:
                        in_lane = closest_lane_msg.in_lane
                    except AttributeError:
                        in_lane = abs(lane_d) < 0.3

                    # Calculate reward using the provided function
                    current_reward = compute_reward(lane_d, lane_phi, current_velocity, in_lane)

                    # Update histories
                    reward_history.append(current_reward)
                    time_history.append(timestamp - start_time)
                    lane_d_history.append(lane_d)
                    lane_phi_history.append(lane_phi)

            # Write to CSV
            csv_file.write(f"{timestamp},{current_velocity},{cmd_omega},{executed_v},{executed_omega},"
                           f"{lane_d},{lane_phi},{1 if in_lane else 0},{current_reward if current_reward is not None else ''}\n")

            # Create a clean copy of the frame for visualization
            display_frame = frame.copy()

            # Add a semi-transparent overlay at the top for text
            overlay = display_frame.copy()
            cv2.rectangle(overlay, (0, 0), (display_frame.shape[1], 120), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, display_frame, 0.3, 0, display_frame)

            # Add timestamp and frame number
            cv2.putText(display_frame, f"Time: {timestamp - start_time:.2f}s  Frame: {frames_processed}",
                        (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Add velocity commands
            velocity_text = f"Commanded: v={current_velocity:.3f} omega={cmd_omega:.3f}"
            cv2.putText(display_frame, velocity_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Add executed velocity
            executed_text = f"Executed: v={executed_v:.3f} omega={executed_omega:.3f}"
            cv2.putText(display_frame, executed_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            # Add lane pose data
            lane_text = f"Lane: d={lane_d:.3f} phi={lane_phi:.3f} In Lane: {'Yes' if in_lane else 'No'}"
            lane_color = (0, 255, 0) if in_lane else (0, 0, 255)  # Green if in lane, red if not
            cv2.putText(display_frame, lane_text, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, lane_color, 1)

            # Add reward
            if current_reward is not None:
                reward_text = f"Reward: {current_reward:.3f}"
                cv2.putText(display_frame, reward_text, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            # Draw reward graph in bottom right corner if we have history
            if reward_history:
                graph_width = 200
                graph_height = 100
                graph_x = display_frame.shape[1] - graph_width - 10
                graph_y = display_frame.shape[0] - graph_height - 10

                # Draw graph background
                cv2.rectangle(display_frame, (graph_x, graph_y),
                              (graph_x + graph_width, graph_y + graph_height), (0, 0, 0), -1)
                cv2.rectangle(display_frame, (graph_x, graph_y),
                              (graph_x + graph_width, graph_y + graph_height), (255, 255, 255), 1)

                # Draw rewards
                reward_min = min(reward_history) if reward_history else -2
                reward_max = max(reward_history) if reward_history else 2
                reward_range = max(0.1, reward_max - reward_min)  # Avoid division by zero

                # Draw points and connect them
                last_x, last_y = None, None
                for i in range(len(reward_history)):
                    norm_x = int(graph_x + (i / len(reward_history)) * graph_width)
                    norm_y = int(graph_y + graph_height -
                                 ((reward_history[i] - reward_min) / reward_range) * graph_height)

                    cv2.circle(display_frame, (norm_x, norm_y), 2, (0, 255, 255), -1)

                    if last_x is not None and last_y is not None:
                        cv2.line(display_frame, (last_x, last_y), (norm_x, norm_y), (0, 255, 255), 1)

                    last_x, last_y = norm_x, norm_y

                # Add graph labels
                cv2.putText(display_frame, "Reward History", (graph_x, graph_y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                cv2.putText(display_frame, f"{reward_max:.1f}",
                            (graph_x - 25, graph_y + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                cv2.putText(display_frame, f"{reward_min:.1f}",
                            (graph_x - 25, graph_y + graph_height - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            # Display frame
            if display:
                cv2.imshow("ROS Bag Analysis", display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

            # Write to video file if requested
            if video_writer:
                video_writer.write(display_frame)

    # Cleanup
    bag.close()
    csv_file.close()
    if video_writer:
        video_writer.release()
    if display:
        cv2.destroyAllWindows()

    print(f"Processed {frames_processed} frames")
    print(f"Data saved to {csv_filename}")
    if output_video:
        print(f"Video saved to {output_video}")

    return frames_processed


def main():
    parser = argparse.ArgumentParser(description='Process and visualize ROS bag files with reward calculation')
    parser.add_argument('bag_path', help='Path to the bag file or directory containing bag files')
    parser.add_argument('--output_video', help='Output video file path (optional)')
    parser.add_argument('--no_display', action='store_true', help='Disable video display')
    parser.add_argument('--env_type', choices=['real', 'sim'], default='real',
                        help='Environment type (real or sim) to determine bot name')
    args = parser.parse_args()

    # Check if path is a directory or a file
    if os.path.isdir(args.bag_path):
        # Process all bag files in directory
        bag_files = [f for f in os.listdir(args.bag_path) if f.endswith('.bag')]

        if not bag_files:
            print(f"No bag files found in {args.bag_path}")
            return

        print(f"Found {len(bag_files)} bag files to process")

        for bag_file in bag_files:
            bag_path = os.path.join(args.bag_path, bag_file)

            # Determine output video path if needed
            output_video = None
            if args.output_video:
                base_name = os.path.splitext(bag_file)[0]
                output_dir = os.path.dirname(args.output_video)
                if output_dir and not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                output_video = os.path.join(output_dir if output_dir else '', f"{base_name}_analysis.avi")

            process_bag_file(bag_path, output_video, not args.no_display, args.env_type)
    else:
        # Process single bag file
        if not os.path.exists(args.bag_path):
            print(f"Bag file not found: {args.bag_path}")
            return

        process_bag_file(args.bag_path, args.output_video, not args.no_display, args.env_type)


if __name__ == "__main__":
    main()
