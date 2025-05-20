from time import sleep # Kept for potential future use, though not actively used for delay
import cv2
import rosbag
import numpy as np
from cv_bridge import CvBridge
from collections import deque
import os
import argparse

# Assume utils.py and compute_reward are in the same directory or accessible
from utils import compute_reward

def process_bag_file(bag_path, output_video=None, display=True, env_type="real",
                     display_source_topic_name="camera", show_fsm_state=True):
    """Process a bag file, display video with data, and optionally save to a file. Simplified version."""

    bot_name = "chicinvabot" if env_type == "real" else "vchicinvabot"

    camera_image_topic = f"/{bot_name}/camera_node/image/compressed"
    ground_projection_topic = f"/{bot_name}/ground_projection_node/debug/ground_projection_image/compressed"
    data_topic = f"/{bot_name}/car_cmd_switch_node/cmd"
    accept_topic = f"/{bot_name}/car_cmd_switch_node/cmd_executed"
    lane_pose_topic = f"/{bot_name}/lane_filter_node/lane_pose"
    state_topic = f"/{bot_name}/fsm_node/mode"

    if display_source_topic_name == "camera":
        image_topic_to_display = camera_image_topic
    elif display_source_topic_name == "ground_projection":
        image_topic_to_display = ground_projection_topic
    else: # Default to camera if invalid source provided
        image_topic_to_display = camera_image_topic

    topics_to_read = [image_topic_to_display, data_topic, accept_topic, lane_pose_topic]
    if show_fsm_state:
        topics_to_read.append(state_topic)

    bag = rosbag.Bag(bag_path) # Assume bag_path is valid and bag is readable
    bridge = CvBridge()

    data_buffer = deque(maxlen=20)
    accept_buffer = deque(maxlen=20)
    lane_pose_buffer = deque(maxlen=20)
    state_buffer = deque(maxlen=10)

    video_writer = None
    reward_history = deque(maxlen=100)

    # Assume messages exist for timestamp calculation
    # This will raise ValueError if topics_to_read leads to no messages, per simplification.
    total_timestamps = [t.to_sec() for _, _, t in bag.read_messages(topics=topics_to_read)]
    if not total_timestamps: # Minimal check to prevent crash on min/max with empty list
        bag.close()
        return 0


    start_time = min(total_timestamps)
    end_time = max(total_timestamps)
    # duration = end_time - start_time # Not explicitly used after this simplification

    frame_shape = None
    # Assume the display topic has at least one message and it's a decodable image
    for topic, msg, _ in bag.read_messages(topics=[image_topic_to_display]): # Only read this topic for shape
        if topic == image_topic_to_display:
            if msg._type == "sensor_msgs/Image":
                frame = bridge.imgmsg_to_cv2(msg, "bgr8")
            elif msg._type == "sensor_msgs/CompressedImage":
                frame = bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
            else:
                # Assuming valid image type, so this branch ideally isn't hit
                continue
            frame_shape = frame.shape
            break

    # "Assume everything is there" implies frame_shape will be set.
    # If not, VideoWriter might fail or use default/no dimensions.

    if output_video and frame_shape:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_writer = cv2.VideoWriter(output_video, fourcc, 30.0,
                                       (frame_shape[1], frame_shape[0]))

    frames_processed = 0
    temp_not_in_lane_count = 0
    latest_fsm_state = "N/A" # Default FSM state

    for topic, msg, t in bag.read_messages(topics=topics_to_read):
        timestamp = t.to_sec()

        if topic == data_topic:
            data_buffer.append((timestamp, msg))
        elif topic == accept_topic:
            accept_buffer.append((timestamp, msg))
        elif topic == lane_pose_topic:
            lane_pose_buffer.append((timestamp, msg))
        elif topic == state_topic and show_fsm_state:
            # Assuming msg.data for std_msgs/String or similar
            latest_fsm_state = msg.state
            state_buffer.append((timestamp, latest_fsm_state))

        elif topic == image_topic_to_display:
            frames_processed += 1
            if msg._type == "sensor_msgs/Image":
                frame = bridge.imgmsg_to_cv2(msg, "bgr8")
            elif msg._type == "sensor_msgs/CompressedImage":
                frame = bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
            else:
                continue # Skip if not a recognized image type

            current_fsm_state_display = "N/A"
            if show_fsm_state and state_buffer:
                # The last element in the buffer is the latest received state
                latest_time_state, latest_msg_state = state_buffer[-1]
                current_fsm_state_display = latest_msg_state
            elif show_fsm_state and latest_fsm_state: # If buffer is empty but we have a latest state
                current_fsm_state_display = latest_fsm_state

            current_cmd_v = 0.0
            current_cmd_omega = 0.0
            if data_buffer:
                closest_time_data, closest_msg_data = min(data_buffer, key=lambda x: abs(x[0] - timestamp))
                if abs(closest_time_data - timestamp) < 0.1:
                    current_cmd_v = closest_msg_data.v
                    current_cmd_omega = closest_msg_data.omega

            executed_v = 0.0
            executed_omega = 0.0
            if accept_buffer:
                closest_time_accept, closest_msg_accept = min(accept_buffer, key=lambda x: abs(x[0] - timestamp))
                if abs(closest_time_accept - timestamp) < 0.1:
                    executed_v = closest_msg_accept.v
                    executed_omega = closest_msg_accept.omega

            lane_d = 0.0
            lane_phi = 0.0
            in_lane = True # Default
            current_reward = None

            if lane_pose_buffer:
                closest_time_lane, closest_lane_msg = min(lane_pose_buffer, key=lambda x: abs(x[0] - timestamp))
                if abs(closest_time_lane - timestamp) < 0.1:
                    lane_d = closest_lane_msg.d
                    lane_phi = closest_lane_msg.phi

                    # Simplified in_lane logic based on assumption of attribute presence or fallback
                    in_lane_attr = getattr(closest_lane_msg, 'in_lane', abs(lane_d) < 0.3) # Default to calc if attr missing
                    if isinstance(in_lane_attr, (bool, np.bool_)):
                        in_lane = in_lane_attr
                    elif hasattr(in_lane_attr, 'data') and isinstance(in_lane_attr.data, bool): # For std_msgs/Bool like
                         in_lane = in_lane_attr.data
                    else: # Fallback if it's not a direct boolean or wrapped boolean
                        in_lane = abs(lane_d) < 0.3


                    if not in_lane:
                        temp_not_in_lane_count += 1
                    else:
                        temp_not_in_lane_count = 0

                    current_reward = compute_reward(lane_d, lane_phi, current_cmd_v, temp_not_in_lane_count < 3)
                    reward_history.append(current_reward)

            display_frame = frame.copy()
            text_area_height = 120
            if show_fsm_state:
                text_area_height += 20

            overlay = display_frame.copy()
            cv2.rectangle(overlay, (0, 0), (display_frame.shape[1], text_area_height), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, display_frame, 0.3, 0, display_frame)

            current_y_offset = 20
            if show_fsm_state:
                fsm_text = f"FSM State: {current_fsm_state_display}"
                cv2.putText(display_frame, fsm_text, (10, current_y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 255), 1)
                current_y_offset += 25

            cv2.putText(display_frame, f"T: {timestamp - start_time:.2f}s Fr: {frames_processed}",
                        (10, current_y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            current_y_offset += 20
            cv2.putText(display_frame, f"Cmd: v={current_cmd_v:.3f} w={current_cmd_omega:.3f}", (10, current_y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            current_y_offset += 20
            cv2.putText(display_frame, f"Exe: v={executed_v:.3f} w={executed_omega:.3f}", (10, current_y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            current_y_offset += 20
            lane_text = f"Lane: d={lane_d:.3f} p={lane_phi:.3f} In: {'Y' if in_lane else 'N'}" # Shortened labels
            lane_color = (0, 255, 0) if in_lane else (0, 0, 255)
            cv2.putText(display_frame, lane_text, (10, current_y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, lane_color, 1)
            current_y_offset += 20
            if current_reward is not None:
                cv2.putText(display_frame, f"Rwd: {current_reward:.3f}", (10, current_y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            if reward_history:
                graph_width = 200; graph_height = 100
                graph_x = display_frame.shape[1] - graph_width - 10
                graph_y = display_frame.shape[0] - graph_height - 10
                cv2.rectangle(display_frame, (graph_x, graph_y), (graph_x + graph_width, graph_y + graph_height), (50, 50, 50), -1)
                cv2.rectangle(display_frame, (graph_x, graph_y), (graph_x + graph_width, graph_y + graph_height), (200, 200, 200), 1)

                reward_min_val = min(reward_history) # Assumes reward_history is not empty here
                reward_max_val = max(reward_history)
                reward_range = max(0.1, reward_max_val - reward_min_val)

                points = []
                for i_rh, rh_val in enumerate(reward_history):
                    # Ensure division by non-zero for norm_x when len(reward_history) is 1
                    denominator_x = max(1, len(reward_history) -1) if len(reward_history) > 1 else 1
                    norm_x = int(graph_x + (i_rh / denominator_x) * graph_width)
                    norm_y = int(graph_y + graph_height - ((rh_val - reward_min_val) / reward_range) * graph_height)
                    points.append((norm_x, norm_y))

                if len(points) > 1:
                    cv2.polylines(display_frame, [np.array(points)], isClosed=False, color=(0, 255, 255), thickness=1)
                elif len(points) == 1: # Draw a single point if only one data point
                     cv2.circle(display_frame, points[0], 2, (0, 255, 255), -1)

                cv2.putText(display_frame, "Rwd Hist.", (graph_x, graph_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220,220,220),1)
                cv2.putText(display_frame, f"{reward_max_val:.1f}", (graph_x + graph_width + 5, graph_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220,220,220),1)
                cv2.putText(display_frame, f"{reward_min_val:.1f}", (graph_x + graph_width + 5, graph_y + graph_height - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220,220,220),1)

            if display:
                cv2.imshow("ROS Bag Analysis (Simplified)", display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'): break
                elif key == ord('p'): cv2.waitKey(-1) # Pause
                sleep(0.05)
            if video_writer:
                video_writer.write(display_frame) # Assume write is successful

    bag.close()
    if video_writer:
        video_writer.release()
    if display:
        cv2.destroyAllWindows()
    return frames_processed

def main():
    parser = argparse.ArgumentParser(description='Process and visualize ROS bag files (Simplified)')
    parser.add_argument('bag_path', help='Path to the bag file or directory containing bag files')
    parser.add_argument('--output_video', help='Output video file path (optional)')
    parser.add_argument('--no_display', action='store_true', help='Disable video display')
    parser.add_argument('--env_type', choices=['real', 'sim'], default='real',
                        help='Environment type (real or sim) to determine bot name')
    parser.add_argument('--display_source', choices=['camera', 'ground_projection'], default='camera',
                        help='Image topic to use for display (camera or ground_projection)')
    parser.add_argument('--hide_fsm_state', action='store_true', default=False,
                        help='Hide the FSM state from the display (default: show)')
    args = parser.parse_args()

    display_enabled = not args.no_display
    show_fsm_state_flag = not args.hide_fsm_state

    if os.path.isdir(args.bag_path):
        bag_files = sorted([f for f in os.listdir(args.bag_path) if f.endswith('.bag')])
        if not bag_files:
            print(f"No bag files found in directory: {args.bag_path}") # Kept for crucial feedback
            return

        for bag_file_name in bag_files:
            current_bag_path = os.path.join(args.bag_path, bag_file_name)
            current_output_video = None
            if args.output_video:
                # Simplified output video path generation for batch
                output_dir = os.path.dirname(args.output_video) if \
                             os.path.splitext(args.output_video)[1] else \
                             args.output_video # If extensionless, assume it's a dir

                if not output_dir : output_dir = "." # Default to current dir if empty

                if not os.path.exists(output_dir) and output_dir != ".":
                    os.makedirs(output_dir) # Assume makedirs is successful

                base_name = os.path.splitext(bag_file_name)[0]
                # Use a fixed suffix or derive from original output_video arg if it was a file name
                video_file_suffix = "_analysis"
                if os.path.splitext(args.output_video)[1] and not os.path.isdir(args.output_video): # if original arg was a file
                    video_file_suffix = "_" + os.path.splitext(os.path.basename(args.output_video))[0]

                video_ext = os.path.splitext(args.output_video)[1] if os.path.splitext(args.output_video)[1] else ".avi"
                current_output_video = os.path.join(output_dir, f"{base_name}{video_file_suffix}{video_ext}")

            process_bag_file(current_bag_path, current_output_video, display_enabled,
                             args.env_type, args.display_source, show_fsm_state_flag)
    else: # Single bag file
        if not os.path.exists(args.bag_path):
            print(f"Bag file not found: {args.bag_path}") # Kept for crucial feedback
            return
        process_bag_file(args.bag_path, args.output_video, display_enabled,
                         args.env_type, args.display_source, show_fsm_state_flag)

if __name__ == "__main__":
    main()
