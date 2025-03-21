import cv2
import rosbag
from cv_bridge import CvBridge
from collections import deque

# Define reward function (same as in your original code)
def compute_reward(lane_pose, velocity, in_lane):
    # Reference values
    d_ref = 0.0  # Ideal lateral offset
    phi_ref = 0.0  # Ideal heading angle
    curvature_ref = 0.0  # Ideal curvature

    # Tunable weights
    alpha = 1.0
    beta = 0.5
    gamma = 2.0
    delta = 0.2
    lambda_ = 10.0  # High penalty for leaving the lane

    # Extract lane pose values
    d = lane_pose.d
    phi = lane_pose.phi
    curvature = getattr(lane_pose, 'curvature', curvature_ref)  # Use default if not available

    # Compute reward components
    r_d = -alpha * abs(d - d_ref)
    r_phi = -beta * abs(phi - phi_ref)
    r_v = gamma * velocity  # Reward movement
    r_c = -delta * abs(curvature - curvature_ref)
    r_lane = -lambda_ if not in_lane else 0  # High penalty for being out of lane

    # Total reward
    reward = r_d + r_phi + r_v + r_c + r_lane
    return reward

# Bag file path
bag_path = "record/lab_record/human_control_light_50/chicinvabot_2025-03-12-18-53-48.bag"

# Define topics
image_topic = "/chicinvabot/camera_node/image/compressed"
data_topic = "/chicinvabot/car_cmd_switch_node/cmd"
accept_topic = "/chicinvabot/car_cmd_switch_node/cmd_executed"
lane_pose_topic = "/chicinvabot/lane_filter_node/lane_pose"  # LanePose topic

# Open bag file
bag = rosbag.Bag(bag_path)
bridge = CvBridge()

# Buffers for synchronizing messages
data_buffer = deque()
accept_buffer = deque()
lane_pose_buffer = deque()

# Read messages
for topic, msg, t in bag.read_messages(topics=[image_topic, data_topic, accept_topic, lane_pose_topic]):
    timestamp = t.to_sec()  # Convert ROS time to seconds

    if topic == data_topic:
        print(f"Velocity Msg - Time: {timestamp:.3f}, vel: {msg.v:.3f}, omega: {msg.omega:.3f}")
        data_buffer.append((timestamp, msg))
        if len(data_buffer) > 10:
            data_buffer.popleft()

    if topic == accept_topic:
        print(f"Executed Velocity Msg - Time: {timestamp:.3f}, vel: {msg.v:.3f}, omega: {msg.omega:.3f}")
        accept_buffer.append((timestamp, msg))
        if len(accept_buffer) > 10:
            accept_buffer.popleft()

    if topic == lane_pose_topic:
        print(f"Lane Pose - Time: {timestamp:.3f}, d: {msg.d:.3f}, phi: {msg.phi:.3f}, v_ref: {msg.v_ref:.3f}")
        lane_pose_buffer.append((timestamp, msg))
        if len(lane_pose_buffer) > 10:
            lane_pose_buffer.popleft()

    elif topic == image_topic:
        print(f"Image Msg - Time: {timestamp:.3f}")
        frame = bridge.compressed_imgmsg_to_cv2(msg, "bgr8")

        # Find closest velocity message
        latest_data = "N/A"
        current_velocity = 0.0
        if data_buffer:
            closest_time, closest_msg = min(data_buffer, key=lambda x: abs(x[0] - timestamp))
            print(f"Closest Velocity Msg - Time: {closest_time:.3f}, Diff: {abs(closest_time - timestamp):.3f}")
            current_velocity = closest_msg.v
            latest_data = f"Vel: {round(closest_msg.v, 3)} Omega: {round(closest_msg.omega, 3)}"

        cv2.putText(frame, latest_data, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Find closest executed velocity message
        latest_exec_data = "N/A"
        if accept_buffer:
            closest_time, closest_msg = min(accept_buffer, key=lambda x: abs(x[0] - timestamp))
            print(f"Closest Executed Velocity Msg - Time: {closest_time:.3f}, Diff: {abs(closest_time - timestamp):.3f}")
            latest_exec_data = f"Vel Exec: {round(closest_msg.v, 3)} Omega Exec: {round(closest_msg.omega, 3)}"

        cv2.putText(frame, latest_exec_data, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Find closest LanePose message
        latest_lane_pose = "N/A"
        current_reward = "N/A"
        if lane_pose_buffer:
            closest_time, closest_lane_msg = min(lane_pose_buffer, key=lambda x: abs(x[0] - timestamp))
            print(f"Closest Lane Pose - Time: {closest_time:.3f}, Diff: {abs(closest_time - timestamp):.3f}")

            # Extract in_lane status (you might need to adjust based on your ROS message format)
            in_lane = getattr(closest_lane_msg, 'in_lane', abs(closest_lane_msg.d) < 0.3)

            # Calculate reward
            reward = compute_reward(closest_lane_msg, current_velocity, in_lane)
            current_reward = f"Reward: {reward:.3f}"

            latest_lane_pose = (
                f"d: {round(closest_lane_msg.d, 3)} phi: {round(closest_lane_msg.phi, 3)} "
                f"v_ref: {round(closest_lane_msg.v_ref, 3)} In Lane: {in_lane}"
            )

        cv2.putText(frame, latest_lane_pose, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # Display reward
        if current_reward != "N/A":
            # Use a different color for reward to make it stand out
            cv2.putText(frame, current_reward, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Display frame
        cv2.imshow("ROS Video with Data and Reward", frame)

        if cv2.waitKey(33) & 0xFF == ord("q"):
            break

# Cleanup
bag.close()
cv2.destroyAllWindows()
