import cv2
import rosbag
from cv_bridge import CvBridge
from collections import deque

# Bag file path
bag_path = "/data/logs/chicinvabot_2025-02-17-19-41-46.bag"

# Define topics
image_topic = "/chicinvabot/camera_node/image/compressed"
data_topic = "/chicinvabot/wheels_driver_node/wheels_cmd"
accept_topic = "/chicinvabot/wheels_driver_node/wheels_cmd_executed"

# Open bag file
bag = rosbag.Bag(bag_path)
bridge = CvBridge()

# Buffer for wheel command data with timestamps
data_buffer = deque()
accept_buffer = deque()

# Read messages
for topic, msg, t in bag.read_messages(topics=[image_topic, data_topic]):
    timestamp = t.to_sec()  # Convert ROS time to seconds

    if topic == data_topic:
        # Print wheel velocity message with timestamp
        print(f"Wheel Velocity Msg - Time: {timestamp:.3f}, vel_left: {msg.vel_left:.3f}, vel_right: {msg.vel_right:.3f}")

        # Store latest message with timestamp
        data_buffer.append((timestamp, msg))
        if len(data_buffer) > 10:  # Limit buffer size for efficiency
            data_buffer.popleft()
    
    if topic == accept_topic:
        # Print wheel velocity message with timestamp
        print(f"Wheel Velocity accept Msg - Time: {timestamp:.3f}, vel_left: {msg.vel_left:.3f}, vel_right: {msg.vel_right:.3f}")

        # Store latest message with timestamp
        accept_buffer.append((timestamp, msg))
        if len(accept_buffer) > 10:  # Limit buffer size for efficiency
            accept_buffer.popleft()

    elif topic == image_topic:
        # Print image message timestamp
        print(f"Image Msg - Time: {timestamp:.3f}")

        # Convert ROS compressed image to OpenCV format
        frame = bridge.compressed_imgmsg_to_cv2(msg, "bgr8")

        # Find the closest data message for synchronization
        latest_data = "N/A"
        if data_buffer:
            closest_time, closest_msg = min(data_buffer, key=lambda x: abs(x[0] - timestamp))

            # Print timestamp difference for debugging
            print(f"Closest Wheel Msg - Time: {closest_time:.3f}, Diff: {abs(closest_time - timestamp):.3f}")

            # Extract and round vel_left and vel_right
            latest_data = f"L: {round(closest_msg.vel_left, 3)} R: {round(closest_msg.vel_right, 3)}"

        # Overlay the latest data topic value
        cv2.putText(frame, latest_data, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        if accept_buffer:
            closest_time, closest_msg = min(accept_buffer, key=lambda x: abs(x[0] - timestamp))

            # Print timestamp difference for debugging
            print(f"Closest Wheel Msg - Time: {closest_time:.3f}, Diff: {abs(closest_time - timestamp):.3f}")

            # Extract and round vel_left and vel_right
            latest_data = f"L accept: {round(closest_msg.vel_left, 3)} R accept: {round(closest_msg.vel_right, 3)}"

        # Overlay the latest data topic value
        cv2.putText(frame, latest_data, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Display frame
        cv2.imshow("ROS Video with Data", frame)

        # Slow down playback (2x slower)
        if cv2.waitKey(33) & 0xFF == ord("q"):
            break

# Cleanup
bag.close()
cv2.destroyAllWindows()
