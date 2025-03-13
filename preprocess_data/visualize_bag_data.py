import cv2
import rosbag
from cv_bridge import CvBridge
from collections import deque

# Bag file path
bag_path = "/home/perception/Desktop/duckiebot_project/preprocess_data/record/lab_record/lane_following_light_50/chicinvabot_2025-03-12-18-57-14.bag"

# Define topics
image_topic = "/chicinvabot/camera_node/image/compressed"
data_topic = "/chicinvabot/car_cmd_switch_node/cmd"
accept_topic = "/chicinvabot/car_cmd_switch_node/cmd_executed"

# Open bag file
bag = rosbag.Bag(bag_path)
bridge = CvBridge()

# Buffer for wheel command data with timestamps
data_buffer = deque()
accept_buffer = deque()

# Read messages
for topic, msg, t in bag.read_messages(topics=[image_topic, data_topic, accept_topic]):
    timestamp = t.to_sec()  # Convert ROS time to seconds

    if topic == data_topic:
        # Print velocity and omega message with timestamp
        print(f"Velocity Msg - Time: {timestamp:.3f}, vel: {msg.v:.3f}, omega: {msg.omega:.3f}")
        
        # Store latest message with timestamp
        data_buffer.append((timestamp, msg))
        if len(data_buffer) > 10:  # Limit buffer size for efficiency
            data_buffer.popleft()
    
    if topic == accept_topic:
        # Print executed velocity and omega message with timestamp
        print(f"Executed Velocity Msg - Time: {timestamp:.3f}, vel: {msg.v:.3f}, omega: {msg.omega:.3f}")
        
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
            print(f"Closest Velocity Msg - Time: {closest_time:.3f}, Diff: {abs(closest_time - timestamp):.3f}")
            
            # Extract and round vel and omega
            latest_data = f"Vel: {round(closest_msg.v, 3)} Omega: {round(closest_msg.omega, 3)}"
        
        # Overlay the latest data topic value
        cv2.putText(frame, latest_data, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        if accept_buffer:
            closest_time, closest_msg = min(accept_buffer, key=lambda x: abs(x[0] - timestamp))
            
            # Print timestamp difference for debugging
            print(f"Closest Executed Velocity Msg - Time: {closest_time:.3f}, Diff: {abs(closest_time - timestamp):.3f}")
            
            # Extract and round vel and omega
            latest_data = f"Vel Exec: {round(closest_msg.v, 3)} Omega Exec: {round(closest_msg.omega, 3)}"
        
        # Overlay the latest executed data topic value
        cv2.putText(frame, latest_data, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Display frame
        cv2.imshow("ROS Video with Data", frame)
        
        # Slow down playback (2x slower)
        if cv2.waitKey(33) & 0xFF == ord("q"):
            break

# Cleanup
bag.close()
cv2.destroyAllWindows()
