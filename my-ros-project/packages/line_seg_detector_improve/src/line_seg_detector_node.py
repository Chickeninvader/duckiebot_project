#!/usr/bin/env python3

import os
import rospy
import numpy as np
from duckietown_msgs.msg import SegmentList, Segment, CarControl
from duckietown.dtros import DTROS, NodeType, TopicType
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
from std_msgs.msg import String
import cv2

from get_features import get_trajectory_and_error, get_weight_matrix, distance_point_to_line, rescale_and_shift_point


class LineSegmentCheckerNode(DTROS):
    def __init__(self, node_name):
        super(LineSegmentCheckerNode, self).__init__(node_name=node_name, node_type=NodeType.PERCEPTION)

        self._vehicle_name = os.environ.get('VEHICLE_NAME', 'default_vehicle')

        # Initialize CV bridge for image conversion
        self.bridge = CvBridge()

        # Initialize the previous time variable
        self.previous_time = rospy.get_time()  # Initialize with the current time

        self.steer_max = 1.0
        self.omega_max = 8.0

        self.desired_yellow_distance = 10
        self.desired_white_distance = 10

        self.alignment_steering_gain = 3.0
        self.alignment_duration = 3.0

        # State variable
        self.red_line_detected = False
        self.red_line_detection_time = 0.0
        self.intersection_routine = False
        self._state = None

        self.red_line_distance = 50

        # Subscriber to the SegmentList topic
        self.sub_segments = rospy.Subscriber(
            f"/{self._vehicle_name}/ground_projection_node/lineseglist_out", SegmentList, self.callback_segments, queue_size=1
        )

        # # Subscriber to the State machine topic
        self.sub_segments = rospy.Subscriber(
            f"/{self._vehicle_name}/state", String, self.FSM_cb, queue_size=1
        )

        # Publisher for car control
        self.pub_car_control = rospy.Publisher(
            f"/{self._vehicle_name}/line_seg_detector", CarControl, queue_size=1
        )

        # Publishers for yellow and white segments
        self.pub_debug_image = rospy.Publisher(
            f"/{self._vehicle_name}/line_seg_detector_white_yellow/compressed", CompressedImage, queue_size=1, dt_topic_type=TopicType.DEBUG
        )

        # Publishers for red segments
        self.pub_debug_image_red = rospy.Publisher(
            f"/{self._vehicle_name}/line_seg_detector_red/compressed", CompressedImage, queue_size=1, dt_topic_type=TopicType.DEBUG
        )

        self.log("LineSegmentCheckerNode initialized.")

    def FSM_cb(self, msg: String):
        self.log(f'line seg detector node recieve fsm msg: {msg.data}')
        self._state = msg.data

    def callback_segments(self, msg: SegmentList):

        # Get the current time
        current_time = rospy.get_time()

        # Calculate delta time
        delta_t = current_time - self.previous_time

        # Log or process delta time
        self.log(f"Delta time between callbacks of LineSegmentCheckerNode: {delta_t:.6f} seconds")

        # Update the previous time
        self.previous_time = current_time

        white_segments = []
        yellow_segments = []
        red_segments = []

        for segment in msg.segments:
            if segment.color == Segment.WHITE:
                white_segments.append(segment)
            elif segment.color == Segment.YELLOW:
                yellow_segments.append(segment)
            elif segment.color == Segment.RED:
                red_segments.append(segment)

        # Check for red line detection
        if (self.detect_red_line(red_segments, -20, 20, 10, 50) or self.red_line_detected) and not self.intersection_routine:

            # If it's the first detection of a red line, initialize detection time
            if not self.red_line_detected:
                self.red_line_detected = True
                self.red_line_detection_time = rospy.get_time()
                self.publish_car_control(0.0, stop=True) # Signal transistion state for wheel control node
                self.publish_car_control(0.0, stop=True) # Signal transistion state for wheel control node
                self.publish_car_control(0.0, stop=True) # Signal transistion state for wheel control node
                self.log(f'line seg node sleep for 2s, set red_line_deteted = True')
                rospy.sleep(2)
                self.log(f'finish sleep')
                
                self.log("Red line detected. Starting alignment.")

            # Get the elapsed time since the first red line detection
            elapsed_time = rospy.get_time() - self.red_line_detection_time

            # Calculate alignment steering
            alignment_steering, distance_to_red = self.calculate_red_line_alignment(red_segments, -20, 20, 10, self.red_line_distance)

            # we keep track the red segment that is close to the bot
            self.red_line_distance = min(distance_to_red + 2.5, 40) if distance_to_red is not None else 40
            self.log(f'red line distance reduce: {self.red_line_distance}')

            # Stop if the distance is below the threshold or if alignment_duration seconds have elapsed
            if self.red_line_distance <= 20:
                self.log(f"Red line alignment complete. distance {self.red_line_distance}. Publishing stop command.")
                
                if np.abs(alignment_steering) > 0.05:
                    self.log(f'try to fix orientation if needed: curr orientation: {alignment_steering}')
                    
                    self.publish_car_control(alignment_steering, stop=False, velocity=0.25)
                    self.publish_car_control(alignment_steering, stop=False, velocity=0.25)
                    self.publish_car_control(alignment_steering, stop=False, velocity=0.25)

                self.log(f'publish stop state 32 and ready for transistion to next state')
                rospy.sleep(0.2)
                self.publish_car_control(0.0, stop=True)  # Signal transistion state for wheel control node
                self.publish_car_control(0.0, stop=True)  # Signal transistion state for wheel control node
                self.publish_car_control(0.0, stop=True)  # Signal transistion state for wheel control node
                self.red_line_detected = False  # Reset detection flag
                self.red_line_distance = 50 # reset red line distance
                self.log(f'line seg node sleep for 5s, set intersection routine = True')
                self.intersection_routine = True
                rospy.sleep(5)
                self.log(f'finish sleep')
                return

            # Continue making steering adjustments
            if alignment_steering is not None:
                self.publish_car_control(alignment_steering, stop=False, velocity=0.25)
            else:
                self.publish_car_control(0.0, stop=False)

        # TODO: Define logic to handle at intersection
        elif self.intersection_routine:
            # Update lane following
            steering_angle = self.calculate_steering(yellow_segments, white_segments, method="distance_error", start_yellow=[-15, 20], start_white=[10, 20])
            
            if steering_angle is not None and self._state =='state62':
                self.publish_car_control(steering_angle)

            if self._state == 'state1':
                self.intersection_routine = False

        elif not self.red_line_detected:
            # Normal lane following
            steering_angle = self.calculate_steering(yellow_segments, white_segments, method="distance_error")
            if steering_angle is not None:
                self.publish_car_control(steering_angle)

    def rescale(self, value, min_val, max_val):
        """Rescale a value from [min_val, max_val] to [0, 1]."""
        return (value - min_val) / (max_val - min_val) if max_val > min_val else 0.0

    def calculate_steering(self, yellow_segments, white_segments, method, start_yellow=[-10, 10], start_white=[10, 10]):
        white_distance = 0
        yellow_distance = 0
        # Use the specified method for line fitting

        if len(white_segments) == 0:
            self.log('No white segments')

        if len(yellow_segments) == 0:
            self.log('No yellow segments')

        
        yellow_weight_mask, white_weight_mask, yellow_distance, white_distance, steer = get_trajectory_and_error(
            yellow_segments, white_segments, start_yellow=start_yellow, start_white=start_white
        )

        steer_scaled = (
            np.sign(steer)
            * self.rescale(min(np.abs(steer), self.steer_max), 0, self.steer_max)
        )

        # Create debug visualization
        self.create_debug_visualization(
            yellow_weight_mask,
            white_weight_mask,
        )

        # self.log(f'distance white: {white_distance}, distance yellow: {yellow_distance}')
        # self.log(f"Steer scaled: {steer_scaled}")
        # return steer_scaled * self.omega_max, yellow_initial_mask, white_initial_mask, yellow_distance, white_distance
        return steer_scaled * self.omega_max
        
    def create_debug_visualization(
        self,
        yellow_mask,
        white_mask,
    ):
        """Create debug visualization showing detected lanes and fitted lines."""
        # Create colored masks
        yellow_colored = np.zeros_like(yellow_mask[..., None]).repeat(3, axis=2)
        white_colored = np.zeros_like(white_mask[..., None]).repeat(3, axis=2)
        
        # Set colors for masks
        yellow_colored[yellow_mask > 0] = [0, 255, 255]  # BGR format
        white_colored[white_mask > 0] = [255, 255, 255]
        
        # Combine masks
        combined = cv2.addWeighted(yellow_colored, 0.5, white_colored, 0.5, 0)

        # Publish debug image
        debug_msg = self.bridge.cv2_to_compressed_imgmsg(combined)
        self.pub_debug_image.publish(debug_msg)
  
    def publish_car_control(self, steering_angle, stop=False, velocity=0.3):
        car_control_msg = CarControl()
        car_control_msg.header.stamp = rospy.Time.now()

        if stop:
            car_control_msg.speed = 0.0
            car_control_msg.steering = 0.0
            self.log("Publishing stop command.")
        else:
            car_control_msg.speed = velocity
            car_control_msg.steering = steering_angle
            self.log(f"Publishing CarControl: speed={car_control_msg.speed}, steering={car_control_msg.steering} rad")

        car_control_msg.need_steering = True
        self.pub_car_control.publish(car_control_msg)

    def detect_red_line(self, red_segments, x_min, x_max, y_min, y_max):
        count = 0
        for segment in red_segments:
            pt1 = (segment.points[0].y * -100, segment.points[0].x * 100)
            pt2 = (segment.points[1].y * -100, segment.points[1].x * 100)
            if x_min <= pt1[0] <= x_max and y_min <= pt1[1] <= y_max \
                    and x_min <= pt2[0] <= x_max and y_min <= pt2[1] <= y_max:
                count += 1
        if count > 8:
            return True
        return False

    def calculate_red_line_alignment(self, red_segments, x_min, x_max, y_min, y_max):
        """
        Calculate steering angle to align perpendicular to red line
        Returns: steering_angle in radians or None if no valid angle can be calculated
        """
        if not red_segments:
            return None, None
        
        # Collect all points from red segments
        red_points = []
        for segment in red_segments:
            pt1 = (segment.points[0].y * -100, segment.points[0].x * 100)
            pt2 = (segment.points[1].y * -100, segment.points[1].x * 100)
            if x_min <= pt1[0] <= x_max and y_min <= pt1[1] <= y_max \
                    and x_min <= pt2[0] <= x_max and y_min <= pt2[1] <= y_max:
                red_points.extend([rescale_and_shift_point(pt1), rescale_and_shift_point(pt2)])
        
        if len(red_points) < 4:  # Need at least 4 points to form a rectangle
            return None, None
        
        # Convert to numpy array safely
        points_array = np.array(red_points, dtype=np.float32)
        if points_array.shape[1] != 2:  # Ensure we have 2D points
            self.log("Invalid points format")
            return None, None

        # Calculate median y-coordinate
        median_y = np.median(points_array[:, 1])
        threshold = 3  # 3cm = 30mm in pixel space

        # Filter points within threshold of median
        mask = np.abs(points_array[:, 1] - median_y) <= threshold
        filtered_points = points_array[mask].tolist()

        if len(filtered_points) < 4:
            self.log(f"Too few points ({len(filtered_points)}) after outlier filtering")
            return None, None

        # Find extreme coordinates
        points_array = np.array(filtered_points)
        max_x = np.max(points_array[:, 0])
        min_x = np.min(points_array[:, 0])
        max_y = np.max(points_array[:, 1])
        min_y = np.min(points_array[:, 1])
    
        # Initialize corner points and distances
        corners = {
            'top_right': {'point': None, 'min_dist': float('inf')},
            'top_left': {'point': None, 'min_dist': float('inf')},
            'bottom_right': {'point': None, 'min_dist': float('inf')},
            'bottom_left': {'point': None, 'min_dist': float('inf')}
        }

        # Find corner points
        for point in filtered_points:
            x, y = point
            
            # Calculate distances to corners
            distances = {
                'top_right': ((x - max_x) ** 2 + (y - max_y) ** 2),
                'top_left': ((x - min_x) ** 2 + (y - max_y) ** 2),
                'bottom_right': ((x - max_x) ** 2 + (y - min_y) ** 2),
                'bottom_left': ((x - min_x) ** 2 + (y - min_y) ** 2)
            }
            
            # Update corner points if closer
            for corner, dist in distances.items():
                if dist < corners[corner]['min_dist']:
                    corners[corner]['point'] = point
                    corners[corner]['min_dist'] = dist

        # Extract corner points
        top_right_point = corners['top_right']['point']
        top_left_point = corners['top_left']['point']
        bottom_right_point = corners['bottom_right']['point']
        bottom_left_point = corners['bottom_left']['point']

        # Check if we found all corners
        if not all([top_right_point, top_left_point, bottom_right_point, bottom_left_point]):
            self.log("Failed to find all corner points")
            return None, None
        
        # Calculate dx and dy for top and bottom edges
        dx_top = top_right_point[0] - top_left_point[0]
        dx_bottom = bottom_right_point[0] - bottom_left_point[0]
        dy_top = top_right_point[1] - top_left_point[1]
        dy_bottom = bottom_right_point[1] - bottom_left_point[1]
        
        # Average dx and dy
        dx_avg = (dx_top + dx_bottom) / 2
        dy_avg = (dy_top + dy_bottom) / 2
        
        # Calculate angle using average dx and dy
        line_angle = np.arctan2(dy_avg, dx_avg)
        target_angle = line_angle  # + np.pi/2
        steering_angle = self.alignment_steering_gain * target_angle

        # Calculate mean distance safely using numpy
        mean_distance = np.mean(points_array[:, 1])

        self.log(f"steering_angle red {steering_angle}")
        # self.log(f"distance to red: {mean_distance}")

        # # Create debug visualization
        # red_mask = np.zeros((100, 100), dtype=np.uint8)

        # # Draw points
        # for point in filtered_points:
        #     pt = tuple(map(int, point))
        #     cv2.circle(red_mask, pt, 1, 255, -1)

        # # Draw polygon of corners safely
        # corners_array = np.array([
        #     top_left_point,
        #     top_right_point,
        #     bottom_right_point,
        #     bottom_left_point
        # ], dtype=np.int32)

        # # cv2.polylines(red_mask, [corners_array], True, 255, 1)
        # red_mask = cv2.flip(red_mask, 1)

        # # Create colored version
        # red_colored = cv2.cvtColor(red_mask, cv2.COLOR_GRAY2BGR)
        # red_colored[red_mask > 0] = [0, 0, 255]  # BGR format for red

        # # Publish debug image
        # try:
        #     debug_msg = self.bridge.cv2_to_compressed_imgmsg(red_colored)
        #     self.pub_debug_image_red.publish(debug_msg)
        # except Exception as e:
        #     self.log(f"Error publishing debug image: {str(e)}")

        # Return negative of angle
        return np.clip(steering_angle, -self.omega_max, self.omega_max), mean_distance

if __name__ == "__main__":
    node = LineSegmentCheckerNode('line_seg_check')
    rospy.spin()
