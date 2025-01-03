#!/usr/bin/env python3

import os
import rospy
import numpy as np
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import Twist2DStamped, CarControl, ObstacleImageDetectionList
from predefined_car_routine import PredefinedCarRoutine
from std_msgs.msg import String

class WheelControlNode(DTROS):
    def __init__(self, node_name):
        super(WheelControlNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)

        # Static parameters
        vehicle_name = os.environ['VEHICLE_NAME']
        twist_topic = f"/{vehicle_name}/car_cmd_switch_node/cmd"
        state_topic = f"/{vehicle_name}/state"

        # Initialize velocities
        self._v = 0.0  # Linear velocity (m/s)
        self._omega = 0.0  # Angular velocity (rad/s)
        self._theta = 0.85  # Decay factor for smoothing
        self._window_size = 4  # Window size for smoothing
        self.max_omega = 5.0  # Maximum steering value

        # State variables
        self._state = "state1"
        self._stop_sign_detected = 0
        self._red_line = False
        self._red_light_detected = 0
        self._red_light_missed = 0
        self._duck_detected = False
        self._green_light_detected = False
        self._last_stop_sign_time = rospy.get_time()  # Track when we last saw a stop sign
        self._last_red_light_time = rospy.get_time()

        # Initialize the previous time variable
        self.previous_time = rospy.get_time()  # Initialize with the current time
        self.previous_time_line_seg = rospy.get_time()

        # Smoothed steering values
        self._steering_window = []

        # Pid variable (error here denote steering error)
        self.theta_ref = 0.0  # Desire orientation of the bot relative to the road
        self.e_int = 0        # Culmulating error
        self.prev_e = 0       # Derivative error

        # Construct publisher
        self._wheel_publisher = rospy.Publisher(twist_topic, Twist2DStamped, queue_size=1)
        
        self._state_publisher = rospy.Publisher(state_topic, String, queue_size=1)

        # Subscribe to car control and object detection topics
        self._line_seg_detector_sub = rospy.Subscriber(
            f"/{vehicle_name}/line_seg_detector", CarControl, self.line_seg_detector_cb
        )
        self._object_detector_sub = rospy.Subscriber(
            f"/{vehicle_name}/detection_list", ObstacleImageDetectionList, self.object_detector_cb
        )

        # Intersection routine helper
        self._predefined_routine = PredefinedCarRoutine(self._wheel_publisher)

        self.log("WheelControlNode initialized.")

    def reset_variable(self):
        # Pid variable (error here denote steering error)
        self.theta_ref = 0.0  # Desire orientation of the bot relative to the road
        self.e_int = 0        # Culmulating error
        self.prev_e = 0       # Derivative error

    def line_seg_detector_cb(self, msg):
        # if self._state == 'state1' or self._state == 'state32':
        #     self.log(f"Callback triggered with message: speed={msg.speed:.3f}, steering={msg.steering:.3f}")

        # Get the current time
        current_time = rospy.get_time()

        # Calculate delta time
        delta_t = current_time - self.previous_time_line_seg

        # Log or process delta time
        # self.log(f"Delta time between callbacks: {delta_t:.6f} seconds")

        # Update the previous time
        self.previous_time_line_seg = current_time
        
        # Different control when encouter red line
        # Control logic for state31 and state32
        if self._state == 'state31':
            self._v = 0
            self._omega = 0

        elif self._state == 'state32':
            # # Add the current steering to the window
            # self._steering_window.append(msg.steering)

            # # Ensure the window does not exceed size self._window_size
            # if len(self._steering_window) > self._window_size:
            #     self._steering_window.pop(0)

            # if len(self._steering_window) == self._window_size:
            #     # Calculate the average of the previous steering values
            #     average_steering_prev = np.mean(self._steering_window[:-1])

            #     # Apply the weighted moving average formula
            #     self._omega = (
            #         self._theta * average_steering_prev +
            #         (1 - self._theta) * self._steering_window[-1]
            #     )
            # else:
            #     # Use the current steering if less than 3 values are available
            # Adjust the speed
            self._v = msg.speed * 0.6
            self._omega = msg.steering * 6.0

        elif self._state == 'state62':
            self._v = msg.speed * 0.6
            self._omega = msg.steering
        # Normal pid control when encouter state 1
        elif self._state == 'state1':
            if msg.speed < 0.01:
                self._red_line = True
                self.log("Speed is below threshold. Setting _red_line to True.")
                return
            
            # Update linear velocity
            if np.abs(msg.steering) > 1.0:
                self._v = msg.speed * 0.6  # Reduce speed at corner
            else:
                self._v = msg.speed * 0.6  # Scale the speed appropriately
        
            # if self.prev_e == 0:
            #     theta_hat = msg.steering  # The msg have the needed steering value. However, pid want the error. Theta hat is estimate of orientation 
            #     theta_hat = np.clip(msg.steering, -self.max_omega, self.max_omega)
            #     self.prev_e = self.theta_ref - theta_hat
            #     return
            
            # # Implement of pid control
            # theta_hat = msg.steering  # The msg have the needed steering value. However, pid want the error. Theta hat is estimate of orientation 
            # theta_hat = np.clip(msg.steering, -self.max_omega, self.max_omega)
            
            # prev_int = self.e_int
            # prev_e = self.prev_e
            # # Tracking error
            # e = self.theta_ref - theta_hat

            # # integral of the error
            # e_int = prev_int + e*delta_t

            # # anti-windup - preventing the integral error from growing too much
            # e_int = max(min(e_int,2),-2)

            # # derivative of the error
            # e_der = (e - prev_e)/delta_t

            # # controller coefficients
            # Kp = 0.625      # How fast the system react to change given the current e. Higher the value, higher the reaction
            # Ki = 0    # How fast the system react to change given culmulating e. Higher the value, lower the err
            # Kd = 0.10    # How fast the system react to change given the diff e. Higher the value, lower the respond 
            

            # # PID controller for omega
            # self._omega = Kp*e + Ki*e_int + Kd*e_der
        
            # self.e_int = e_int
            # self.prev_e = e

            # self._omega = -self._omega
            # self._omega = np.clip(self._omega, -self.max_omega, self.max_omega)
            self._omega = msg.steering

    def object_detector_cb(self, msg):
        """
        Callback for processing detected objects.
        Resets detection flags at each callback and updates them based on current detections.
        """
        # Reset all detection flags
        self._duck_detected = False

        # Threshold for the size of the bounding box
        STOP_SIGN_SIZE_THRESHOLD = 5000  # Example value, adjust as needed

        # Process current detections
        for detection in msg.list:
            obj_type = detection.type.type
            bbox = detection.bounding_box
            bbox_size = bbox.w * bbox.h  # Calculate the size of the bounding box

            if obj_type == 4:  # Stop sign
                # if bbox_size > STOP_SIGN_SIZE_THRESHOLD:  # Check if size exceeds the threshold
                current_time = rospy.get_time()
                # Only increment if within a reasonable time window (e.g., 2 seconds)
                if current_time - self._last_stop_sign_time < 2.0:
                    self._stop_sign_detected += 1
                else:
                    # If too much time has passed, reset the counter
                    self._stop_sign_detected = 0
                self._last_stop_sign_time = current_time
                self._stop_sign_detected += 1
            elif obj_type in [0, 1]:  # Duck or Duckiebot
                self._duck_detected = True
            elif obj_type == 30 and self._state == 'state4':  # Red light detection in a specific state
                self._red_light_detected += 1
                # elif obj_type == 31:
                #     self._red_light_detected = 0

    def intersection_routine(self):
        """
        Handles intersection logic using the IntersectionRoutine class.
        """
        self._predefined_routine.execute_intersection_routine(
            direction="straight"
        )

    def switch_lane_routine(self):
        """
        Handles lane switching logic.
        """
        self.log("Executing lane switch routine.")

    def publish_twist(self):
        message = Twist2DStamped()
        message.v = self._v
        message.omega = self._omega
        self._wheel_publisher.publish(message)
        # Log the current velocities
        # log_message = f"Velocities: linear={self._v:.3f}, angular={self._omega:.3f}"
        # self.log(log_message)

    def publish_FSM(self):
        message = String()
        message.data = self._state
        self._state_publisher.publish(message)

    def run(self):
        rate = rospy.Rate(10)
        last_state_print_time = rospy.get_time()  # Track the last time the state was printed
        red_line_check_start = None # Track when we start checking red line status
        traffic_light_check_start = None  # Track when we start checking traffic light status
        state2_start_time = None  # Track when we enter state2
        state2_phase = "running"  # Track which phase of state2 we're in
        intersection_routine_start = None # Track intersection routine stage 2

        while not rospy.is_shutdown():
            current_time = rospy.get_time()
            
            # Print and publish state every 1 second
            if current_time - last_state_print_time >= 1.0:
                self.log(f"Current state: {self._state}")
                last_state_print_time = current_time
                self.publish_FSM()

            if self._state == "state1":
                
                self.publish_twist()
                if self._stop_sign_detected > 3:
                    self._state = "state2"
                    state2_start_time = current_time
                    state2_phase = "running"
                    self.log(f"Stop sign detected {self._stop_sign_detected} times. Entering state2 running phase")
                elif self._red_line:
                    self._state = "state31"
                    red_line_check_start = current_time
                    self.log("Red line detected. Entering state31 for red line routine")
                    

            elif self._state == "state2":
                
                if state2_phase == "running":
                    self.log(f'enter {self._state}')
                    self.publish_twist()
                    self.log(f"State2 running phase: Time elapsed = {current_time - state2_start_time:.2f}s")
                    if current_time - state2_start_time >= 3.0:
                        state2_phase = "stopping"
                        state2_start_time = current_time
                        self.log("State2: Transitioning to stopping phase")
                
                elif state2_phase == "stopping":
                    self._predefined_routine.execute_intersection_routine(direction="stop")
                    self.log(f"State2 stopping phase: Time elapsed = {current_time - state2_start_time:.2f}s")
                    if current_time - state2_start_time >= 2.0:
                        self._state = "state1"
                        self._stop_sign_detected = 0
                        self.log("State2: Stop completed. Returning to state1")

            # When red line is detected, go to state31. This is just stop
            elif self._state == 'state31':
                
                self.log('publish stop cmd for 1s')
                self._v = 0
                self._omega = 0
                self.publish_twist()
                if current_time - red_line_check_start >= 1.0:
                    self.log('finish publishing!')
                    red_line_check_start = current_time
                    self._state = 'state32'
                
            # continue moving for a moment before stop
            elif self._state == "state32":
                
                self.publish_twist()
                if current_time - red_line_check_start >= 4.0:
                    self._state = "state4"
                    traffic_light_check_start = current_time
                    self.log("Red line routine finished. Entering state4 for traffic light check")
                    
            elif self._state == "state4":
                
                self._predefined_routine.execute_intersection_routine(direction="stop")
                self.log(f"State4: Checking traffic light. Red light detections = {self._red_light_detected}, Time elapsed = {current_time - traffic_light_check_start:.2f}s")
                
                if traffic_light_check_start and (current_time - traffic_light_check_start >= 1.5):
                    if self._red_light_detected < 3:
                        self._state = "state61"
                        traffic_light_check_start = None
                        self.log(f"State4: No red light detected: {self._red_light_detected}. Transitioning to state61")
                    else:
                        traffic_light_check_start = current_time
                        self.log(f"State4: Red light detected: {self._red_light_detected}. Resetting check timer")
                        self._red_light_detected = 0

            elif self._state == "state61":
                self.log("State61: Executing intersection routine")
                self.intersection_routine()
                rospy.sleep(5)
                self._state = "state1"
                self.log("State61: Intersection routine first phase completed. Returning to state1")
                intersection_routine_start = current_time
                self._v = 0
                self._omega = 0
                self.publish_twist()
                
            elif self._state == "state62":
                self.log("State62: Executing intersection routine second phase")
                self.publish_twist()
                
                if current_time - intersection_routine_start > 7.0:
                    self.log("State62: Intersection routine completed. Returning to state1")
                    self._state = "state1"
                    self.reset_variable()
                    self._red_line = False

            rate.sleep()

    def on_shutdown(self):
        stop_message = Twist2DStamped(v=0.0, omega=0.0)
        self._wheel_publisher.publish(stop_message)

if __name__ == '__main__':
    node = WheelControlNode(node_name='wheel_control_node')
    node.run()
    rospy.spin()
