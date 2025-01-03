#!/usr/bin/env python3

import os
import rospy
import numpy as np
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import WheelsCmdStamped, CarControl, ObstacleImageDetectionList
from predefined_car_routine import PredefinedCarRoutine  # Import your IntersectionRoutine class


class WheelControlNode(DTROS):

    def __init__(self, node_name):
        super(WheelControlNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)

        # static parameters
        vehicle_name = os.environ['VEHICLE_NAME']
        wheels_topic = f"/{vehicle_name}/wheels_driver_node/wheels_cmd"

        # Base speed for both wheels
        self._base_speed_left = 0.2
        self._base_speed_right = 0.24

        # Initialize smoothed velocities
        self._smoothed_vel_left = self._base_speed_left
        self._smoothed_vel_right = self._base_speed_right

        # Set decay factor
        self._decay_factor = 0.9

        # State variables
        self._state = "state1"  # Initial state is normal execution
        self._stop_sign_counter = 0
        self._red_line = False
        self._red_light_detected = 0
        self._red_light_missed = 0

        # Object detection flags
        self._duck_detected = False

        # Construct publisher
        self._publisher = rospy.Publisher(wheels_topic, WheelsCmdStamped, queue_size=1)

        # Subscribe to car control and object detection topics
        self._line_seg_detector_sub = rospy.Subscriber(
            f"/{vehicle_name}/line_seg_detector", CarControl, self.line_seg_detector_cb
        )
        self._object_detector_sub = rospy.Subscriber(
            f"/{vehicle_name}/detection_list", ObstacleImageDetectionList, self.object_detector_cb
        )

        # Intersection routine helper
        self._predefined_routine = PredefinedCarRoutine(self._publisher)

        self.log("WheelControlNode initialized.")

    def line_seg_detector_cb(self, msg):
        self.log(f"Callback triggered with message: speed={msg.speed:.3f}, steering={msg.steering:.3f}")

        if self._state == "state1":
            speed = msg.speed
            steering = msg.steering

            if speed < 0.01:
                self._red_line = True
                self.log("Speed is below threshold. Setting _red_line to True.")
                return

            if msg.speed <= 0.01:
                self._base_speed = 0

            max_steering_adjustment = 1.7
            adjustment = max_steering_adjustment * steering

            self._vel_left = self._base_speed_left * np.max([1, 1 + adjustment])
            self._vel_right = self._base_speed_right * np.max([1, 1 - adjustment])


            self._smoothed_vel_left = (
                self._decay_factor * self._vel_left + (1 - self._decay_factor) * self._smoothed_vel_left
            )
            self._smoothed_vel_right = (
                self._decay_factor * self._vel_right + (1 - self._decay_factor) * self._smoothed_vel_right
            )
            # Assigning the formatted log message to a variable
            log_message = (f"Smoothed velocities: smoothed_vel_left_normalized={(self._smoothed_vel_left/self._base_speed_left):.3f}, "
                        f"smoothed_vel_right={(self._smoothed_vel_right/self._base_speed_right):.3f}, "
                        f"difference: {(self._smoothed_vel_left / self._smoothed_vel_right):.3f}")

            # Logging the message
            self.log(log_message)

    def object_detector_cb(self, msg):
        stop_sign_detected = False
        self._duck_detected = False
        red_light_detected = False
        green_light_detected = False

        for detection in msg.list:
            obj_type = detection.type.type

            if obj_type == 4:
                stop_sign_detected = True
                self._stop_sign_counter += 1

            elif obj_type == 0 or obj_type == 1:
                self._duck_detected = True

            elif obj_type == 30:
                red_light_detected = True

            elif obj_type == 31:
                green_light_detected = True

        # Transition logic for state machine
        # if self._state == "state1":
        #     if stop_sign_detected:
        #         self._state = "state2"
        #         rospy.sleep(3)  # Simulate stop for 3 seconds
        #         self._state = "state1"

        #     # elif self._duck_detected:
        #     #     self._state = "state3"
        #     #     self.switch_lane_routine()
        #     #     rospy.sleep(3)  # Continue normal execution for 3 seconds
        #     #     self.switch_lane_routine()
        #     #     self._state = "state1"

        #     elif self._red_line:
        #         self._state = "state4"
        #         rospy.sleep(1)  # Continue for 1 second before stopping

        # elif self._state == "state4":
        #     if red_light_detected:
        #         self._red_light_detected += 1
        #         if self._red_light_detected >= 2 and self._red_light_missed <= 20:
        #             self._state = "state5"
        #     else:
        #         self._red_light_missed += 1
        #         if self._red_light_detected < 2 and self._red_light_missed > 20:
        #             self._state = "state6"

        # elif self._state == "state6":
            self.intersection_routine()
            self._state = "state1"

    def intersection_routine(self):
        """
        Handles intersection logic using the IntersectionRoutine class.
        """
        self._predefined_routine.execute_intersection_routine(
            direction="straight", velocity=0.1, duration=5
        )

    def switch_lane_routine(self):
        """
        Handles lane switching logic.
        """
        self.log("Executing lane switch routine.")
        # self._predefined_routine.object_avoid_routine(velocity=0.1, duration=3)

    def publish_wheel_speeds(self):
        message = WheelsCmdStamped()
        message.vel_left = self._smoothed_vel_left
        message.vel_right = self._smoothed_vel_right
        self._publisher.publish(message)

    def run(self):
        while not rospy.is_shutdown():
            if self._state == "state1":
                self.publish_wheel_speeds()

    def on_shutdown(self):
        stop_message = WheelsCmdStamped(vel_left=0, vel_right=0)
        self._publisher.publish(stop_message)


if __name__ == '__main__':
    node = WheelControlNode(node_name='wheel_control_node')
    node.run()
    rospy.spin()


            # # Moving average implementation
            # # Update steering window
            # if len(self._steering_window) >= self._window_size:
            #     self._steering_window.pop(0)  # Remove the oldest value
            # self._steering_window.append(msg.steering)

            # # Compute smoothed steering with moving average and decay
            # if len(self._steering_window) == self._window_size:
            #     window_avg = np.mean(self._steering_window)
            #     self._omega = self._theta * self._omega + (1 - self._theta) * window_avg

            # self._omega = np.clip(self._omega, -self.max_omega, self.max_omega)