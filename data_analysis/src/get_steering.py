import numpy as np
import cv2  # Ensure OpenCV is installed
from get_feature import get_weight_matrix, distance_point_to_line, get_trajectory_and_error  # Corrected import path

class SteeringController:
    def __init__(self, steer_max=1.0, omega_max=8.0):
        self.steer_max = steer_max
        self.omega_max = omega_max
        self.v_0 = 2.0
        self.theta_ref = 0.0
        self.e_int = 0
        self.steer_history = []  # To store steering values for smoothing
        self.decay_factor = 0.95  # Decay factor for smoothing
        self.desired_yellow_distance = 10
        self.desired_white_distance = 10
        self.prev_e = 0

    def calculate_steering(self, white_segments, yellow_segments, method):
        white_distance = 0
        yellow_distance = 0
        # Use the specified method for line fitting
        if method == "trajectory_error":
            yellow_weight_mask, white_weight_mask, yellow_distance, white_distance, steer = get_trajectory_and_error(white_segments, yellow_segments)

        else:
            yellow_weight_mask, yellow_slope, yellow_intercept, yellow_initial_mask = get_weight_matrix(yellow_segments, "yellow", method=method)
            white_weight_mask, white_slope, white_intercept, white_initial_mask = get_weight_matrix(white_segments, "white", method=method)

            if method == 'matrix' or method == 'update_matrix':
                total_weight_matrix = yellow_weight_mask + white_weight_mask
                steer = float(np.sum(total_weight_matrix))
                self.log(f"Steer from weight matrix: {steer}")
            else:
                # Adjust this center to make the car stay in the lane
                point_a = (50, 10)
                if yellow_slope is not None and white_slope is not None:
                    yellow_distance = distance_point_to_line(
                        point_a[0], point_a[1], yellow_slope, yellow_intercept
                    )
                    white_distance = distance_point_to_line(
                        point_a[0], point_a[1], white_slope, white_intercept
                    )
                    # Normalize the steering value (negative: turn left, positive: turn right)
                    steer = (yellow_distance - white_distance) / (yellow_distance + white_distance)
                elif yellow_slope is not None:
                    # Only yellow line visible - try to maintain constant distance
                    yellow_distance = distance_point_to_line(
                        point_a[0], point_a[1], yellow_slope, yellow_intercept
                    )
                    white_distance = self.desired_yellow_distance
                    steer = (yellow_distance - self.desired_yellow_distance) / (yellow_distance + white_distance)
                elif white_slope is not None:
                    # Only white line visible - try to maintain constant distance
                    white_distance = distance_point_to_line(
                        point_a[0], point_a[1], white_slope, white_intercept
                    )
                    yellow_distance = self.desired_white_distance
                    steer = (self.desired_white_distance - white_distance) / (yellow_distance + white_distance)
                else:
                    steer = -100
                    # raise ValueError(f"no yellow / white segment is found!")

        steer_scaled = (
            np.sign(steer)
            * self.rescale(min(np.abs(steer), self.steer_max), 0, self.steer_max)
        )

        self.log(f"Steer scaled: {steer_scaled}")
        # return steer_scaled * self.omega_max, yellow_initial_mask, white_initial_mask, yellow_distance, white_distance
        return steer_scaled * self.omega_max, yellow_weight_mask, white_weight_mask, yellow_distance, white_distance
        
    def get_velocity_and_orientation(self, v_0, white_segments, yellow_segments, steering_method, controller_method, delta_t=0.0):
        # Calculate new orientation using steering logic
        theta_hat, yellow_initial_mask, white_initial_mask, yellow_distance, white_distance = self.calculate_steering(white_segments, yellow_segments, steering_method)

        # Apply smoothing based on controller_method
        if controller_method == 'moving_average+decay':
            omega = self.apply_moving_average_decay(theta_hat)
        elif controller_method == 'moving_median+decay':
            omega = self.apply_moving_median_decay(theta_hat)

        elif controller_method == 'PID':
            prev_int = self.e_int
            prev_e = self.prev_e
            # Tracking error
            e = self.theta_ref - theta_hat

            # integral of the error
            e_int = prev_int + e*delta_t

            # anti-windup - preventing the integral error from growing too much
            e_int = max(min(e_int,15),-15)

            # derivative of the error
            e_der = (e - prev_e)/delta_t

            # controller coefficients
            Kp = 0.3  # How fast the system react to change given the current e. Higher the value, higher the reaction
            Kd = 0.2    # How fast the system react to change given the diff e. Higher the value, lower the respond 
            Ki = 0.2    # How fast the system react to change given culmulating e. Higher the value, lower the err

            # PID controller for omega
            omega = Kp*e + Ki*e_int + Kd*e_der
        
            self.e_int = e_int
            self.prev_e = e
        else:
            raise ValueError(f'Control method {controller_method} not implemented!')

        return omega, yellow_initial_mask, white_initial_mask, yellow_distance, white_distance, theta_hat

    def apply_moving_average_decay(self, new_value):
        """Apply moving average with decay to smooth steering values."""
        # Append new value to history
        self.steer_history.append(new_value)
        if len(self.steer_history) > 5:  # Keep the last 5 values
            self.steer_history.pop(0)

        # Calculate moving average
        moving_avg = np.mean(self.steer_history)

        # Apply decay factor
        smoothed_value = self.decay_factor * moving_avg + (1 - self.decay_factor) * new_value
        self.log(f"Moving Average + Decay: {smoothed_value}")
        return smoothed_value

    def apply_moving_median_decay(self, new_value):
        """Apply moving median with decay to smooth steering values."""
        # Append new value to history
        self.steer_history.append(new_value)
        if len(self.steer_history) > 5:  # Keep the last 5 values
            self.steer_history.pop(0)

        # Calculate moving median
        moving_median = np.median(self.steer_history)

        # Apply decay factor
        smoothed_value = self.decay_factor * moving_median + (1 - self.decay_factor) * new_value
        self.log(f"Moving Median + Decay: {smoothed_value}")
        return smoothed_value

    @staticmethod
    def rescale(value, min_val, max_val):
        """Rescale a value from [min_val, max_val] to [0, 1]."""
        return (value - min_val) / (max_val - min_val) if max_val > min_val else 0.0

    @staticmethod
    def log(message):
        """Log messages for debugging purposes."""
        print(message)
