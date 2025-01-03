import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import (LinearRegression, RANSACRegressor, 
                                HuberRegressor, TheilSenRegressor, 
                                Ridge, Lasso, ElasticNet, 
                                BayesianRidge, ARDRegression, 
                                SGDRegressor, PassiveAggressiveRegressor)
from sklearn.preprocessing import PolynomialFeatures
import cv2

# Global variable

scale=1
center_x=50
epsilon = 3
# Optional: Add prior knowledge
prior_mean = 6.0  # Expected slope
prior_std = 0.7   # Uncertainty in slope
lower_bound_y = 0
upper_bound_y = 40

def rescale( a, L, U):
        if np.allclose(L, U):
            return 0.0
        return (a - L) / (U - L)

def rescale_and_shift_point(point):
    x, y = point
    new_x = int(x * scale + center_x)
    new_y = int(y * scale)
    return new_x, new_y

def fit_line_from_points(points, method="linear"):
    """
    Fit a line from the given points using the specified method.
    
    Parameters:
        points (list of tuples): List of (x, y) points.
        method (str): Method for line fitting. Options are "linear", "ransac", "polynomial", etc.
    
    Returns:
        tuple: Slope and intercept of the fitted line.
    """
    if len(points) < 2:
        return None, None  # Not enough points to fit a line

    x_vals, y_vals = zip(*points)
    x_vals = np.array(x_vals)
    y_vals = np.array(y_vals)

    if method == "linear":
        coef = np.polyfit(x_vals, y_vals, 1)  # Linear regression y = mx + c
        return coef[0], coef[1]  # Slope (m), Intercept (c)
    elif method == "ransac":
        model = RANSACRegressor(random_state=42)
        model.fit(x_vals.reshape(-1, 1), y_vals)
        slope = model.estimator_.coef_[0]
        intercept = model.estimator_.intercept_
        return slope, intercept
    else:
        raise ValueError(f"Unsupported fitting method: {method}")

def distance_point_to_line( x, y, slope, intercept):
    if slope is None or intercept is None:
        return 0.0  # Return a large distance if the line cannot be fitted
    return abs(slope * x - y + intercept) / (np.sqrt(slope**2 + 1))

def find_next_point(current_point, transformed_points, direction=[0,1]):
    """
    Find the next point from transformed_points that satisfies distance and direction conditions.
    
    Parameters:
    current_point (tuple): Current point coordinates (x, y)
    transformed_points (list): List of transformed points
    direction (numpy.ndarray): Direction vector to filter points
    
    Returns:
    tuple: Next point with maximum valid distance, or None if no valid point found
    """
    current_point = np.array(current_point)
    max_distance = 0
    next_point = None
    
    for point in transformed_points:
        point = np.array(point)
        # Vector from current point to candidate point
        vector = point - current_point
        distance = np.linalg.norm(vector)
        
        # Check if point is within 5cm
        if distance >= 5 * scale:
            continue

        # If direction is provided, check if point is in forward direction
        if np.dot(direction, vector) <= 0:
            continue
        
        # Update next point if this point has larger distance
        if distance > max_distance:
            max_distance = distance
            next_point = point
    
    return tuple(next_point) if next_point is not None else None

def generate_curve_points_and_get_mask(start_point, transformed_points, scale=2, num_iter=12):
    """
    Generate curve points and create a mask, limited to 6 iterations.
    
    Parameters:
    start_point (tuple): Starting point coordinates (x, y)
    transformed_points (list): List of transformed points
    scale (int): Scale factor for the mask
    
    Returns:
    numpy.ndarray: Generated mask
    """
    mask = np.zeros((100 * scale, 100 * scale), dtype=np.float32)
    points = [start_point]
    current_point = np.array(start_point)
    direction = np.array([0, 1])
    
    # Maximum 6 iterations
    for _ in range(num_iter):
        next_point = find_next_point(current_point, transformed_points, direction)
        if next_point is None or next_point[1] < current_point[1] - epsilon:
            break
        
        points.append(next_point)
        # Update direction for next iteration
        direction = next_point - current_point
        direction = direction / np.linalg.norm(direction)
        current_point = np.array(next_point)

    # Draw curve on mask
    for i in range(len(points) - 1):
        pt1 = points[i]
        pt2 = points[i + 1]
        mask = cv2.line(mask, pt1, pt2, 1, thickness=2)
    
    return cv2.flip(mask, 1), points

def get_weight_matrix(segments, color, method="linear"):
    """
    Compute the weight matrix and fit a line using the specified method.
    
    Parameters:
        segments (list): List of line segments.
        color (str): Color to filter segments ("yellow" or "white").
        method (str): Method for line fitting. Options are "linear", "ransac", "polynomial", etc.
    
    Returns:
        tuple: Weight mask, slope, and intercept of the fitted line.
    """
    if color == "yellow":
        start_point = (-12, 20)
        start_point = rescale_and_shift_point(start_point)
        left_bound_x, right_bound_x = -35, 35
        left_weight, right_weight = -0.01, -0.5
    elif color == "white":
        start_point = (12, 20)
        start_point = rescale_and_shift_point(start_point)
        left_bound_x, right_bound_x = -35, 35
        left_weight, right_weight = 0.01, 0.5
    else:
        raise ValueError(f"Unsupported color: {color}")

    initial_mask = np.zeros((100 * scale, 100 * scale), dtype=np.float32)
    weight_mask = np.zeros_like(initial_mask)
    transformed_points = []

    for segment in segments:
        pt1 = (segment.points[0].y * -100, segment.points[0].x * 100)
        pt2 = (segment.points[1].y * -100, segment.points[1].x * 100)
        if (
            left_bound_x <= pt1[0] <= right_bound_x
            and lower_bound_y <= pt1[1] <= upper_bound_y
            and left_bound_x <= pt2[0] <= right_bound_x
            and lower_bound_y <= pt2[1] <= upper_bound_y
        ):
            new_pt1 = rescale_and_shift_point(pt1)
            new_pt2 = rescale_and_shift_point(pt2)
            transformed_points.extend([new_pt1, new_pt2])
    
    if method == 'update_matrix':
        initial_mask, _ = generate_curve_points_and_get_mask(start_point, transformed_points)

        weight_mask[
            lower_bound_y * scale : upper_bound_y * scale,
            (left_bound_x * scale + center_x) : (right_bound_x * scale + center_x),
        ] = initial_mask[
            lower_bound_y * scale : upper_bound_y * scale,
            (left_bound_x * scale + center_x) : (right_bound_x * scale + center_x),
        ] * left_weight
        return weight_mask, None, None, initial_mask
    
    elif method == 'distance_error':
        # Iterate through all points in transformed_points
        filtered_points = [
            point for point in transformed_points if point[1] < 60 and 80 <= point[0] <= 120
        ]

        # Check if there are any points meeting the condition
        if filtered_points:
            # Find the point with the lowest y value
            start_point = min(filtered_points, key=lambda p: p[1])
        
        initial_mask , points = generate_curve_points_and_get_mask(start_point, transformed_points, num_iter=6)
        # Define two points
        if len(points) < 2:
            return initial_mask, None, None, initial_mask
        x1, y1 = points[0]
        x2, y2 = points[-1]

        # Calculate slope (m) and intercept (b) of the line: y = mx + b
        if x2 - x1 < 0.001:
            slope = (y2 - y1) / 0.001
        else:
            slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1

        return initial_mask, slope, intercept, initial_mask

    # Fit line using the specified method
    slope, intercept = fit_line_from_points(transformed_points, method=method)

    return weight_mask, slope, intercept, initial_mask


## New implementation of steering:

import numpy as np
from scipy.spatial.distance import cdist


def estimate_trajectory(start_pos, velocity, steering, dt, steps):
    """
    Estimate robot trajectory given velocity and steering
    """
    trajectory = [start_pos]
    x, y = start_pos
    theta = 0  # Initial heading
    
    for _ in range(steps):
        theta += steering * dt
        x += velocity * dt * np.cos(steering + np.pi / 2)
        y += velocity * dt * np.sin(steering + np.pi / 2)
        trajectory.append((x, y))
        
    return np.array(trajectory)

def calculate_lane_error(trajectory, yellow_points, white_points, start_idx):
    """
    Calculate error between trajectory and lane markers, and return shortest distance points
    Handles cases where points might be missing or are outliers (>15 distance)
    """
    total_error = 0
    valid_trajectory = True
    shortest_distance_points = []  # [(yellow_point, white_point) for each trajectory point]
    
    dist_yellow_cumulate = 0
    dist_white_cumulate = 0

    for i, pos in enumerate(trajectory[start_idx:]):
        nearest_yellow = None
        nearest_white = None
        dist_yellow = 12
        dist_white = 10
        
        # Process yellow points if they exist
        if len(yellow_points) > 3:
            yellow_dists = cdist([pos], yellow_points)
            dist_yellow = yellow_dists.min()
            nearest_yellow = yellow_points[yellow_dists.argmin()]
        
        # Process white points if they exist
        if len(white_points) > 3:
            white_dists = cdist([pos], white_points)
            dist_white = white_dists.min()
            nearest_white = white_points[white_dists.argmin()]
        
        # Append points (could be None for missing/outlier points)
        shortest_distance_points.append((nearest_yellow, nearest_white))
        
        # Check lane constraints only if both points exist
        if nearest_yellow is not None and nearest_white is not None and len(trajectory) > 1:
            trajectory_vector = trajectory[start_idx + i] - trajectory[start_idx + i - 1]
            yellow_vector = nearest_yellow - pos
            white_vector = nearest_white - pos
            
            if np.cross(trajectory_vector, yellow_vector) < 0 or np.cross(trajectory_vector, white_vector) > 0:
                valid_trajectory = False
        
        dist_yellow_cumulate += dist_yellow
        dist_white_cumulate += dist_white
        
    # Calculate error based on all available points
    total_error += abs(dist_yellow_cumulate - dist_white_cumulate)
    # If neither point exists, don't add to error (could also add a penalty here)
    
    return total_error, valid_trajectory, shortest_distance_points, dist_yellow_cumulate, dist_white_cumulate

def balance_bot(transformed_points, color_labels, start_pos=(50, 0)):
    """
    Find optimal steering and return trajectory with shortest distance points
    """
    # Separate yellow and white points
    yellow_points = np.array([p for p, c in zip(transformed_points, color_labels) if c == "yellow"])
    white_points = np.array([p for p, c in zip(transformed_points, color_labels) if c == "white"])
    
    # Parameters for trajectory estimation
    velocity = 20  # cm/s
    dt = 0.25      # seconds
    steps = 7     # number of steps to look ahead
    start_idx = 3

    # Search for optimal steering angle
    best_error = float('inf')
    optimal_steering = 0
    optimal_trajectory = None
    optimal_shortest_points = None
    dist_white = 1000 # Init with large number
    dist_yellow = 1000 # Init with large number
    
    # Threshold for minimum number of points
    threshold = 5

    # Check the number of yellow and white points
    num_yellow_points = len(yellow_points)
    num_white_points = len(white_points)

    # Case 1: there are both yellow and white, process the steering angle at normal
    if num_yellow_points >= threshold and num_white_points >= threshold:
        steering_angles = np.concatenate([
            np.linspace(-0.4, -0.1, 6),    # Dense at negative extreme
            np.linspace(-0.1, 0.1, 10),    # More dense in center
            np.linspace(0.1, 0.4, 6)       # Dense at positive extreme
        ])
    # Case 2: there is only white: the segment will center more on the right side
    elif num_yellow_points < threshold and num_white_points >= threshold:
        steering_angles = np.linspace(0.2, 0.8, 20)  # Dense at positive extreme
    # Case 3: there is only yellow: the segment will center more on the left side
    elif num_yellow_points >= threshold and num_white_points < threshold:
        steering_angles = np.linspace(-0.8, -0.2, 20)  # Dense at negative extreme
    # Case 4: neither yellow nor white points are detected (optional case)
    else:
        steering_angles = []

    
    # Try different steering angles
    for steering in steering_angles:  # rad/s
        trajectory = estimate_trajectory(start_pos, velocity, steering, dt, steps)
        error, valid, shortest_points, dist_yellow, dist_white = calculate_lane_error(trajectory, yellow_points, white_points, start_idx)
        
        if valid and error < best_error:
            best_error = error
            optimal_steering = steering
            optimal_trajectory = trajectory[start_idx:]
            optimal_shortest_points = shortest_points
    
    return optimal_steering, optimal_trajectory, optimal_shortest_points, best_error, dist_yellow, dist_white

def filter_points_iteratively(points, start_x, start_y, radius=6.5):
    """
    Filter points iteratively starting from a point, selecting furthest point in direction of travel
    
    Args:
        points (np.array): Array of points [(x,y), ...]
        start_x (float): Starting x coordinate
        start_y (float): Starting y coordinate
        radius (float): Radius to search for points in cm
        
    Returns:
        list: Filtered points in order of selection
    """
    if len(points) == 0:
        return []
        
    filtered_points = []
    start_x, start_y = rescale_and_shift_point([start_x, start_y])
    current_x, current_y = start_x, start_y
    previous_x, previous_y = None, None
    used_points = set()  # Keep track of used point indices
    
    while True:
        # Get points within radius of current position
        distances = np.sqrt(((points[:, 0] - current_x) ** 2) + 
                          ((points[:, 1] - current_y) ** 2))
        nearby_indices = np.where(distances <= radius)[0]
        
        # Remove already used points
        nearby_indices = [idx for idx in nearby_indices if idx not in used_points]
        
        if len(nearby_indices) == 0:
            break
            
        # Get nearby points
        nearby_points = points[nearby_indices]
        
        # For first point, choose furthest in y direction
        if previous_x is None:
            next_point_idx = nearby_indices[np.argmax(nearby_points[:, 1])]
        else:
            # Calculate direction vector from previous to current
            direction_x = current_x - previous_x
            direction_y = current_y - previous_y
            
            # Project points onto direction vector and choose furthest
            projections = ((nearby_points[:, 0] - current_x) * direction_x + 
                         (nearby_points[:, 1] - current_y) * direction_y)
            next_point_idx = nearby_indices[np.argmax(projections)]
        
        # Update positions and add point
        previous_x, previous_y = current_x, current_y
        current_x, current_y = points[next_point_idx]
        filtered_points.append(points[next_point_idx])
        used_points.add(next_point_idx)
    
    return np.array(filtered_points) if filtered_points else np.array([])

def get_trajectory_and_error(yellow_segments, white_segments, start_yellow=[-10, 10], start_white=[10, 10]):
    """
    Process segments and return masks with visualized trajectory and shortest distances
    """
    # Initialize masks
    yellow_mask = np.zeros((100 * scale, 100 * scale), dtype=np.float32)
    white_mask = np.zeros_like(yellow_mask)
    
    # Process yellow segments
    transformed_points = []
    color_labels = []
    yellow_points = []
    white_points = []
    
    for segment in yellow_segments:
        pt1 = (segment.points[0].y * -100, segment.points[0].x * 100)
        pt2 = (segment.points[1].y * -100, segment.points[1].x * 100)
        
        if (lower_bound_y <= pt1[1] <= upper_bound_y and 
            lower_bound_y <= pt2[1] <= upper_bound_y):
            new_pt1 = rescale_and_shift_point(pt1)
            new_pt2 = rescale_and_shift_point(pt2)
            yellow_points.extend([new_pt1, new_pt2])
    
    # Process white segments
    for segment in white_segments:
        pt1 = (segment.points[0].y * -100, segment.points[0].x * 100)
        pt2 = (segment.points[1].y * -100, segment.points[1].x * 100)
                
        if (lower_bound_y <= pt1[1] <= upper_bound_y and 
            lower_bound_y <= pt2[1] <= upper_bound_y):
            new_pt1 = rescale_and_shift_point(pt1)
            new_pt2 = rescale_and_shift_point(pt2)
            white_points.extend([new_pt1, new_pt2])

    # Filter points iteratively
    yellow_points = np.array(yellow_points)
    white_points = np.array(white_points)

    if len(yellow_points) > 0:
        filtered_yellow = filter_points_iteratively(yellow_points, start_yellow[0], start_yellow[1])
        transformed_points.extend(filtered_yellow)
        color_labels.extend(["yellow"] * len(filtered_yellow))
        # Draw filtered yellow points on yellow mask
        for point in filtered_yellow:
            pt = tuple(map(int, point))
            cv2.circle(yellow_mask, pt, 1, 1, -1)  # Draw small filled circle

    if len(white_points) > 0:
        filtered_white = filter_points_iteratively(white_points, start_white[0], start_white[1])
        transformed_points.extend(filtered_white)
        color_labels.extend(["white"] * len(filtered_white))
        # Draw filtered white points on white mask
        for point in filtered_white:
            pt = tuple(map(int, point))
            cv2.circle(white_mask, pt, 1, 1, -1)  # Draw small filled circle
    
    # If the result show nothing, return default 0 steering 
    if len(yellow_points) == 0 and len(white_points) == 0:
        return yellow_mask, white_mask, 0, 0, 0
        
    # Get optimal trajectory and shortest distance points
    optimal_steering, optimal_trajectory, shortest_distance_points, best_error, dist_yellow, dist_white = balance_bot(
        transformed_points, 
        color_labels
    )
    
    # Draw trajectory and shortest distance lines on masks
    if optimal_trajectory is not None:
        for i in range(len(optimal_trajectory) - 1):
            pt1 = tuple(map(int, optimal_trajectory[i]))
            pt2 = tuple(map(int, optimal_trajectory[i + 1]))
            
            # Draw trajectory on both masks
            yellow_mask = cv2.line(yellow_mask, pt1, pt2, 0.5, thickness=2)
            white_mask = cv2.line(white_mask, pt1, pt2, 0.5, thickness=2)
            
            # Draw shortest distance lines
            if shortest_distance_points and i < len(shortest_distance_points):
                yellow_point, white_point = shortest_distance_points[i]
                if yellow_point is not None:
                    yellow_pt = tuple(map(int, yellow_point))
                    yellow_mask = cv2.line(yellow_mask, pt1, yellow_pt, 0.75, thickness=1)
                if white_point is not None:
                    white_pt = tuple(map(int, white_point))
                    white_mask = cv2.line(white_mask, pt1, white_pt, 0.75, thickness=1)
    

    yellow_mask = cv2.flip(yellow_mask, 1)
    white_mask = cv2.flip(white_mask, 1)
    
    return yellow_mask, white_mask, dist_yellow, dist_white, optimal_steering

### Left over
# def calculate_steering(self, white_segments, yellow_segments):
#         def rescale(a: float, L: float, U: float):
#             if np.allclose(L, U):
#                 return 0.0
#             return (a - L) / (U - L)

#         def rescale_and_shift_point(point, scale=1, center_x=100):
#             x, y = point
#             new_x = int(x * scale + center_x)
#             new_y = int(y * scale)
#             return new_x, new_y

#         def get_weight_matrix(segments, color, scale=2, weight_matrix_size=100):
#             center_x = 100
#             lower_bound_y = 0
#             upper_bound_y = 40

#             if color == "yellow":
#                 left_bound_x, right_bound_x = -15, 25
#                 left_weight, right_weight = -0.75, -0.5
#             elif color == "white":
#                 left_bound_x, right_bound_x = -25, 15
#                 left_weight, right_weight = 0.75 * 170 / 224, 0.5
#             else:
#                 raise ValueError(f"Unsupported color: {color}")

#             initial_mask = np.zeros((weight_matrix_size * scale, weight_matrix_size * scale), dtype=np.float32)
#             weight_mask = np.zeros_like(initial_mask)

#             for segment in segments:
#                 pt1 = (segment.points[0].y * -100, segment.points[0].x * 100)
#                 pt2 = (segment.points[1].y * -100, segment.points[1].x * 100)
#                 # Check if x is not between -50 and 50, and y is not between 0 and 40
#                 if not (-50 <= pt1[0] <= 50) and not (0 <= pt1[1] <= 40) and not (-50 <= pt2[0] <= 50) and not (0 <= pt2[1] <= 40):
#                     continue
#                 new_pt1 = rescale_and_shift_point(pt1, scale=scale, center_x=center_x)
#                 new_pt2 = rescale_and_shift_point(pt2, scale=scale, center_x=center_x)
#                 # self.log(f"pt1: {new_pt1}. pt2: {new_pt2}")
#                 initial_mask = cv2.line(initial_mask, new_pt1, new_pt2, 1, thickness=2)
#             # # Define the points where you want to draw circles
#             # points = [(10, 180), (180, 30), (180, 180)]

#             # # Set the thickness of the circle
#             # thickness = 2

#             # # Loop through the points and draw circles
#             # for pt in points:
#             #     initial_mask = cv2.circle(initial_mask, pt, radius=5, color=255, thickness=thickness)

#             # initial_mask = cv2.flip(initial_mask, 1)
#             # weight_mask[
#             #     lower_bound_y * scale : upper_bound_y * scale,
#             #     (left_bound_x * scale + center_x) : (right_bound_x * scale + center_x),
#             # ] = initial_mask[
#             #     lower_bound_y * scale : upper_bound_y * scale,
#             #     (left_bound_x * scale + center_x) : (right_bound_x * scale + center_x),
#             # ] * left_weight

#             # Normalize the weight_mask and clip the values to the range [0, 255]
#             # normalized_weight_mask = np.abs(weight_mask) * 255
#             # normalized_weight_mask = np.abs(weight_mask) * 255
#             # normalized_weight_mask = np.clip(normalized_weight_mask, 0, 255).astype(np.uint8)

#             # Convert grayscale weight_mask to RGB
#             # rgb_weight_mask = cv2.cvtColor(normalized_weight_mask, cv2.COLOR_GRAY2RGB)
            

#             # # Create a compressed image message
#             # debug_image_msg = self.bridge.cv2_to_compressed_imgmsg(rgb_weight_mask)

#             # if color == "white":
#             #     self.pub_weight_image_white.publish(debug_image_msg)
#             # elif color == "yellow":
#             #     self.pub_weight_image_yellow.publish(debug_image_msg)

#             return weight_mask

#         yellow_weight_mask = get_weight_matrix(yellow_segments, "yellow", scale=2)
#         white_weight_mask = get_weight_matrix(white_segments, "white", scale=2)

#         total_weight_matrix = yellow_weight_mask + white_weight_mask
#         steer = float(np.sum(total_weight_matrix))

#         steer_max = 1.0
#         omega_max = 6.0

#         self.log(f"steer: {steer}")
#         steer_scaled = np.sign(steer) * rescale(min(np.abs(steer), steer_max), 0, steer_max)
        
#         self.log(f"steer scaled: {steer_scaled}")
#         return steer_scaled * omega_max

#     def calculate_steering(self, white_segments, yellow_segments):
#         def rescale(a: float, L: float, U: float):
#             if np.allclose(L, U):
#                 return 0.0
#             return (a - L) / (U - L)

#         def rescale_and_shift_point(point, scale=1, center_x=100):
#             x, y = point
#             new_x = int(x * scale + center_x)
#             new_y = int(y * scale)
#             return new_x, new_y

#         def fit_line_from_points(points):
#             if len(points) < 2:
#                 return None, None  # Not enough points to fit a line

#             x_vals, y_vals = zip(*points)
#             coef = np.polyfit(x_vals, y_vals, 1)  # Linear regression y = mx + c
#             return coef[0], coef[1]  # Return slope (m) and intercept (c)

#         def distance_point_to_line(x, y, slope, intercept):
#             if slope is None or intercept is None:
#                 return float('inf')  # Return a large distance if the line cannot be fitted
#             return abs(slope * x - y + intercept) / (np.sqrt(slope**2 + 1))

#         def get_weight_matrix(segments, color, scale=2, weight_matrix_size=100):
#             center_x = 100
#             lower_bound_y = 0
#             upper_bound_y = 40

#             if color == "yellow":
#                 left_bound_x, right_bound_x = -15, 25
#                 left_weight, right_weight = -0.75, -0.5
#             elif color == "white":
#                 left_bound_x, right_bound_x = -25, 15
#                 left_weight, right_weight = 0.75 * 170 / 224, 0.5
#             else:
#                 raise ValueError(f"Unsupported color: {color}")

#             initial_mask = np.zeros((weight_matrix_size * scale, weight_matrix_size * scale), dtype=np.float32)
#             weight_mask = np.zeros_like(initial_mask)
#             transformed_points = []

#             self.log(f"process segment color: {color}")
#             for segment in segments:
#                 pt1 = (segment.points[0].y * -100, segment.points[0].x * 100)
#                 pt2 = (segment.points[1].y * -100, segment.points[1].x * 100)
#                 self.log(f"pt1: {pt1}, pt2: {pt2}")
#                 if (left_bound_x <= pt1[0] <= right_bound_x) and (lower_bound_y <= pt1[1] <= upper_bound_y) and (left_bound_x <= pt2[0] <= right_bound_x) and (lower_bound_y <= pt2[1] <= upper_bound_y):  
#                     new_pt1 = rescale_and_shift_point(pt1, scale=scale, center_x=center_x)
#                     new_pt2 = rescale_and_shift_point(pt2, scale=scale, center_x=center_x)
#                     transformed_points.extend([pt1, pt2])
#                     initial_mask = cv2.line(initial_mask, new_pt1, new_pt2, 1, thickness=2)
                    

#             initial_mask = cv2.flip(initial_mask, 1)
#             weight_mask[
#                 lower_bound_y * scale : upper_bound_y * scale,
#                 (left_bound_x * scale + center_x) : (right_bound_x * scale + center_x),
#             ] = initial_mask[
#                 lower_bound_y * scale : upper_bound_y * scale,
#                 (left_bound_x * scale + center_x) : (right_bound_x * scale + center_x),
#             ] * left_weight

#             slope, intercept = fit_line_from_points(transformed_points)
#             return weight_mask, slope, intercept

#         yellow_weight_mask, yellow_slope, yellow_intercept = get_weight_matrix(yellow_segments, "yellow", scale=2)
#         white_weight_mask, white_slope, white_intercept = get_weight_matrix(white_segments, "white", scale=2)

#         # Combine the masks into a single image
#         total_weight_matrix = yellow_weight_mask + white_weight_mask
#         combined_mask = np.abs(yellow_weight_mask) + np.abs(white_weight_mask)
#         combined_mask_normalized = np.clip(combined_mask * 255, 0, 255).astype(np.uint8)
#         combined_mask_rgb = cv2.cvtColor(combined_mask_normalized, cv2.COLOR_GRAY2RGB)

#         # Add circles to the debug image
#         point_a = (0, 10)
#         scaled_point_a = rescale_and_shift_point(point_a, scale=2, center_x=100)
#         cv2.circle(combined_mask_rgb, scaled_point_a, radius=5, color=(0, 0, 255), thickness=-1)  # Red circle for Point A

#         # Draw the fitted lines
#         if yellow_slope is not None:
#             start_yellow = rescale_and_shift_point((0, yellow_slope * 0 + yellow_intercept), scale=2, center_x=100)
#             end_yellow = rescale_and_shift_point((40, yellow_slope * 40 + yellow_intercept), scale=2, center_x=100)
#             cv2.line(combined_mask_rgb, start_yellow, end_yellow, color=(0, 255, 255), thickness=2)  # Yellow line

#         if white_slope is not None:
#             start_white = rescale_and_shift_point((0, white_slope * 0 + white_intercept), scale=2, center_x=100)
#             end_white = rescale_and_shift_point((40, white_slope * 40 + white_intercept), scale=2, center_x=100)
#             cv2.line(combined_mask_rgb, start_white, end_white, color=(255, 255, 255), thickness=2)  # White line

#         # Calculate and log distances
#         yellow_distance = distance_point_to_line(point_a[0], point_a[1], yellow_slope, yellow_intercept)
#         white_distance = distance_point_to_line(point_a[0], point_a[1], white_slope, white_intercept)

#         # Publish the combined debug image
#         debug_image_msg = self.bridge.cv2_to_compressed_imgmsg(combined_mask_rgb)
#         self.pub_weight_image_white.publish(debug_image_msg)

#         steer = float(np.sum(total_weight_matrix))
#         steer_max = 1.0
#         omega_max = 6.0
#         steer_scaled = np.sign(steer) * rescale(min(np.abs(steer), steer_max), 0, steer_max)
#         self.log(f"steer: {steer}")
#         self.log(f"steer scaled: {steer_scaled}")
#         return steer_scaled * omega_max

## Fitting line method for calculating steering

    # def calculate_steering(self, white_segments, yellow_segments, method):
    #     """
    #     Calculate steering angle based on detected lane lines.
        
    #     Args:
    #         white_segments: Detected white lane segments
    #         yellow_segments: Detected yellow lane segments
    #         method: Method for line fitting
            
    #     Returns:
    #         float: Scaled steering angle
    #     """
    #     # Get line parameters for both yellow and white lanes
    #     yellow_weight_mask, yellow_slope, yellow_intercept, yellow_initial_mask = get_weight_matrix(
    #         yellow_segments, "yellow", method=method
    #     )
    #     white_weight_mask, white_slope, white_intercept, white_initial_mask = get_weight_matrix(
    #         white_segments, "white", method=method
    #     )

    #     self.log(f"yellow equation: y = {yellow_slope} * x + {yellow_intercept}")
    #     self.log(f"white equation: y = {white_slope} * x + {white_intercept}")

    #     # Reference point for lane centering (can be adjusted)
    #     point_a = (100, 20)  # (x, y) in image coordinates
        
    #     # Calculate distances to both lines if they exist
    #     steering_value = 0
    #     yellow_distance = distance_point_to_line(
    #         point_a[0], point_a[1], yellow_slope, yellow_intercept
    #     )
    #     white_distance = distance_point_to_line(
    #         point_a[0], point_a[1], white_slope, white_intercept
    #     )

    #     self.log(f"yellow_distance: {yellow_distance}")
    #     self.log(f"white_distance")
    #     # Normalize the steering value (negative: turn left, positive: turn right)
    #     steering_value = (yellow_distance - white_distance) / (yellow_distance + white_distance) / 10
        
    #     self.log(f"Raw steering value: {steering_value}")

    #     # Scale the steering value within bounds
    #     steering_scaled = (
    #         np.sign(steering_value)
    #         * self.rescale(
    #             min(abs(steering_value), self.steer_max),
    #             0,
    #             self.steer_max
    #         )
    #     )

    #     # # Create debug visualization
    #     # debug_image = self.create_debug_visualization(
    #     #     yellow_initial_mask,
    #     #     white_initial_mask,
    #     #     yellow_slope,
    #     #     yellow_intercept,
    #     #     white_slope,
    #     #     white_intercept
    #     # )

    #     # # Publish debug image
    #     # debug_msg = self.bridge.cv2_to_compressed_imgmsg(debug_image)
    #     # self.pub_debug_image.publish(debug_msg)

    #     self.log(f"Scaled steering value: {steering_scaled}")
    #     return steering_scaled * self.omega_max