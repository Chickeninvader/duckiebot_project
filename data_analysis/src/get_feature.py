import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import (LinearRegression, RANSACRegressor, 
                                HuberRegressor, TheilSenRegressor, 
                                Ridge, Lasso, ElasticNet, 
                                BayesianRidge, ARDRegression, 
                                SGDRegressor, PassiveAggressiveRegressor)
from sklearn.preprocessing import PolynomialFeatures
import cv2
from scipy.spatial.distance import cdist

# Global variable

scale=1
center_x=50
epsilon = 2.5
# Optional: Add prior knowledge
prior_mean = 6.0  # Expected slope
prior_std = 0.7   # Uncertainty in slope
lower_bound_y = 0
upper_bound_y = 35
    
# Logic to get feature useful for steering

def rescale( a, L, U):
        if np.allclose(L, U):
            return 0.0
        return (a - L) / (U - L)

def rescale_and_shift_point( point):
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
        return 50.0  # Return a large distance if the line cannot be fitted
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

def generate_curve_points_and_get_mask(start_point, transformed_points, num_iter=12):
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
        start_point = (-10, 20)
        start_point = rescale_and_shift_point(start_point)
        left_bound_x, right_bound_x = -35, 35
        left_weight, right_weight = -0.01, -0.5
    elif color == "white":
        start_point = (10, 20)
        start_point = rescale_and_shift_point(start_point)
        left_bound_x, right_bound_x = -35, 35
        left_weight, right_weight = 0.01, 0.5
    else:
        raise ValueError(f"Unsupported color: {color}")

    initial_mask = np.zeros((100 * scale, 100 * scale), dtype=np.float32)
    weight_mask = np.zeros_like(initial_mask)
    transformed_points = []

    for segment in segments:
        pt1 = (segment.points[0].x, segment.points[0].y)
        pt2 = (segment.points[1].x, segment.points[1].y)
        if (
            left_bound_x <= pt1[0] <= right_bound_x
            and lower_bound_y <= pt1[1] <= upper_bound_y
            and left_bound_x <= pt2[0] <= right_bound_x
            and lower_bound_y <= pt2[1] <= upper_bound_y
        ):
            new_pt1 = rescale_and_shift_point(pt1)
            new_pt2 = rescale_and_shift_point(pt2)
            transformed_points.extend([new_pt1, new_pt2])
            initial_mask = cv2.line(initial_mask, new_pt1, new_pt2, 1, thickness=2)

    initial_mask = cv2.flip(initial_mask, 1)
    
    if method == 'matrix':
        weight_mask[
            lower_bound_y * scale : upper_bound_y * scale,
            (left_bound_x * scale + center_x) : (right_bound_x * scale + center_x),
        ] = initial_mask[
            lower_bound_y * scale : upper_bound_y * scale,
            (left_bound_x * scale + center_x) : (right_bound_x * scale + center_x),
        ] * left_weight
        return weight_mask, None, None, initial_mask
    
    elif method == 'update_matrix':
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
            point for point in transformed_points if point[1] < 15 and center_x - 10 * scale <= point[0] <= center_x + 10 * scale
        ]

        # Check if there are any points meeting the condition
        if filtered_points:
            # Find the point with the lowest y value
            start_point = min(filtered_points, key=lambda p: p[1])
    
        weight_mask, points = generate_curve_points_and_get_mask(start_point, transformed_points, num_iter=12)
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
    
    for i, pos in enumerate(trajectory[start_idx:]):
        nearest_yellow = None
        nearest_white = None
        dist_yellow = 10
        dist_white = 10
        
        # Process yellow points if they exist and larger than threshold
        if len(yellow_points) > 3:
            yellow_dists = cdist([pos], yellow_points)
            min_yellow_dist = yellow_dists.min()
            dist_yellow = min_yellow_dist
            nearest_yellow = yellow_points[yellow_dists.argmin()]
            # if min_yellow_dist <= 15:  # Only consider if within threshold
            #     dist_yellow = min_yellow_dist
            #     nearest_yellow = yellow_points[yellow_dists.argmin()]
        
        # Process white points if they exist and larger than threshold
        if len(white_points) > 3:
            white_dists = cdist([pos], white_points)
            min_white_dist = white_dists.min()
            dist_white = min_white_dist
            nearest_white = white_points[white_dists.argmin()]
            # if min_white_dist <= 15:  # Only consider if within threshold
            #     dist_white = min_white_dist
            #     nearest_white = white_points[white_dists.argmin()]
        
        # Append points (could be None for missing/outlier points)
        shortest_distance_points.append((nearest_yellow, nearest_white))
        
        # Check lane constraints only if both points exist
        if nearest_yellow is not None and nearest_white is not None and len(trajectory) > 1:
            trajectory_vector = trajectory[start_idx + i] - trajectory[start_idx + i - 1]
            yellow_vector = nearest_yellow - pos
            white_vector = nearest_white - pos
            
            if np.cross(trajectory_vector, yellow_vector) < 0 or np.cross(trajectory_vector, white_vector) > 0:
                valid_trajectory = False
        
        # Calculate error based on available points
        total_error += abs(dist_white - dist_yellow)
        # If neither point exists, don't add to error (could also add a penalty here)
    
    return total_error, valid_trajectory, shortest_distance_points, dist_yellow, dist_white

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
    steps = 8     # number of steps to look ahead
    start_idx = 3

    # Search for optimal steering angle
    best_error = float('inf')
    optimal_steering = 0
    optimal_trajectory = None
    optimal_shortest_points = None

    steering_angles = np.concatenate([
        np.linspace(-0.4, -0.1, 6),    # Dense at negative extreme
        np.linspace(-0.1, 0.1, 10),    # More dense in center
        np.linspace(0.1, 0.4, 6)       # Dense at positive extreme
    ])
    
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

def filter_points_iteratively(points, start_x, start_y, radius=5):
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

def get_trajectory_and_error(white_segments, yellow_segments):
    """
    Process segments and return masks with visualized trajectory and shortest distances
    """
    # Initialize masks
    yellow_mask = np.zeros((100 * scale, 100 * scale), dtype=np.float32)
    white_mask = np.zeros_like(yellow_mask)
    
    # Process yellow segments
    yellow_points = []
    white_points = []
    transformed_points = []
    color_labels = []
    
    # Collect yellow points
    for segment in yellow_segments:
        pt1 = (segment.points[0].x, segment.points[0].y)
        pt2 = (segment.points[1].x, segment.points[1].y)
        if (lower_bound_y <= pt1[1] <= upper_bound_y and 
            lower_bound_y <= pt2[1] <= upper_bound_y):
            new_pt1 = rescale_and_shift_point(pt1)
            new_pt2 = rescale_and_shift_point(pt2)
            yellow_points.extend([new_pt1, new_pt2])
            
    # Collect white points
    for segment in white_segments:
        pt1 = (segment.points[0].x, segment.points[0].y)
        pt2 = (segment.points[1].x, segment.points[1].y)
        if (lower_bound_y <= pt1[1] <= upper_bound_y and 
            lower_bound_y <= pt2[1] <= upper_bound_y):
            new_pt1 = rescale_and_shift_point(pt1)
            new_pt2 = rescale_and_shift_point(pt2)
            white_points.extend([new_pt1, new_pt2])
            
    # Filter points iteratively
    yellow_points = np.array(yellow_points)
    white_points = np.array(white_points)
    
    if len(yellow_points) > 0:
        filtered_yellow = filter_points_iteratively(yellow_points, -10, 10)
        transformed_points.extend(filtered_yellow)
        color_labels.extend(["yellow"] * len(filtered_yellow))
        # Draw filtered yellow points on yellow mask
        for point in filtered_yellow:
            pt = tuple(map(int, point))
            cv2.circle(yellow_mask, pt, 1, 1, -1)  # Draw small filled circle

    if len(white_points) > 0:
        filtered_white = filter_points_iteratively(white_points, 10, 10)
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