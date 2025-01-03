import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

# Class definitions remain unchanged
class AlignmentTest:
    def __init__(self, alignment_steering_gain=1.0, omega_max=1.0):
        self.alignment_steering_gain = alignment_steering_gain
        self.omega_max = omega_max

    def log(self, message):
        print(message)

    def calculate_red_line_alignment(self, red_segments, x_min, x_max, y_min, y_max):
        """
        Calculate steering angle to align perpendicular to red line
        Returns: steering_angle in radians or None if no valid angle can be calculated
        """
        if not red_segments:
            return None
        
        # Collect all points from red segments
        red_points = []
        for segment in red_segments:
            pt1 = (segment.points[0][0], segment.points[0][1])
            pt2 = (segment.points[1][0], segment.points[1][1])
            if x_min <= pt1[0] <= x_max and y_min <= pt1[1] <= y_max \
                    and x_min <= pt2[0] <= x_max and y_min <= pt2[1] <= y_max:
                red_points.extend([pt1, pt2])
                print(f"pt1: {pt1}, pt2: {pt2}")
                
        
        if len(red_points) < 4:  # Need at least 4 points to form a rectangle
            return None
        
        # Find extreme coordinates
        max_x = max(point[0] for point in red_points)
        min_x = min(point[0] for point in red_points)
        max_y = max(point[1] for point in red_points)
        min_y = min(point[1] for point in red_points)
    
        # Find corner points
        top_right_point = None
        top_left_point = None
        bottom_right_point = None
        bottom_left_point = None

        # Initialize minimum distances for each corner
        min_dist_top_right = float('inf')
        min_dist_top_left = float('inf')
        min_dist_bottom_right = float('inf')
        min_dist_bottom_left = float('inf')

        # For each point, check its distance to each corner
        for point in red_points:
            x, y = point
            
            # Top right corner
            dist_top_right = ((x - max_x) ** 2 + (y - max_y) ** 2)
            if dist_top_right < min_dist_top_right:
                top_right_point = point
                min_dist_top_right = dist_top_right
            
            # Top left corner
            dist_top_left = ((x - min_x) ** 2 + (y - max_y) ** 2)
            if dist_top_left < min_dist_top_left:
                top_left_point = point
                min_dist_top_left = dist_top_left
            
            # Bottom right corner
            dist_bottom_right = ((x - max_x) ** 2 + (y - min_y) ** 2)
            if dist_bottom_right < min_dist_bottom_right:
                bottom_right_point = point
                min_dist_bottom_right = dist_bottom_right
            
            # Bottom left corner
            dist_bottom_left = ((x - min_x) ** 2 + (y - min_y) ** 2)
            if dist_bottom_left < min_dist_bottom_left:
                bottom_left_point = point
                min_dist_bottom_left = dist_bottom_left

        # Check if we found all corners
        if not all([top_right_point, top_left_point, bottom_right_point, bottom_left_point]):
            self.log("Failed to find all corner points")
            return None
        
        print(f"top_right_point: {top_right_point}, top_left_point: {top_left_point}")
        print(f"bottom_right_point: {bottom_right_point}, bottom_left_point: {bottom_left_point}")

        # Calculate dx and dy for top and bottom edges
        dx_top = top_right_point[0] - top_left_point[0]
        dx_bottom = bottom_right_point[0] - bottom_left_point[0]
        
        dy_top = top_right_point[1] - top_left_point[1]
        dy_bottom = bottom_right_point[1] - bottom_left_point[1]
        
        # Calculate angle using average dx and dy
        line_angle_top = np.arctan2(dy_top, dx_top)
        line_angle_bottom = np.arctan2(dy_bottom, dx_bottom)
        line_angle = np.mean([line_angle_top, line_angle_bottom])

        print(f"line angle top: {line_angle_top}, line angle bottom: {line_angle_bottom}")
        
        # Calculate desired angle (perpendicular to line)
        target_angle = line_angle + np.pi/2
        
        # Limit steering angle
        return np.clip(target_angle, -self.omega_max + np.pi/2, self.omega_max + + np.pi/2), \
               [top_right_point, top_left_point, bottom_left_point, bottom_right_point]

# Generate random points for testing
np.random.seed(42)
mean = [0, 0]
covariance = [[6, 0], [0, 1]]
points = np.random.multivariate_normal(mean, covariance, 100)

class Segment:
    def __init__(self, points):
        self.points = points

red_segments = [Segment([np.array([x, y]), np.array([x + np.random.uniform(-0.5, 0.5), y + np.random.uniform(-0.5, 0.5)])]) for x, y in points]

x_min, x_max, y_min, y_max = -5, 5, -0.5, 0.5

alignment_test = AlignmentTest()
steering_angle, corners = alignment_test.calculate_red_line_alignment(red_segments, x_min, x_max, y_min, y_max)

plt.figure(figsize=(10, 10))

# Plot all points
plt.scatter(points[:, 0], points[:, 1], color='red', label='Red Points')

# Plot boundary if corners are found
if corners:
    boundary_polygon = Polygon(corners, closed=True, fill=False, edgecolor='blue', linewidth=2, label='Boundary')
    plt.gca().add_patch(boundary_polygon)

boundary = Polygon([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]], closed=True, fill=False, edgecolor='red', linewidth=2, label='Boundary')
plt.gca().add_patch(boundary)

# Add origin and angle line
if steering_angle is not None:
    dx = np.cos(steering_angle)
    dy = np.sin(steering_angle)
    plt.quiver(0, 0, dx, dy, angles='xy', scale_units='xy', scale=1, color='green', label='Steering Angle')

# Labels and legend
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.grid(color='gray', linestyle='--', linewidth=0.5)
plt.title("Red Line Alignment Test with Boundary")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.axis('equal')
plt.show()
