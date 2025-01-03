import re
import matplotlib.pyplot as plt  # For plotting
from get_steering import SteeringController
import os
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
import cv2
import numpy as np

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

@dataclass
class Point:
    x: float
    y: float

@dataclass
class Segment:
    points: List[Point]

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter

def save_gif(masks, output_folder):
    gif_path = output_folder
    height, width, _ = masks[0].shape
    print(f'result save to {output_folder}')
    
    # Save as AVI first
    out = cv2.VideoWriter(
        gif_path.replace(".gif", ".avi"),
        cv2.VideoWriter_fourcc(*"XVID"),
        10,
        (width, height),
        isColor=True  # Make sure isColor is True for colored masks
    )
    
    # Add frame numbers to masks
    numbered_masks = []
    for i, mask in enumerate(masks):
        # Create a copy of the mask to avoid modifying the original
        mask_with_number = mask.copy().astype(np.uint8)
        
        # Add frame number
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = min(width, height) / 500  # Scale font size based on image dimensions
        thickness = max(2, int(min(width, height) / 250))  # Scale thickness based on image dimensions
        text = f'Frame {i+1}'
        
        # Get text size to position it properly
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Position text in top-left corner with some padding
        text_x = 10
        text_y = text_height + 10
        
        # Add white background rectangle for better visibility
        cv2.rectangle(mask_with_number, 
                     (text_x - 5, text_y - text_height - 5),
                     (text_x + text_width + 5, text_y + 5),
                     (255, 255, 255),
                     -1)
        
        # Add text
        cv2.putText(mask_with_number,
                    text,
                    (text_x, text_y),
                    font,
                    font_scale,
                    (0, 0, 0),
                    thickness)
        
        numbered_masks.append(mask_with_number)
        out.write(mask_with_number)
    
    out.release()
    
    # Convert to GIF
    frames = [cv2.cvtColor(mask, cv2.COLOR_BGR2RGB) for mask in numbered_masks]
    fig, ax = plt.subplots(figsize=(width/100, height/100))
    gif_writer = PillowWriter(fps=10)
    
    with gif_writer.saving(fig, gif_path, dpi=100):
        for frame in frames:
            ax.clear()
            ax.imshow(frame)
            ax.axis('off')
            fig.tight_layout(pad=0)
            gif_writer.grab_frame()
    
    plt.close(fig)
        
def extract_point(line: str):
    """Extract points from a line if it contains point information."""
    point_pattern = r"pt1: \((.+?), (.+?)\), pt2: \((.+?), (.+?)\)"
    match = re.search(point_pattern, line)
    if match:
        pt1 = Point(x=float(match.group(1)), y=float(match.group(2)))
        pt2 = Point(x=float(match.group(3)), y=float(match.group(4)))
        return pt1, pt2
    return None

def extract_segments_list(log_data_path: str) -> List[Tuple[Dict, float]]:
    """
    Extracts segments following the strict pattern:
    delta_time -> yellow segments -> white segments
    
    Args:
        log_data_path (str): Path to the log file
        
    Returns:
        List[Tuple[Dict, float]]: List of (segments, delta_time) pairs
    """
    segment_list = []
    
    with open(log_data_path, "r") as file:
        lines = file.readlines()
        i = 0
        
        while i < len(lines):
            try:
                # 1. Find delta time
                while i < len(lines):
                    if "Delta time between callbacks:" in lines[i]:
                        delta_time = float(re.search(r"Delta time between callbacks: (\d+\.\d+)", lines[i]).group(1))
                        i += 1
                        break
                    i += 1
                else:
                    # No more delta time found
                    break

                current_segments = {"yellow": [], "white": []}

                # 2. Find yellow segment marker
                while i < len(lines):
                    if "process segment color: yellow" in lines[i]:
                        i += 1
                        break
                    i += 1
                else:
                    raise ValueError("Yellow segment marker not found after delta time")

                # 3. Process yellow points until white segment marker
                while i < len(lines):
                    if "process segment color: white" in lines[i]:
                        break
                    
                    points = extract_point(lines[i])
                    if points:
                        current_segments["yellow"].append(Segment(points=list(points)))
                    i += 1
                else:
                    raise ValueError("White segment marker not found after yellow points")

                # 4. Process white points until next delta time or end
                i += 1  # Skip the white marker line
                while i < len(lines):
                    if "Delta time between callbacks:" in lines[i]:
                        break
                    
                    points = extract_point(lines[i])
                    if points:
                        current_segments["white"].append(Segment(points=list(points)))
                    i += 1

                segment_list.append((current_segments, delta_time))

            except Exception as e:
                print(f"Error processing segments: {str(e)}")
                return segment_list  # Return what we have so far

    return segment_list

def plot_steering_results(
    segments_list: List[Tuple[Dict, float]], 
    steering_method_list: List[str], 
    controller_list: List[str], 
    output_folder: str
) -> None:
    """
    Plot steering results using the SteeringController for multiple methods and save combined outputs.

    Args:
        segments_list: List of tuples [(segments, delta_t), ...] where segments contains 'white' and 'yellow' segments
        steering_method_list: List of line fitting methods to use
        controller_list: List of controller methods to use
        output_folder: Base folder name to save the output
    """
    # Fix the path construction and ensure directory exists
    output_folder = os.path.join(os.path.expanduser("~"), "Desktop", "albert", "data_analysis", "output_data", output_folder)
    os.makedirs(output_folder, exist_ok=True)

    # Initialize result storage with proper typing
    steering_results: Dict[str, Dict[str, List[float]]] = {}
    combined_masks: List[np.ndarray] = []
    distance_results: List[Tuple[float, float, float, float]] = []  # Add missing distance_results list

    # Initialize steering controller
    steering_controller = SteeringController()
    
    for steering_method in steering_method_list:
        for controller in controller_list:
            # Initialize steering controller
            steering_controller = SteeringController()
            combined_name = f"{steering_method}-{controller}"
            steering_results[combined_name] = {
                "timestamps": [],
                "original_omegas": [],
                "controlled_omegas": []
            }
            curr_time = 0

            for i, (segments, delta_t) in enumerate(segments_list):
                
                white_segments = segments.get("white", [])
                yellow_segments = segments.get("yellow", [])

                result = steering_controller.get_velocity_and_orientation(
                    v_0=steering_controller.v_0,
                    white_segments=white_segments,
                    yellow_segments=yellow_segments,
                    steering_method=steering_method,
                    controller_method=controller,
                    delta_t=delta_t
                )
                
                controlled_omega, yellow_mask, white_mask, yellow_distance, white_distance, original_omega = result
                
                # Add distance results (only once per timestamp)
                if controller == controller_list[0] and steering_method == steering_method_list[0]:
                    distance_results.append((curr_time, yellow_distance, white_distance, yellow_distance - white_distance))

                curr_time += delta_t
                steering_results[combined_name]["timestamps"].append(curr_time)
                steering_results[combined_name]["original_omegas"].append(original_omega)
                steering_results[combined_name]["controlled_omegas"].append(controlled_omega)

                # Build combined masks only once per frame
                if controller == controller_list[0] and steering_method == steering_method_list[0]:
                    combined = create_combined_mask(yellow_mask, white_mask)
                    combined_masks.append(combined)

    # Save results
    save_visualization_results(output_folder, combined_masks, steering_results, distance_results)

def create_combined_mask(yellow_mask: np.ndarray, white_mask: np.ndarray, scale_factor: int = 4, grid_spacing_cm: int = 20) -> np.ndarray:
    """
    Create a combined visualization mask with proper coloring and grid overlay.
    Origin is set at (100, 0) of the original mask, with coordinates in centimeters.
    
    Args:
        yellow_mask: Binary mask for yellow line segments (100x100)
        white_mask: Binary mask for white line segments (100x100)
        scale_factor: Factor to scale up the visualization (default: 4)
        grid_spacing_cm: Spacing between grid lines in centimeters (default: 20)
    
    Returns:
        np.ndarray: Combined visualization with colored masks and grid overlay
    """
    # Scale up the masks
    height, width = yellow_mask.shape
    scaled_height = height * scale_factor
    scaled_width = width * scale_factor
    
    yellow_scaled = cv2.resize(yellow_mask, (scaled_width, scaled_height), interpolation=cv2.INTER_NEAREST)
    white_scaled = cv2.resize(white_mask, (scaled_width, scaled_height), interpolation=cv2.INTER_NEAREST)
    
    # Create base colored masks
    yellow_colored = np.zeros((scaled_height, scaled_width, 3), dtype=np.uint8)
    white_colored = np.zeros((scaled_height, scaled_width, 3), dtype=np.uint8)
    
    # Apply colors to masks
    yellow_colored[yellow_scaled > 0] = [0, 255, 255]  # BGR yellow
    white_colored[white_scaled > 0] = [255, 255, 255]  # BGR white
    
    # Combine the colored masks
    combined = cv2.addWeighted(yellow_colored, 0.5, white_colored, 0.5, 0)
    
    # Create grid overlay
    grid = np.zeros((scaled_height, scaled_width, 3), dtype=np.uint8)
    
    # Calculate pixels per centimeter after scaling
    pixels_per_cm = (scale_factor * width) / 100  # assuming 100cm total width
    
    # Origin point (scaled)
    origin_x = int(50 * scale_factor * width / 100)  # Scale the 100-pixel position
    origin_y = 0
    
    # Calculate grid spacing in pixels
    grid_spacing_px = int(grid_spacing_cm * pixels_per_cm)
    
    # Draw vertical lines
    for x in range(0, scaled_width, grid_spacing_px):
        cv2.line(grid, (x, 0), (x, scaled_height), (128, 128, 128), 2)
    
    # Draw horizontal lines
    for y in range(0, scaled_height, grid_spacing_px):
        cv2.line(grid, (0, y), (scaled_width, y), (128, 128, 128), 2)
    
    # Draw origin axes with different color
    cv2.line(grid, (origin_x, 0), (origin_x, scaled_height), (0, 0, 255), 2)  # Vertical origin line in red
    cv2.line(grid, (0, origin_y), (scaled_width, origin_y), (0, 0, 255), 2)   # Horizontal origin line in red
    
    # Add coordinate numbers
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6 * scale_factor / 4  # Scale font size appropriately
    font_thickness = max(2, int(scale_factor / 2))
    
    # Add x-axis numbers (in cm)
    for x in range(0, scaled_width, grid_spacing_px):
        coord_cm = int((x - origin_x) / pixels_per_cm)
        if x != origin_x:  # Skip origin to avoid cluttering
            cv2.putText(grid, f"{coord_cm}", 
                       (x - 20, 30), 
                       font, font_scale, (255, 255, 255), 
                       font_thickness)
    
    # Add y-axis numbers (in cm, increasing downward)
    for y in range(0, scaled_height, grid_spacing_px):
        coord_cm = int(y / pixels_per_cm)
        if y != origin_y:  # Skip origin to avoid cluttering
            cv2.putText(grid, f"{coord_cm}", 
                       (origin_x + 10, y + 20), 
                       font, font_scale, (255, 255, 255), 
                       font_thickness)
    
    # Mark origin point
    cv2.circle(grid, (origin_x, origin_y), 5, (0, 0, 255), -1)
    cv2.putText(grid, "(0,0)", (origin_x + 10, 30), 
                font, font_scale, (0, 0, 255), font_thickness)
    
    # Combine the grid with the mask
    result = cv2.addWeighted(combined, 1.0, grid, 0.5, 0)
    
    # Add legend with larger font
    legend_y = 60
    cv2.putText(result, "Yellow Line", (20, legend_y), 
                font, font_scale, (0, 255, 255), font_thickness)
    cv2.putText(result, "White Line", (20, legend_y + 40), 
                font, font_scale, (255, 255, 255), font_thickness)
    cv2.putText(result, "Grid (20cm)", (20, legend_y + 80), 
                font, font_scale, (128, 128, 128), font_thickness)
    cv2.putText(result, "Origin (100,0)", (20, legend_y + 120), 
                font, font_scale, (0, 0, 255), font_thickness)
    
    return result

def save_visualization_results(
    output_folder: str,
    combined_masks: List[np.ndarray],
    steering_results: Dict[str, Dict[str, List[float]]],
    distance_results: List[Tuple[float, float, float, float]]
) -> None:
    """Save all visualization results including GIF, plots, and comparisons."""
    # Save combined GIF
    save_gif(combined_masks, os.path.join(output_folder, "combined_mask.gif"))
    
    # Plot omega comparison
    plt.figure(figsize=(12, 8))
    for combined_name, results in steering_results.items():
        plt.plot(
            results["timestamps"],
            results["original_omegas"],
            label=f"{combined_name} (Original)",
            linestyle="--",
        )
        plt.plot(
            results["timestamps"],
            results["controlled_omegas"],
            label=f"{combined_name} (Controlled)",
        )
    plt.xlabel("Time (s)")
    plt.ylabel("Steering (Omega)")
    plt.title("Original vs Controlled Omega")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_folder, "omega_comparison.png"))
    plt.close()

    # Plot distance comparison
    if distance_results:
        timestamps, yellow_distances, white_distances, differences = zip(*distance_results)
        plt.figure(figsize=(12, 8))
        plt.plot(timestamps, yellow_distances, label="Yellow Distance")
        plt.plot(timestamps, white_distances, label="White Distance")
        plt.plot(timestamps, differences, label="Yellow - White Distance", linestyle="--")
        plt.xlabel("Time (s)")
        plt.ylabel("Distance")
        plt.title("Line Distances Over Time")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_folder, "distance_comparison.png"))
        plt.close()

def main():
    """Main entry point of the script."""
    save_file = "log_2"
    log_data_path = os.path.join(
        os.path.expanduser("~"), 
        "Desktop", 
        "albert", 
        "data_analysis",  # Fixed typo in data_analaysis
        "log_file",
        f"{save_file}.txt"
    )

    segments_list = extract_segments_list(log_data_path)  # Assuming this function exists
    steering_method_list = ['trajectory_error']
    controller_list = ['moving_average+decay', 'PID']
    
    plot_steering_results(segments_list, steering_method_list, controller_list, save_file)

if __name__ == "__main__":
    main()