import h5py
import cv2
import numpy as np
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Visualize a specific demo from an HDF5 dataset.")
parser.add_argument("hdf5_path", type=str, help="Path to the HDF5 dataset.")
parser.add_argument("demo_number", type=int, default=1, help="Demo number to visualize (e.g., 1 for demo_1).")
parser.add_argument("--fps", type=int, default=10, help="Frames per second for playback.")
args = parser.parse_args()

# Open the HDF5 file
with h5py.File(args.hdf5_path, "r") as f:
    # Access the demo key with the proper prefix "data/"
    demo_key = f"data/demo_{args.demo_number}"

    if demo_key not in f:
        print(f"Demo {args.demo_number} not found in dataset.")
        print(f"Available groups: {list(f.keys())}")
        exit(1)

    demo_grp = f[demo_key]  # Access the demo group

    # Load images (assuming the shape is (N, H, W, C))
    images = np.array(demo_grp["obs/observation"])

    # Load actions (assuming the shape is (N, 2))
    actions = np.array(demo_grp["actions"])

# Playback visualization
print(f"Playing demo_{args.demo_number} ({len(images)} frames)...")

for i in range(len(images)):
    frame = images[i].copy()  # Get the current frame
    action = actions[i] if i < len(actions) else [0.0, 0.0]  # Get corresponding action

    # Overlay text with action values
    text = f"Action: Left {action[0]:.2f}, Right {action[1]:.2f}"
    cv2.putText(frame, text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow(f"Demo {args.demo_number}", frame)

    if cv2.waitKey(int(1000 / args.fps)) & 0xFF == ord("q"):
        break  # Press 'q' to quit playback early

cv2.destroyAllWindows()
