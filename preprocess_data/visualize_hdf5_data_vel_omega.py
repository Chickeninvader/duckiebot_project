import h5py
import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt


def plot_rewards(demo_group):
    """Create a plot of rewards for the demo."""
    # Extract reward information
    rewards = np.array(demo_group.get("rewards", []))

    if len(rewards) == 0:
        print("No reward data available to plot")
        return

    # Create a figure
    plt.figure(figsize=(10, 5))
    plt.title('Demo Rewards')

    # Plot rewards
    plt.plot(rewards, label='Rewards', color='green')
    plt.xlabel('Timestep')
    plt.ylabel('Reward')
    plt.grid(True)
    plt.legend()

    # Calculate and display statistics
    avg_reward = np.mean(rewards)
    total_reward = np.sum(rewards)

    plt.figtext(0.02, 0.02, f"Average Reward: {avg_reward:.3f}\nTotal Reward: {total_reward:.3f}",
                fontsize=9, bbox=dict(facecolor='white', alpha=0.5))

    plt.tight_layout()
    plt.show()


def visualize_demo(hdf5_path, demo_number, fps=10):
    """Visualize a specific demo from an HDF5 dataset."""
    # Open the HDF5 file
    with h5py.File(hdf5_path, "r") as f:
        # Access the demo key with the proper prefix "data/"
        demo_key = f"data/demo_{demo_number}"

        if demo_key not in f:
            print(f"Demo {demo_number} not found in dataset.")
            print(f"Available groups: {list(f.keys())}")
            return

        demo_grp = f[demo_key]  # Access the demo group

        # Print demo attributes if available
        print("\nDemo attributes:")
        for attr_name, attr_value in demo_grp.attrs.items():
            print(f"  {attr_name}: {attr_value}")

        # Load images
        images = np.array(demo_grp["obs/observation"])

        # Load actions
        actions = np.array(demo_grp["actions"])

        # Load rewards (if available)
        try:
            rewards = np.array(demo_grp["rewards"])
        except Exception as e:
            print(f"Warning: Couldn't load rewards: {e}")
            rewards = np.zeros(len(images))

        # Load dones (if available)
        try:
            dones = np.array(demo_grp["dones"])
        except Exception:
            dones = np.zeros(len(images))
            if len(dones) > 0:
                dones[-1] = 1  # Mark last frame as done

    # Plot rewards
    plot_rewards(demo_grp)

    # Playback visualization
    print(f"Playing demo_{demo_number} ({len(images)} frames)...")

    # Calculate cumulative reward for tracking
    cumulative_reward = 0

    for i in range(len(images)):
        frame = images[i].copy()  # Get the current frame
        action = actions[i] if i < len(actions) else [0.0, 0.0]  # Get corresponding action
        reward = rewards[i] if i < len(rewards) else 0
        done = dones[i] if i < len(dones) else 0

        # Update cumulative reward
        cumulative_reward += reward

        # Prepare overlay information
        info_lines = [
            f"Frame: {i + 1}/{len(images)}",
            f"Velocity: {action[0]:.2f}",
            f"Omega: {action[1]:.2f}",
            f"Reward: {reward:.3f}",
            f"Cumulative Reward: {cumulative_reward:.3f}"
        ]

        if done:
            info_lines.append("DONE")

        # Add colored background for the text for better visibility
        text_bg = frame.copy()
        cv2.rectangle(text_bg, (10, 10), (300, 30 + len(info_lines) * 30), (0, 0, 0), -1)
        # Blend the background with the original frame
        alpha = 0.7
        frame = cv2.addWeighted(text_bg, alpha, frame, 1 - alpha, 0)

        # Overlay text with multiple lines
        for j, line in enumerate(info_lines):
            cv2.putText(frame, line, (20, 30 + j * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0), 2)

        cv2.imshow(f"Demo {demo_number}", frame)

        # Wait for key press, quit if 'q' is pressed
        key = cv2.waitKey(int(1000 / fps)) & 0xFF
        if key == ord("q"):
            break  # Press 'q' to quit playback early
        elif key == ord(" "):
            # Pause/resume on spacebar
            cv2.waitKey(0)

    cv2.destroyAllWindows()


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Visualize a specific demo from an HDF5 dataset.")
    parser.add_argument("hdf5_path", type=str, nargs="?",
                        default="converted_standard/all_demos_new.hdf5",
                        help="Path to the HDF5 dataset.")
    parser.add_argument("demo_number", type=int, nargs="?",
                        default=1,
                        help="Demo number to visualize (e.g., 1 for demo_1).")
    parser.add_argument("--fps", type=int, default=10,
                        help="Frames per second for playback.")
    args = parser.parse_args()

    # Visualize the specified demo
    visualize_demo(args.hdf5_path, args.demo_number, args.fps)


if __name__ == "__main__":
    main()
