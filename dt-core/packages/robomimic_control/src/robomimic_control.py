#!/usr/bin/env python3
import glob
import os
import rospy
import cv2
import torch
import numpy as np
from collections import deque
from cv_bridge import CvBridge
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage
from duckietown_msgs.msg import Twist2DStamped
from robomimic.utils import file_utils as FileUtils
from robomimic.utils import torch_utils as TorchUtils
import time
import torchvision.transforms.v2 as transforms
from torchvision.models import ResNet18_Weights


def find_latest_model_weights(base_weights_dir, method=None, run_type=None, subset=False):
    """
    Intelligently find the latest model weights based on the folder structure.

    Args:
        base_weights_dir: Base directory for weights
        method: Optional method name (e.g., "bc", "diffusion")
        run_type: Optional run type filter (e.g., "sim", "lab", "sim_lab")
        subset: Optional subset filter (e.g., "subset", "no")

    Returns:
        Path to the latest model weights file
    """
    # Step 1: Find method folder
    if method:
        method_dir = os.path.join(base_weights_dir, f"{method}_trained_models")
    else:
        # Find first method folder if not specified
        method_dirs = glob.glob(os.path.join(base_weights_dir, "*_trained_models"))
        if not method_dirs:
            raise FileNotFoundError(f"No method folders found in {base_weights_dir}")
        method_dir = method_dirs[0]

    # Step 2: Find training run folders
    run_pattern = "training_run_"
    if run_type:
        run_pattern += f"*{run_type}*"

    if subset:
        run_pattern += f"*nonhuman_subset*"
    else:
        run_pattern += f"*nonhuman_*"

    run_dirs = glob.glob(os.path.join(method_dir, run_pattern + "*"))
    if not run_dirs:
        raise FileNotFoundError(f"No matching training run folders found in {method_dir}")

    # Sort by creation time (newest first) as a fallback
    run_dirs.sort(key=os.path.getctime, reverse=True)
    run_dir = run_dirs[0]

    # Step 3: Find random number folder (take the first one if multiple exist)
    random_dirs = glob.glob(os.path.join(run_dir, "*"))
    if not random_dirs:
        raise FileNotFoundError(f"No random number folders found in {run_dir}")
    random_dir = random_dirs[0]

    # Step 4: Find models folder
    models_dir = os.path.join(random_dir, "models")
    if not os.path.exists(models_dir):
        raise FileNotFoundError(f"Models directory not found in {random_dir}")

    # Step 5: Find the highest epoch model
    model_files = glob.glob(os.path.join(models_dir, "model_epoch_*.pth"))
    if not model_files:
        raise FileNotFoundError(f"No model files found in {models_dir}")

    # Extract epoch numbers and find the highest
    epoch_numbers = []
    for model_file in model_files:
        try:
            epoch = int(os.path.basename(model_file).split("_")[-1].split(".")[0])
            epoch_numbers.append((epoch, model_file))
        except (ValueError, IndexError):
            continue

    if not epoch_numbers:
        raise ValueError(f"Could not parse epoch numbers from model files in {models_dir}")

    # Sort by epoch number (highest first)
    epoch_numbers.sort(reverse=True)
    return epoch_numbers[0][1]


class InferenceNode(DTROS):
    def __init__(self, node_name):
        super(InferenceNode, self).__init__(node_name=node_name, node_type=NodeType.PERCEPTION)
        # Load constants
        self.OMEGA_MAX = 8.0
        self.GAIN = 1.0
        self.transformers = False
        self.pretrained_resnet = True
        self.method = "iql"  # Set to "bc", "bc_rnn", etc. if known
        self.run_type = "sim_lab"  # Set to "sim", "lab", or "sim_lab" if known
        self.subset = False  # Set to True for "subset" if use sim_lab as runtype
        self.publish_velocity = False  # Set to False for testing without publishing commands

        # Manually defined parameters
        self.context_length = 1  # Default to 1 if not specified
        self.batch_size = 1  # Default to 4 if not specified

        # Configure frame history settings
        self.frame_history = deque(maxlen=self.batch_size)
        self.frame_skip = 1  # Process every frame (set higher to skip frames)
        self.curr_frame_count = 0

        # Inference timing
        self.inference_times = []
        self.avg_inference_time = 0

        # Load environment variables
        self.vehicle_name = os.getenv("VEHICLE_NAME", "default_vehicle")
        self.camera_topic = f"/{self.vehicle_name}/camera_node/image/compressed"
        self.cmd_topic = f"/{self.vehicle_name}/car_cmd_switch_node/cmd"
        self.current_directory = os.getenv("DT_REPO_PATH", os.getcwd())

        # Load the policy
        weights_dir = os.path.join(self.current_directory, "packages/robomimic_control/weights")
        try:
            self.ckpt_path = find_latest_model_weights(
                weights_dir,
                method=self.method,  # Set to specific method like "bc" if known
                run_type=self.run_type,  # Set to "sim", "lab", or "sim_lab" if preferred
                subset=self.subset  # Set to True for "subset", False for "no", or None for any
            )
            self.log(f"Using model weights: {self.ckpt_path}")
        except (FileNotFoundError, ValueError) as e:
            self.log(f"Error finding model weights: {e}")
            exit(1)

        self.device = TorchUtils.get_torch_device(try_to_use_cuda=True)
        self.policy, _ = FileUtils.policy_from_checkpoint(ckpt_path=self.ckpt_path, device=self.device, verbose=True)

        # Initialize OpenCV bridge
        self.bridge = CvBridge()

        # ROS Subscribers & Publishers
        self.sub_camera = rospy.Subscriber(self.camera_topic, CompressedImage, self.process_image, queue_size=1)
        self.pub_cmd = rospy.Publisher(self.cmd_topic, Twist2DStamped, queue_size=1)

        self.log(
            f"InferenceNode initialized with context length {self.context_length} and batch size {self.batch_size}")

    def process_image(self, msg):
        """Processes an image from the camera, runs inference, and sends velocity commands."""
        try:
            # Measure inference time
            start_time = time.time()

            # Count frames for potential skipping
            self.curr_frame_count += 1
            if self.frame_skip > 0 and (self.curr_frame_count % (self.frame_skip + 1) != 0):
                return

            # Convert image message to OpenCV format
            np_arr = np.frombuffer(msg.data, np.uint8)
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            # Preprocess image
            image = self.preprocess_image(image)  # Uncomment if need  preprocessing

            self.log(f"Processed frame {self.curr_frame_count}: shape {image.shape}")

            # Add frame to history
            self.frame_history.append(image)

            # Fill history with duplicates if not enough frames yet
            # if len(self.frame_history) < self.context_length:
            #     self.log(f"Building frame history: {len(self.frame_history)}/{self.context_length}")
            #     # Duplicate the first frame until we have enough history
            #     while len(self.frame_history) < self.context_length:
            #         self.frame_history.appendleft(processed_frame.clone())
            #     self.log("Frame history filled")

            # Create observation dict with stacked frames in B, T, C, H, W format
            obs_dict = self.prepare_observation()

            # Get action from policy
            action = self.get_policy_action(obs_dict)

            # Publish velocity command
            self.publish_velocity_command(action)

            end_time = time.time()
            # Track inference time
            inference_time = (end_time - start_time) * 1000  # ms

            self.inference_times.append(inference_time)
            if len(self.inference_times) > 30:  # Keep a moving window
                self.inference_times.pop(0)
            self.avg_inference_time = np.mean(self.inference_times)

        except Exception as e:
            self.log(f"Error in process_image: {e}", type="error")

    def preprocess_image(self, image):
        """Preprocesses a single image frame."""
        # # If using a pretrained ResNet, apply ResNet-specific preprocessing
        # if self.pretrained_resnet:
        #     transform = ResNet18_Weights.IMAGENET1K_V1.transforms()

        #     # Convert OpenCV image (NumPy) to PyTorch tensor
        #     image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)  # Convert HWC to CHW
        #     image = transform(image)  # Apply ResNet18 preprocessing
        #     return image.to(self.device)

        image = cv2.resize(image, (640, 480))  # Resize to model input size
        image = image.astype(np.float32) / 255.0  # Normalize
        image = np.transpose(image, (2, 0, 1))  # Convert HWC to CHW
        return torch.tensor(image, dtype=torch.float32).to(self.device)

    def prepare_observation(self):
        """Creates the observation dictionary with stacked frames in B, T, C, H, W format."""
        frames = list(self.frame_history)

        # Stack frames along time dimension first
        # Then expand to create batch dimension, only when transformer is True
        if self.batch_size != 1:
            batch_frames = torch.stack(frames).unsqueeze(1) if self.transformers else torch.stack(
                frames)  # [B, T, C, H, W]

        else:
            batch_frames = frames[0].unsqueeze(0).unsqueeze(0) if self.transformers else frames[0].unsqueeze(0)
        if self.context_length > 1:
            raise NotImplementedError('context length > 1 not implemented yet')

        return {"observation": batch_frames}

    def get_policy_action(self, obs_dict):
        """Runs inference and returns the action."""
        with torch.no_grad():
            try:
                # self.log(f"Input shape: {obs_dict['observation'].shape}")
                action = self.policy.policy.get_action(obs_dict)

                return action.cpu().numpy().flatten()

            except Exception as e:
                import traceback
                self.log(f"Error during inference: {traceback.format_exc()}", type="error")
                self.log(f"Observation shape: {obs_dict['observation'].shape}", type="error")
                # Fallback to zero action
                return np.array([0.0, 0.0])

    def publish_velocity_command(self, action):
        """Publishes the action as a velocity command."""
        cmd_msg = Twist2DStamped()
        # cmd_msg.v = float(action[0] * self.GAIN)  # Linear velocity
        cmd_msg.v = float(0.19)  # Linear velocity
        cmd_msg.omega = float(action[1] * self.OMEGA_MAX)  # Angular velocity
        self.log(f"Velocity: {cmd_msg.v:.3f}, Omega: {cmd_msg.omega:.3f} (Inference: {self.avg_inference_time:.1f}ms)")
        if self.publish_velocity:
            self.pub_cmd.publish(cmd_msg)  # Uncomment for actual control

    def on_shutdown(self):
        """Stops the robot on shutdown."""
        stop_cmd = Twist2DStamped(v=0, omega=0)
        self.pub_cmd.publish(stop_cmd)
        self.log("Shutting down, stopping robot")


if __name__ == "__main__":
    node = InferenceNode(node_name="inference_node")
    rospy.spin()
