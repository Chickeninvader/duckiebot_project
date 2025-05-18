#!/usr/bin/env python3
import cv2
import numpy as np
import copy
import os
import rospy
from duckietown.dtros import DTROS, NodeType, TopicType
from duckietown_msgs.msg import ObstacleImageDetectionList, ObstacleImageDetection, Rect, BoolStamped
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage
from nn_model.constants import IMAGE_SIZE
from nn_model.model import Wrapper
from solution.integration_activity import NUMBER_FRAMES_SKIPPED, filter_by_classes, filter_by_bboxes, filter_by_scores
from collections import deque


class ObjectDetectionNode(DTROS):
    def __init__(self, node_name):
        # Initialize the DTROS parent class
        super(ObjectDetectionNode, self).__init__(node_name=node_name, node_type=NodeType.PERCEPTION)
        self.initialized = False
        self.log("Initializing!")
        self.veh = os.environ['VEHICLE_NAME']

        # Construct publishers
        self.pub_detections_image = rospy.Publisher("~image/compressed", CompressedImage, queue_size=1,
                                                    dt_topic_type=TopicType.DEBUG)
        self.pub_detections_box_image = rospy.Publisher("~box_image/compressed", CompressedImage, queue_size=1,
                                                        dt_topic_type=TopicType.DEBUG)

        self.pub_detections_list = rospy.Publisher(
            f"/{self.veh}/lane_controller_node/detection_list",
            ObstacleImageDetectionList,
            queue_size=1,
            dt_topic_type=TopicType.DEBUG,
        )

        self.obstacle_caused_stop = rospy.Publisher(
            f"/{self.veh}/object_detection_node/pedestrian_caused_stop",
            BoolStamped,
            queue_size=1,
            dt_topic_type=TopicType.DEBUG,
        )

        self.obstacle_timeout = rospy.Publisher(
            f"/{self.veh}/object_detection_node/timeout",
            BoolStamped,
            queue_size=1,
            dt_topic_type=TopicType.DEBUG,
        )

        # Construct subscribers
        self.sub_image = rospy.Subscriber(
            f"/{self.veh}/camera_node/image/compressed",
            CompressedImage,
            self.image_cb,
            buff_size=10000000,
            queue_size=1,
        )

        self.bridge = CvBridge()
        aido_eval = rospy.get_param("~AIDO_eval", False)
        self._debug = rospy.get_param("~debug", True)

        self.log("Starting model loading!")
        self.model_wrapper = Wrapper(aido_eval)
        self.log("Finished model loading!")

        self.frame_id = 0
        self.detection_buffer = deque(maxlen=5)
        self.stop_start_time = None
        self.initialized = True
        self.last_detection_time = rospy.Time.now()
        self.log("Initialized!")

    def image_cb(self, image_msg):
        if not self.initialized:
            return

        self.frame_id += 1
        self.frame_id = self.frame_id % (1 + NUMBER_FRAMES_SKIPPED())
        if self.frame_id != 0:
            return

        try:
            bgr = self.bridge.compressed_imgmsg_to_cv2(image_msg)
        except ValueError as e:
            self.logerr(f"Could not decode image: {e}")
            return

        rgb = bgr[..., ::-1]
        rgb = cv2.resize(rgb, (IMAGE_SIZE, IMAGE_SIZE))
        bboxes, classes, scores = self.model_wrapper.predict(rgb)

        # Call publish_detections with the model outputs and get updated results
        updated_bboxes, updated_classes, updated_scores = self.publish_detections(bboxes, classes, scores, image_msg, rgb)

        # Use the updated values for visualization, including confidence scores
        self.visualize_detections(rgb, updated_bboxes, updated_classes, updated_scores)

    def get_hsi(self, bbox, rgb, brightness_threshold=200):
        """
        Extract HSV values from a region at the center of the traffic light bounding box.
        Only considers pixels with Value > 150 for more robust detection.
        """
        try:
            # Convert CompressedImage to OpenCV BGR format

            y1, y2 = int(bbox[1]), int(bbox[3])
            x1, x2 = int(bbox[0]), int(bbox[2])

            # Ensure coordinates are within image bounds
            height, width = rgb.shape[:2]
            y1 = max(0, min(y1, height - 1))
            y2 = max(0, min(y2, height - 1))
            x1 = max(0, min(x1, width - 1))
            x2 = max(0, min(x2, width - 1))

            # Extract the region
            roi = copy.deepcopy(rgb[y1:y2, x1:x2])

            # Convert RGB to BGR before publishing
            roi_bgr = roi[..., ::-1]
            obj_det_img_box = self.bridge.cv2_to_compressed_imgmsg(roi_bgr)
            self.pub_detections_box_image.publish(obj_det_img_box)

            # Convert ROI to HSV color space
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)

            # Create mask for pixels with V > 120
            mask = hsv_roi[:, :, 2] > brightness_threshold

            # If no pixels meet the brightness threshold, return error values
            if not np.any(mask):
                self.log("No bright pixels found in ROI")
                return -1, -1, -1

            # Calculate average HSV values only for bright pixels
            bright_pixels = hsv_roi[mask]

            # Convert hue to radians and use circular mean
            hues = bright_pixels[:, 0] * (2 * np.pi / 180)  # Convert to radians
            x = np.mean(np.cos(hues))
            y = np.mean(np.sin(hues))
            mean_hue = np.arctan2(y, x) * (180 / (2 * np.pi))  # Convert back to degrees
            mean_hue = (mean_hue + 360) % 360  # Ensure positive angle

            saturation = np.mean(bright_pixels[:, 1])
            intensity = np.mean(bright_pixels[:, 2])

            # Log both center pixel and filtered average values
            self.log(f"average HSV - Hue: {mean_hue:.1f}, "
                     f"Saturation: {saturation:.1f}, Value: {intensity:.1f}")

            return mean_hue, saturation, intensity

        except Exception as e:
            rospy.logerr(f"Error processing traffic light HSV: {e}")
            return -1, -1, -1

    def get_hsi_blue_yellow(self, bbox, rgb, brightness_threshold=200):
        """
        Extract HSV values from a region of a bounding box, optimized for yellow and blue detection.
        Uses median values for more robust color detection.
        """
        try:
            # Extract coordinates from bbox
            y1, y2 = int(bbox[1]), int(bbox[3])
            x1, x2 = int(bbox[0]), int(bbox[2])

            # Ensure coordinates are within image bounds
            height, width = rgb.shape[:2]
            y1 = max(0, min(y1, height - 1))
            y2 = max(0, min(y2, height - 1))
            x1 = max(0, min(x1, width - 1))
            x2 = max(0, min(x2, width - 1))

            # Extract the region
            roi = copy.deepcopy(rgb[y1:y2, x1:x2])

            # Convert RGB to BGR before publishing visualization
            roi_bgr = roi[..., ::-1]
            obj_det_img_box = self.bridge.cv2_to_compressed_imgmsg(roi_bgr)
            self.pub_detections_box_image.publish(obj_det_img_box)

            # Convert ROI to HSV color space
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)

            # Create mask for pixels with V > brightness_threshold
            mask = hsv_roi[:, :, 2] > brightness_threshold

            # If no pixels meet the brightness threshold, return error values
            if not np.any(mask):
                self.log("No bright pixels found in ROI")
                return -1, -1, -1

            # Get HSV values only for bright pixels
            bright_pixels = hsv_roi[mask]

            # For hue, we need to handle its circular nature
            hues = bright_pixels[:, 0]

            # Convert to Cartesian coordinates
            x_coords = np.cos(hues * (2 * np.pi / 180))
            y_coords = np.sin(hues * (2 * np.pi / 180))

            # Find the median direction
            median_x = np.median(x_coords)
            median_y = np.median(y_coords)

            # Convert back to angle in degrees
            median_hue = np.arctan2(median_y, median_x) * (180 / (2 * np.pi))
            median_hue = (median_hue + 360) % 360  # Ensure positive angle

            # Get median saturation and intensity
            median_saturation = np.median(bright_pixels[:, 1])
            median_intensity = np.median(bright_pixels[:, 2])

            # Calculate percentage of blue and yellow pixels
            # Blue range in HSV
            BLUE_HUE_RANGE = (200, 250)
            # Yellow range in HSV
            YELLOW_HUE_RANGE = (40, 65)

            blue_mask = (BLUE_HUE_RANGE[0] <= hues) & (hues <= BLUE_HUE_RANGE[1])
            yellow_mask = (YELLOW_HUE_RANGE[0] <= hues) & (hues <= YELLOW_HUE_RANGE[1])

            total_bright_pixels = len(hues)
            blue_percentage = np.sum(blue_mask) / total_bright_pixels * 100 if total_bright_pixels > 0 else 0
            yellow_percentage = np.sum(yellow_mask) / total_bright_pixels * 100 if total_bright_pixels > 0 else 0

            # Log color information
            self.log(f"\n*** Color Analysis ***\n"
                     f"median HSV - Hue: {median_hue:.1f}, "
                     f"Saturation: {median_saturation:.1f}, Value: {median_intensity:.1f}\n"
                     f"Blue pixels: {blue_percentage:.1f}%, Yellow pixels: {yellow_percentage:.1f}%\n"
                     f"Total bright pixels: {total_bright_pixels}\n"
                     f"**********************")

            return median_hue, median_saturation, median_intensity

        except Exception as e:
            rospy.logerr(f"Error processing HSV values: {e}")
            return -1, -1, -1

    def publish_detections(self, bboxes, classes, scores, image_msg, rgb):
        detection_list_msg = ObstacleImageDetectionList()
        detection_list_msg.header = image_msg.header
        detection_list_msg.imwidth = IMAGE_SIZE
        detection_list_msg.imheight = IMAGE_SIZE

        names = {0: "duckie", 1: "duckiebot", 2: "intersection_sign", 3: "traffic_light", 4: "stop_sign"}

        # Create copies of input arrays to avoid modifying originals
        updated_bboxes = bboxes.copy()
        updated_classes = classes.copy()
        updated_scores = scores.copy()

        #################################################################
        # add a empty object, as placeholder
        detection = ObstacleImageDetection()
        detection.bounding_box = Rect(
            x=0,
            y=0,
            w=0,
            h=0
        )
        detection.type.type = 200
        detection_list_msg.list.append(detection)
        #################################################################

        valid_detections = []  # To store indices of valid detections

        for i, (bbox, cls, score) in enumerate(zip(bboxes, classes, scores)):
            cls = int(cls)
            if not isinstance(cls, int) or not 0 <= cls <= 4:
                continue

            if (cls in [0, 1] and score <= 0.5) or (cls not in [0, 1] and score <= 0.15):
                continue

            # Traffic Light Processing
            if cls == 3 and score >= 0.5:
                hue, saturation, intensity = self.get_hsi(bbox, rgb)

                ####################################
                self.log("\n********************************\n" +
                         f"score: {score}\n" +
                         f"hue: {hue:.1f}\n" +
                         "\n************************************\n", "debug")
                ######################################

                RED_HUE_RANGE_1 = (0, 30)
                # RED_HUE_RANGE_2 = (160, 180)
                RED_HUE_RANGE_2 = (330, 360)
                GREEN_HUE_RANGE = (40, 100)

                if ((RED_HUE_RANGE_1[0] <= hue <= RED_HUE_RANGE_1[1] or
                     RED_HUE_RANGE_2[0] <= hue <= RED_HUE_RANGE_2[1])):
                    cls = 30  # Red light
                    updated_classes[i] = cls  # Update class in the copy
                    self.log(f"detect red traffic light!")
                elif GREEN_HUE_RANGE[0] <= hue <= GREEN_HUE_RANGE[1]:
                    cls = 31  # Green light
                    updated_classes[i] = cls  # Update class in the copy
                    self.log(f"detect green traffic light!")

            ###################################################################
            # Duckiebot Processing
            elif (cls == 0 or cls == 1) and score >= 0.6:
                hue, saturation, intensity = self.get_hsi_blue_yellow(bbox, rgb, brightness_threshold=0)

                # Define color ranges specifically for duckie object detection
                YELLOW_HUE_RANGE = (10, 30)  # Pedestrian (Duck)
                BLUE_HUE_RANGE = (290, 310)  # Vehicle

                # More specific detection based on color
                if YELLOW_HUE_RANGE[0] <= hue <= YELLOW_HUE_RANGE[1]:
                    cls = 10  # Pedestrian duck (yellow)
                    updated_classes[i] = cls
                    self.log(f"detect pedestrian (duck) with yellow color!")
                elif BLUE_HUE_RANGE[0] <= hue <= BLUE_HUE_RANGE[1]:
                    cls = 11  # Vehicle (blue)
                    updated_classes[i] = cls
                    self.log(f"detect vehicle with blue color!")
                else:
                    cls = 12  # Other duckiebot (undetermined color)
                    updated_classes[i] = cls
                    self.log(f"detect duckiebot with undetermined color")

                yy1, yy2 = int(bbox[1]), int(bbox[3])
                xx1, xx2 = int(bbox[0]), int(bbox[2])
                duckiebot_area = (xx2 - xx1) * (yy2 - yy1)
                updated_classes[i] = cls  # Update class in the copy

                # Compute center of bounding box
                center_x = (xx1 + xx2) // 2
                center_y = (yy1 + yy2) // 2

                # Define stop region
                REGION_X_MIN = IMAGE_SIZE / 4
                REGION_X_MAX = IMAGE_SIZE / 4 * 3
                REGION_Y_MIN = IMAGE_SIZE / 2
                REGION_Y_MAX = IMAGE_SIZE

                AREA_THRESHOLD = 1500

                current_time = rospy.Time.now()

                # --- Detection Validity Check ---
                inside_stop_region = (REGION_X_MIN <= center_x <= REGION_X_MAX and
                                      REGION_Y_MIN <= center_y <= REGION_Y_MAX)
                is_valid = inside_stop_region and duckiebot_area > AREA_THRESHOLD

                # --- Update Detection Buffer ---
                if is_valid:
                    self.last_detection_time = current_time
                    self.detection_buffer.append(True)
                else:
                    self.detection_buffer.append(False)

                # --- Reset buffer if no valid detection in last 5s ---
                if (current_time - self.last_detection_time).to_sec() > 5.0:
                    self.detection_buffer = deque([False] * len(self.detection_buffer), maxlen=len(self.detection_buffer))

                count_true = sum(self.detection_buffer)
                should_stop = count_true >= 3

                # === Timeout logic ===
                if should_stop:
                    if self.stop_start_time is None:
                        self.stop_start_time = current_time
                    elif (current_time - self.stop_start_time).to_sec() >= 5.0:
                        timeout_msg = BoolStamped()
                        timeout_msg.data = True
                        timeout_msg.header = image_msg.header
                        for _ in range(4):
                            self.obstacle_timeout.publish(timeout_msg)
                else:
                    self.stop_start_time = None  # reset if not stopping

                # --- Publish Stop Decision ---
                msg = BoolStamped()
                msg.data = should_stop
                msg.header = image_msg.header
                for _ in range(4):
                    self.obstacle_caused_stop.publish(msg)

                # --- Debug ---
                self.log("\n************************************\n" +
                         f"detect duckiebot and confident\n" +
                         f"score: {score}\n" +
                         f"duckiebot area: {duckiebot_area}\n" +
                         f"center: ({center_x}, {center_y})\n" +
                         f"inside_stop_region: {inside_stop_region}\n" +
                         f"is_valid: {is_valid}, buffer: {list(self.detection_buffer)}\n" +
                         f"should_stop: {should_stop}\n" +
                         ("OBSTACLE TIMEOUT TRIGGERED\n" if should_stop and self.stop_start_time and
                          (current_time - self.stop_start_time).to_sec() >= 5.0 else "") +
                         "************************************\n")

            else:
                self.log(f"detect {cls} {names[cls]} with score {score:.2f}!\n")

            detection = ObstacleImageDetection()
            detection.bounding_box = Rect(
                x=int(bbox[0]),
                y=int(bbox[1]),
                w=int(bbox[2] - bbox[0]),
                h=int(bbox[3] - bbox[1])
            )
            detection.type.type = int(cls)
            detection_list_msg.list.append(detection)

            # Save valid detection index
            valid_detections.append(i)

        self.pub_detections_list.publish(detection_list_msg)

        # Filter valid detections
        filtered_bboxes = np.array([updated_bboxes[i] for i in valid_detections])
        filtered_classes = np.array([updated_classes[i] for i in valid_detections])
        filtered_scores = np.array([updated_scores[i] for i in valid_detections])

        # Return updated values for visualization, including scores
        return filtered_bboxes, filtered_classes, filtered_scores

    def visualize_detections(self, rgb, bboxes, classes, scores=None):
        colors = {0: (0, 255, 255),  # duckie (yellow)
                  1: (0, 165, 255),  # duckiebot (orange)
                  2: (0, 250, 0),  # intersection_sign (green)
                  3: (0, 0, 255),  # traffic_light (red)
                  4: (255, 0, 0),  # stop_sign (blue)
                  10: (0, 255, 255),  # pedestrian duck (yellow)
                  11: (255, 127, 0),  # vehicle (blue)
                  12: (192, 192, 192),  # other duckiebot (gray)
                  30: (0, 0, 255),  # red_light (red)
                  31: (0, 250, 0)}  # green_light (green)

        names = {0: "duckie",
                 1: "duckiebot",
                 2: "intersection_sign",
                 3: "traffic_light",
                 4: "stop_sign",
                 10: "pedestrian",
                 11: "vehicle",
                 12: "duckiebot",
                 30: "red_light",
                 31: "green_light"}
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Make a copy of rgb to avoid modifying the original
        vis_img = rgb.copy()

        for i, (clas, box) in enumerate(zip(classes, bboxes)):
            clas = int(clas)  # Ensure class is an integer
            if clas not in colors:
                continue  # Skip if class is not in our colors dictionary

            pt1 = tuple(map(int, box[:2]))
            pt2 = tuple(map(int, box[2:]))
            color = tuple(reversed(colors[clas]))
            name = names.get(clas, f"unknown_{clas}")

            # Add score to display text if available
            if scores is not None and i < len(scores):
                score = scores[i]
                display_text = f"{name}: {score:.2f}"
            else:
                display_text = name

            # Draw bounding box
            vis_img = cv2.rectangle(vis_img, pt1, pt2, color, 2)

            # Draw text with score
            text_location = (pt1[0], min(pt2[1] + 30, IMAGE_SIZE))
            vis_img = cv2.putText(vis_img, display_text, text_location, font, 0.8, color, thickness=2)

        bgr = vis_img[..., ::-1]
        obj_det_img = self.bridge.cv2_to_compressed_imgmsg(bgr)
        self.pub_detections_image.publish(obj_det_img)


if __name__ == "__main__":
    object_detection_node = ObjectDetectionNode(node_name="object_detection_node")
    rospy.spin()
