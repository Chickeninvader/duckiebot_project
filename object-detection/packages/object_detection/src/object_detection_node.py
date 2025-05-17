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
        self.pub_detections_image = rospy.Publisher("~image/compressed", CompressedImage, queue_size=1, dt_topic_type=TopicType.DEBUG)
        self.pub_detections_box_image = rospy.Publisher("~box_image/compressed", CompressedImage, queue_size=1, dt_topic_type=TopicType.DEBUG)
        
        self.pub_detections_list = rospy.Publisher(
            f"/{self.veh}/lane_controller_node/detection_list",
            ObstacleImageDetectionList,
            queue_size=1,
            dt_topic_type=TopicType.DEBUG,
        )

        self.obstacle_caused_stop = rospy.Publisher(
            f"/{self.veh}/vehicle_detection_node/obstacle_caused_stop",
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
        self.initialized = True
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

        # Publish detection results
        self.publish_detections(bboxes, classes, scores, image_msg, rgb)
        
        # if self._debug:
        #     self.visualize_detections(rgb, bboxes, classes)
        self.visualize_detections(rgb, bboxes, classes)

    def get_traffic_light_hsi(self, bbox, rgb):
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
            y1 = max(0, min(y1, height-1))
            y2 = max(0, min(y2, height-1))
            x1 = max(0, min(x1, width-1))
            x2 = max(0, min(x2, width-1))
            
            # Extract the region
            roi = copy.deepcopy(rgb[y1:y2, x1:x2])

            # Convert RGB to BGR before publishing
            roi_bgr = roi[..., ::-1]
            obj_det_img_box = self.bridge.cv2_to_compressed_imgmsg(roi_bgr)
            self.pub_detections_box_image.publish(obj_det_img_box)
            
            # Convert ROI to HSV color space
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
            
            # Create mask for pixels with V > 120
            mask = hsv_roi[:, :, 2] > 200
            
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
            self.log(f"Traffic light filtered average HSV - Hue: {mean_hue:.1f}, "
                    f"Saturation: {saturation:.1f}, Value: {intensity:.1f}")
                    
            return mean_hue, saturation, intensity
            
        except Exception as e:
            rospy.logerr(f"Error processing traffic light HSV: {e}")
            return -1, -1, -1

    def publish_detections(self, bboxes, classes, scores, image_msg, rgb):
        detection_list_msg = ObstacleImageDetectionList()
        detection_list_msg.header = image_msg.header
        detection_list_msg.imwidth = IMAGE_SIZE
        detection_list_msg.imheight = IMAGE_SIZE

        names = {0: "duckie", 1: "duckiebot", 2: "intersection_sign", 3: "traffic_light", 4: "stop_sign"}

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

        ###########################
        idx = 0
        ###########################

        for bbox, cls, score in zip(bboxes, classes, scores):
            cls = int(cls)
            if not isinstance(cls, int) or not 0 <= cls <= 4:
                continue
                
            if (cls in [0, 1] and score <= 0.5) or (cls not in [0, 1] and score <= 0.15):
                continue

            # Traffic Light Processing
            if cls == 3 and score >= 0.5:
                hue, saturation, intensity = self.get_traffic_light_hsi(bbox, rgb)

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
                    self.log(f"detect red traffic light!")
                elif GREEN_HUE_RANGE[0] <= hue <= GREEN_HUE_RANGE[1]:
                    cls = 31  # Green light
                    self.log(f"detect green traffic light!")

            ###################################################################
            # Duckiebot Processing
            elif cls == 1 and score >= 0.5:
                yy1, yy2 = int(bbox[1]), int(bbox[3])
                xx1, xx2 = int(bbox[0]), int(bbox[2])
                duckiebot_area = (xx2 - xx1) * (yy2 - yy1)
                cls = 11  # confident duckiebot detected

                # Compute center of bounding box
                center_x = (xx1 + xx2) // 2
                center_y = (yy1 + yy2) // 2

                # Define your stop region (e.g., central lower part of the image)
                REGION_X_MIN = 100
                REGION_X_MAX = 220
                REGION_Y_MIN = 160
                REGION_Y_MAX = 240
                AREA_THRESHOLD = 3000  # example value

                # Check if bounding box is within stop region and large enough
                inside_stop_region = (REGION_X_MIN <= center_x <= REGION_X_MAX and
                                      REGION_Y_MIN <= center_y <= REGION_Y_MAX)
                is_valid = inside_stop_region and duckiebot_area > AREA_THRESHOLD

                # Add result to buffer
                self.detection_buffer.append(is_valid)

                # Voting logic: if >= 3 out of 5 are True, then publish stop
                count_true = sum(self.detection_buffer)
                should_stop = count_true >= 3

                # Publish decision message
                msg = BoolStamped()
                msg.data = should_stop
                msg.header = image_msg.header
                self.obstacle_caused_stop.publish(msg)

                # Optional debug log
                self.log("\n************************************\n" +
                         f"detect duckiebot and confident\n" +
                         f"score: {score}\n" +
                         f"duckiebot area: {duckiebot_area}\n" +
                         f"center: ({center_x}, {center_y})\n" +
                         f"inside_stop_region: {inside_stop_region}\n" +
                         f"is_valid: {is_valid}, buffer: {list(self.detection_buffer)}\n" +
                         f"stop published: {should_stop}" +
                         "\n************************************\n", "debug")
            else:
                self.log(f"detect {names[cls]}")

            detection = ObstacleImageDetection()
            detection.bounding_box = Rect(
                x=int(bbox[0]),
                y=int(bbox[1]),
                w=int(bbox[2] - bbox[0]),
                h=int(bbox[3] - bbox[1])
            )
            detection.type.type = int(cls)
            detection_list_msg.list.append(detection)

            ##############################################
            classes[idx] = int(cls)
            idx += 1
            ##############################################

        self.pub_detections_list.publish(detection_list_msg)

    def visualize_detections(self, rgb, bboxes, classes):
        colors = {0: (0, 255, 255), 1: (0, 165, 255), 2: (0, 250, 0), 
                 3: (0, 0, 255), 4: (255, 0, 0), 11: (0, 165, 255), 30: (0, 0, 255), 31: (0, 250, 0)}
        names = {0: "duckie", 1: "duckiebot", 2: "intersection_sign", 
                3: "traffic_light", 4: "stop_sign", 11: "duckiebot", 30: "red_light", 31: "green_light" }
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        for clas, box in zip(classes, bboxes):
            pt1 = tuple(map(int, box[:2]))
            pt2 = tuple(map(int, box[2:]))
            color = tuple(reversed(colors[clas]))
            name = names[clas]
            rgb = cv2.rectangle(rgb, pt1, pt2, color, 2)
            text_location = (pt1[0], min(pt2[1] + 30, IMAGE_SIZE))
            rgb = cv2.putText(rgb, name, text_location, font, 1, color, thickness=2)
            
        bgr = rgb[..., ::-1]
        obj_det_img = self.bridge.cv2_to_compressed_imgmsg(bgr)
        self.pub_detections_image.publish(obj_det_img)

if __name__ == "__main__":
    object_detection_node = ObjectDetectionNode(node_name="object_detection_node")
    rospy.spin()
