#!/usr/bin/env python3
import cv2
import numpy as np
import rospy
from duckietown.dtros import DTROS, NodeType, TopicType
from duckietown_msgs.msg import Twist2DStamped, EpisodeStart, ObstacleImageDetectionList, ObstacleImageDetection, Rect
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage
from nn_model.constants import IMAGE_SIZE
from nn_model.model import Wrapper
from solution.integration_activity import NUMBER_FRAMES_SKIPPED, filter_by_classes, filter_by_bboxes, filter_by_scores
class ObjectDetectionNode(DTROS):
    def __init__(self, node_name):
        # Initialize the DTROS parent class
        super(ObjectDetectionNode, self).__init__(node_name=node_name, node_type=NodeType.PERCEPTION)
        self.initialized = False
        self.log("Initializing!")
        self.veh = "chicinvabot"
        self.avoid_duckies = False
        # Construct publishers
        car_cmd_topic = f"/{self.veh}/joy_mapper_node/car_cmd"
        self.pub_car_cmd = rospy.Publisher(car_cmd_topic, Twist2DStamped, queue_size=1, dt_topic_type=TopicType.CONTROL)
        episode_start_topic = f"/{self.veh}/episode_start"
        rospy.Subscriber(episode_start_topic, EpisodeStart, self.cb_episode_start, queue_size=1)
        self.pub_detections_image = rospy.Publisher("~image/compressed", CompressedImage, queue_size=1, dt_topic_type=TopicType.DEBUG)
        # New publisher for detection results
        self.pub_detections_list = rospy.Publisher(
            f"/{self.veh}/detection_list",
            ObstacleImageDetectionList,
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
        self.v = rospy.get_param("~speed", 0.4)
        aido_eval = rospy.get_param("~AIDO_eval", False)
        self.log(f"AIDO EVAL VAR: {aido_eval}")
        self.log("Starting model loading!")
        self._debug = rospy.get_param("~debug", False)
        self.model_wrapper = Wrapper(aido_eval)
        self.log("Finished model loading!")
        self.frame_id = 0
        self.first_image_received = False
        self.initialized = True
        self.log("Initialized!")
    def cb_episode_start(self, msg: EpisodeStart):
        self.avoid_duckies = False
        # self.pub_car_commands(True, msg.header)
    def image_cb(self, image_msg):
        if not self.initialized:
            # self.pub_car_commands(True, image_msg.header)
            return
        self.frame_id += 1
        self.frame_id = self.frame_id % (1 + NUMBER_FRAMES_SKIPPED())
        if self.frame_id != 0:
            # self.pub_car_commands(self.avoid_duckies, image_msg.header)
            return
        try:
            bgr = self.bridge.compressed_imgmsg_to_cv2(image_msg)
        except ValueError as e:
            self.logerr(f"Could not decode image: {e}")
            return
        rgb = bgr[..., ::-1]
        rgb = cv2.resize(rgb, (IMAGE_SIZE, IMAGE_SIZE))
        bboxes, classes, scores = self.model_wrapper.predict(rgb)
        detection = self.det2bool(bboxes, classes, scores)
        if detection:
            self.log("Duckie pedestrian detected... stopping")
            self.avoid_duckies = True
        # self.pub_car_commands(self.avoid_duckies, image_msg.header)
        # Publish detection results
        self.publish_detections(bboxes, classes, scores, image_msg)
        if self._debug:
            self.visualize_detections(rgb, bboxes, classes)
    def publish_detections(self, bboxes, classes, scores, image_msg):
        # Initialize the detection list message
        detection_list_msg = ObstacleImageDetectionList()
        detection_list_msg.header = image_msg.header
        detection_list_msg.imwidth = IMAGE_SIZE
        detection_list_msg.imheight = IMAGE_SIZE
        rospy.loginfo("Publishing detections...")
        # Debug: Print full bounding boxes, classes, and scores
        rospy.loginfo(f"Bounding Boxes: {bboxes}")
        rospy.loginfo(f"Classes: {classes}")
        rospy.loginfo(f"Scores: {scores}")
        for bbox, cls, score in zip(bboxes, classes, scores):
            cls = int(cls)  # Ensure cls is an integer
            rospy.loginfo(f"Processing detection: BBox={bbox}, Class={cls}, Score={score}")
            if not isinstance(cls, int):
                rospy.logerr(f"Invalid class type: {cls} (expected int)")
                continue
            if not 0 <= cls <= 4:  # Assuming valid classes are 0-4
                rospy.logerr(f"Class ID out of range: {cls}")
                continue
            if (cls in [0, 1] and score <= 0.5) or (cls not in [0, 1] and score <= 0.15):  # Confidence threshold
                rospy.loginfo(f"Skipping detection with low confidence: {score}")
                continue

            # Additional Layer for Traffic Light Processing
            if cls == 3:  # Traffic light
                rospy.loginfo("Traffic light detected, analyzing hue, saturation, and intensity...")
                # Extract hue, saturation, and intensity values
                hue, saturation, intensity = self.get_traffic_light_hsi(bbox, image_msg)
                # Threshold values for determining states
                RED_HUE_RANGE_1 = (70, 125)     # First range for red light
                RED_HUE_RANGE_2 = (0, 30)  # Second range for red light
                GREEN_HUE_RANGE = (45, 75)    # Hue range for green light
                SATURATION_RANGE = (0, 200)  # Saturation range
                INTENSITY_RANGE = (0, 200)   # Intensity (Value) range
                is_saturation_valid = SATURATION_RANGE[0] <= saturation <= SATURATION_RANGE[1]
                is_intensity_valid = INTENSITY_RANGE[0] <= intensity <= INTENSITY_RANGE[1]
                if ((RED_HUE_RANGE_1[0] <= hue <= RED_HUE_RANGE_1[1] or RED_HUE_RANGE_2[0] <= hue <= RED_HUE_RANGE_2[1])
                        and is_saturation_valid and is_intensity_valid):
                    rospy.loginfo(f"Red light detected - Hue: {hue}, Saturation: {saturation}, Intensity: {intensity}")
                    cls = 30  # Red light
                elif GREEN_HUE_RANGE[0] <= hue <= GREEN_HUE_RANGE[1] and is_saturation_valid and is_intensity_valid:
                    rospy.loginfo(f"Green light detected - Hue: {hue}, Saturation: {saturation}, Intensity: {intensity}")
                    cls = 31  # Green light
                else:
                    rospy.loginfo(f"No significant traffic light signal detected - Hue: {hue}, Saturation: {saturation}, Intensity: {intensity}")
                    cls = 31  # We belive it is green light
            # Create detection message
            detection = ObstacleImageDetection()
            detection.bounding_box = Rect(
                x=int(bbox[0]),
                y=int(bbox[1]),
                w=int(bbox[2] - bbox[0]),
                h=int(bbox[3] - bbox[1])
            )
            detection.type.type = int(cls)
            rospy.loginfo(f"Created detection: {detection}")
            detection_list_msg.list.append(detection)
        # Publish the detection list
        rospy.loginfo(f"Publishing {len(detection_list_msg.list)} detections.")
        self.pub_detections_list.publish(detection_list_msg)
    def get_traffic_light_hsi(self, bbox, image_msg):
        """
        Extract average hue, saturation, and intensity (value) for the traffic light bounding box.
        """
        try:
            # Convert CompressedImage to OpenCV BGR format
            np_arr = np.frombuffer(image_msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)  # Decompress the image
            # Extract ROI (Region of Interest)
            roi = cv_image[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
            # Convert ROI to HSV color space
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            # Calculate average hue, saturation, and intensity
            hue = hsv_roi[:, :, 0].mean()  # Hue channel
            saturation = hsv_roi[:, :, 1].mean()  # Saturation channel
            intensity = hsv_roi[:, :, 2].mean()  # Value (intensity) channel
            rospy.loginfo(f"Traffic light HSI - Hue: {hue}, Saturation: {saturation}, Intensity: {intensity}")
            return hue, saturation, intensity
        except Exception as e:
            rospy.logerr(f"Error processing traffic light HSI: {e}")
            return -1, -1, -1  # Return invalid values on error
    def visualize_detections(self, rgb, bboxes, classes):
        colors = {0: (0, 255, 255), 1: (0, 165, 255), 2: (0, 250, 0), 3: (0, 0, 255), 4: (255, 0, 0)}
        names = {0: "duckie", 1: "duckiebot", 2: "intersection_sign", 3: "traffic_light", 4: "stop_sign"}
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
    def det2bool(self, bboxes, classes, scores):
        box_ids = np.array(list(map(filter_by_bboxes, bboxes))).nonzero()[0]
        cla_ids = np.array(list(map(filter_by_classes, classes))).nonzero()[0]
        sco_ids = np.array(list(map(filter_by_scores, scores))).nonzero()[0]
        box_cla_ids = set(box_ids).intersection(set(cla_ids))
        box_cla_sco_ids = set(sco_ids).intersection(box_cla_ids)
        return len(box_cla_sco_ids) > 0
    def pub_car_commands(self, stop, header):
        car_control_msg = Twist2DStamped()
        car_control_msg.header = header
        car_control_msg.v = 0.0 if stop else self.v
        car_control_msg.omega = 0.0
        self.pub_car_cmd.publish(car_control_msg)
if __name__ == "__main__":
    object_detection_node = ObjectDetectionNode(node_name="object_detection_node")
    rospy.spin()