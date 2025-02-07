#!/usr/bin/env python3
import numpy as np
import rospy
from duckietown.dtros import DTParam, DTROS, NodeType, ParamType
from duckietown_msgs.msg import BoolStamped, FSMState, LanePose, SegmentList, StopLineReading
from geometry_msgs.msg import Point

class StopLineFilterNode(DTROS):
    def __init__(self, node_name):
        super(StopLineFilterNode, self).__init__(node_name=node_name, node_type=NodeType.PERCEPTION)
        
        # Initialize parameters
        self.stop_distance = DTParam("~stop_distance", param_type=ParamType.FLOAT)
        self.min_segs = DTParam("~min_segs", param_type=ParamType.INT)
        self.off_time = DTParam("~off_time", param_type=ParamType.FLOAT)
        self.max_y = DTParam("~max_y", param_type=ParamType.FLOAT)
        
        # Stop line detection region
        self.red_x_max, self.red_y_max = 0.985, 0.9999
        self.red_x_min, self.red_y_min = 0.120, 0.630
        self.min_stop_line_area = 0.055
        self.stop_line_areas = np.zeros(8)
        
        # State variables
        self.lane_pose = LanePose()
        self.state = "JOYSTICK_CONTROL"
        self.sleep = False
        
        # Publishers and subscribers
        self.sub_segs = rospy.Subscriber("~segment_list", SegmentList, self.cb_segments)
        self.sub_lane = rospy.Subscriber("~lane_pose", LanePose, self.cb_lane_pose)
        self.sub_mode = rospy.Subscriber("fsm_node/mode", FSMState, self.cb_state_change)
        self.pub_stop_line_reading = rospy.Publisher("~stop_line_reading", StopLineReading, queue_size=1)
        self.pub_at_stop_line = rospy.Publisher("~at_stop_line", BoolStamped, queue_size=1)

    def cb_state_change(self, msg):
        if self.state == "INTERSECTION_CONTROL" and msg.state == "LANE_FOLLOWING":
            self.after_intersection_work()
        self.state = msg.state

    def after_intersection_work(self):
        self.loginfo("Blocking stop line detection after the intersection")
        self.pub_stop_line_reading.publish(StopLineReading(stop_line_detected=False, at_stop_line=False))
        self.sleep = True
        rospy.sleep(self.off_time.value)
        self.sleep = False
        self.loginfo("Resuming stop line detection after the intersection")
    
    def cb_lane_pose(self, lane_pose_msg):
        self.lane_pose = lane_pose_msg
    
    def cb_segments(self, segment_list_msg):
        if self.sleep and self.state != "LANE_FOLLOWING":
            return

        red_segments = [
            [[seg.pixels_normalized[0].x, seg.pixels_normalized[0].y], [seg.pixels_normalized[1].x, seg.pixels_normalized[1].y]]
            for seg in segment_list_msg.segments if seg.color == seg.RED and 
            self.red_x_min <= min(seg.pixels_normalized[0].x, seg.pixels_normalized[1].x) <= self.red_x_max and
            self.red_y_min <= min(seg.pixels_normalized[0].y, seg.pixels_normalized[1].y) <= self.red_y_max
        ]
        
        stop_line_area = 0.0
        if red_segments:
            segment_x = np.array([p[0] for seg in red_segments for p in seg])
            segment_y = np.array([p[1] for seg in red_segments for p in seg])
            stop_line_area = (segment_x.max() - segment_x.min()) * (segment_y.max() - segment_y.min())
        
        self.stop_line_areas[:-1] = self.stop_line_areas[1:]
        self.stop_line_areas[-1] = stop_line_area
        avg_stop_line_area = self.stop_line_areas.mean()

        stop_line_reading_msg = StopLineReading()
        stop_line_reading_msg.header.stamp = segment_list_msg.header.stamp
        stop_line_reading_msg.stop_line_detected = avg_stop_line_area >= self.min_stop_line_area
        stop_line_reading_msg.at_stop_line = stop_line_reading_msg.stop_line_detected
        self.pub_stop_line_reading.publish(stop_line_reading_msg)
        
        if stop_line_reading_msg.at_stop_line:
            self.pub_at_stop_line.publish(BoolStamped(header=stop_line_reading_msg.header, data=True))
        else:
            self.pub_at_stop_line.publish(BoolStamped(header=stop_line_reading_msg.header, data=False))

if __name__ == "__main__":
    lane_filter_node = StopLineFilterNode(node_name="stop_line_filter")
    rospy.spin()
