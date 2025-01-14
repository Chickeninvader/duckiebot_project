#!/usr/bin/env python3
import numpy as np

import rospy
from duckietown.dtros import DTParam, DTROS, NodeType, ParamType
from duckietown_msgs.msg import BoolStamped, FSMState, LanePose, SegmentList, StopLineReading
from geometry_msgs.msg import Point


# import math


class TrafficLightFilterNode(DTROS):
    def __init__(self, node_name):
        # Initialize the DTROS parent class
        super(TrafficLightFilterNode, self).__init__(node_name=node_name, node_type=NodeType.PERCEPTION)

        # Initialize the parameters
        # self.stop_distance = DTParam("~stop_distance", param_type=ParamType.FLOAT)
        # self.min_segs = DTParam("~min_segs", param_type=ParamType.INT)
        self.off_time = DTParam("~off_time", param_type=ParamType.FLOAT)
        self.max_y = DTParam("~max_y", param_type=ParamType.FLOAT)


        ## state vars
        self.lane_pose = LanePose()

        self.state = "JOYSTICK_CONTROL"
        self.sleep = False

        ###########################################
        self.is_red_light = np.array([.0, .0, .0, .0, .0, .0, .0, .0])
        self.threshold_red_light = 0.70
        ###########################################

        ## publishers and subscribers
        self.sub_segs_top = rospy.Subscriber("~segment_list_top", SegmentList, self.cb_segments_top)
        self.sub_lane = rospy.Subscriber("~lane_pose", LanePose, self.cb_lane_pose)
        self.sub_mode = rospy.Subscriber("fsm_node/mode", FSMState, self.cb_state_change)
        self.pub_traffic_light_reading = rospy.Publisher("~traffic_light_reading", StopLineReading, queue_size=1)


    def cb_state_change(self, msg):
        if (self.state == "INTERSECTION_CONTROL") and (msg.state == "LANE_FOLLOWING"):
            self.after_intersection_work()
        self.state = msg.state

    def after_intersection_work(self):
        self.loginfo("Blocking traffic light detection after the intersection")
        traffic_light_reading_msg = StopLineReading()
        traffic_light_reading_msg.stop_line_detected = False
        traffic_light_reading_msg.at_stop_line = False
        self.pub_traffic_light_reading.publish(traffic_light_reading_msg)
        self.sleep = True
        rospy.sleep(self.off_time.value)
        self.sleep = False
        self.loginfo("Resuming traffic light detection after the intersection")

    # def cbSwitch(self, switch_msg):
    #     self.active = switch_msg.data
    #     if self.active and self.state == "INTERSECTION_CONTROL":
    #         self.after_intersection_work()
    #
    #

    def cb_lane_pose(self, lane_pose_msg):
        self.lane_pose = lane_pose_msg

    def cb_segments_top(self, segment_list_msg_top):

        ##############################################
        # self.log("\n###########################################\n"+"segment_list_msg_top:\n"+ \
        #          str(segment_list_msg_top)+ \
        #          "\nlen(segment_list_msg_top.segments) = " + str(len(segment_list_msg_top.segments)) + \
        #          "\n###########################################\n", "error")
        ##############################################

        if not self.switch or self.sleep:
            return

        # Create the traffic light reading message
        traffic_light_reading_msg = StopLineReading()
        traffic_light_reading_msg.header.stamp = segment_list_msg_top.header.stamp

        # if there is red segment in the top image, then, the red light is detected

        if len(segment_list_msg_top.segments) > 0:
            ##################################################################
            self.is_red_light[0:-1] = self.is_red_light[1:]
            self.is_red_light[-1] = True
            ####################################################################
        else:
            ##################################################################
            self.is_red_light[0:-1] = self.is_red_light[1:]
            self.is_red_light[-1] = False
            ####################################################################

        ####################################################
        # self.log("\n*****************************************\n" +
        #          "self.is_red_light.mean(): " + str(self.is_red_light.mean()) +
        #          "\nthreshold_red_light: " + str(self.threshold_red_light) +
        #          "\n*****************************************\n", "error")
        ####################################################

        if self.is_red_light.mean() > self.threshold_red_light:
            traffic_light_reading_msg.stop_line_detected = True  # at this stage, stop_line_detected == red_traffic_light_detected
        else:
            traffic_light_reading_msg.stop_line_detected = False


        self.pub_traffic_light_reading.publish(traffic_light_reading_msg)

        ####################################################################
        # self.log("\n#####################################\n" + \
        #          "is red light detected in the top image: " + str(traffic_light_reading_msg.stop_line_detected) + \
        #          "\n#####################################\n", "error")
        #####################################################################

    def to_lane_frame(self, point):
        p_homo = np.array([point.x, point.y, 1])
        phi = self.lane_pose.phi
        d = self.lane_pose.d
        T = np.array([[np.cos(phi), -np.sin(phi), 0], [np.sin(phi), np.cos(phi), d], [0, 0, 1]])
        p_new_homo = T.dot(p_homo)
        p_new = p_new_homo[0:2]
        return p_new

    # def onShutdown(self):
    #     rospy.loginfo("[StopLineFilterNode] Shutdown.")


if __name__ == "__main__":
    traffi_light_filter_node = TrafficLightFilterNode(node_name="traffic_light_filter")
    rospy.spin()
