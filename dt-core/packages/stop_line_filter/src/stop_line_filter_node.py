#!/usr/bin/env python3
import numpy as np

import rospy
from duckietown.dtros import DTParam, DTROS, NodeType, ParamType
from duckietown_msgs.msg import BoolStamped, FSMState, LanePose, SegmentList, StopLineReading
from geometry_msgs.msg import Point


# import math


class StopLineFilterNode(DTROS):
    def __init__(self, node_name):
        # Initialize the DTROS parent class
        super(StopLineFilterNode, self).__init__(node_name=node_name, node_type=NodeType.PERCEPTION)

        # Initialize the parameters
        self.stop_distance = DTParam("~stop_distance", param_type=ParamType.FLOAT)
        self.min_segs = DTParam("~min_segs", param_type=ParamType.INT)
        self.off_time = DTParam("~off_time", param_type=ParamType.FLOAT)
        self.max_y = DTParam("~max_y", param_type=ParamType.FLOAT)

        #####################################################
        # set stop line region
        '''
        0.9, 0.9916666666666667
        0.14375000000000002, 0.7666666666666666
        '''
        self.red_x_max = 0.985
        self.red_y_max = 0.9999
        self.red_x_min = 0.120
        self.red_y_min = 0.630
        self.min_stop_line_area = 0.055
        self.stop_line_areas = np.array([.0, .0, .0, .0, .0, .0, .0, .0])
        # self.log("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n" + 
        #         "red_y_min: "+str(self.red_y_min)+
        #         "\nmin_stop_line_area: "+str(self.min_stop_line_areas), "error")
        #####################################################

        ## params
        # self.stop_distance = self.setupParam("~stop_distance", 0.22) # distance from the stop line that we should stop
        # self.min_segs      = self.setupParam("~min_segs", 2) # minimum number of red segments that we should detect to estimate a stop
        # self.off_time      = self.setupParam("~off_time", 2)
        # self.max_y         = self.setupParam("~max_y ", 0.2) # If y value of detected red line is smaller than max_y we will not set at_stop_line true.

        ## state vars
        self.lane_pose = LanePose()

        self.state = "JOYSTICK_CONTROL"
        self.sleep = False

        ## publishers and subscribers
        self.sub_segs = rospy.Subscriber("~segment_list", SegmentList, self.cb_segments)
        self.sub_lane = rospy.Subscriber("~lane_pose", LanePose, self.cb_lane_pose)
        self.sub_mode = rospy.Subscriber("fsm_node/mode", FSMState, self.cb_state_change)
        self.pub_stop_line_reading = rospy.Publisher("~stop_line_reading", StopLineReading, queue_size=1)
        self.pub_at_stop_line = rospy.Publisher("~at_stop_line", BoolStamped, queue_size=1)

    # def setupParam(self,param_name,default_value):
    #     value = rospy.get_param(param_name,default_value)
    #     rospy.set_param(param_name,value) #Write to parameter server for transparancy
    #     rospy.loginfo("[%s] %s = %s " %(self.node_name,param_name,value))
    #     return value
    #
    # def updateParams(self,event):
    #     self.stop_distance = rospy.get_param("~stop_distance")
    #     self.min_segs      = rospy.get_param("~min_segs")
    #     self.off_time      = rospy.get_param("~off_time")
    #     self.max_y         = rospy.get_param("~max_y")

    def cb_state_change(self, msg):
        if (self.state == "INTERSECTION_CONTROL") and (msg.state == "LANE_FOLLOWING"):
            self.after_intersection_work()
        self.state = msg.state

    def after_intersection_work(self):
        self.loginfo("Blocking stop line detection after the intersection")
        stop_line_reading_msg = StopLineReading()
        stop_line_reading_msg.stop_line_detected = False
        stop_line_reading_msg.at_stop_line = False
        self.pub_stop_line_reading.publish(stop_line_reading_msg)
        self.sleep = True
        rospy.sleep(self.off_time.value)
        self.sleep = False
        self.loginfo("Resuming stop line detection after the intersection")

    # def cbSwitch(self, switch_msg):
    #     self.active = switch_msg.data
    #     if self.active and self.state == "INTERSECTION_CONTROL":
    #         self.after_intersection_work()
    #
    #

    def cb_lane_pose(self, lane_pose_msg):
        self.lane_pose = lane_pose_msg

    def cb_segments(self, segment_list_msg):

        if not self.switch or self.sleep:
            return

        # good_seg_count = 0
        stop_line_x_accumulator = 0.0
        stop_line_y_accumulator = 0.0
        #######################################
        red_segments = []
        # self.log("\n*************************************\nshit\n*************************************\n", "error")
        #######################################
        for segment in segment_list_msg.segments:
            if segment.color != segment.RED:
                continue
            # if segment.points[0].x < 0 or segment.points[1].x < 0:  # the point is behind us
            max_x = np.array([segment.pixels_normalized[0].x, segment.pixels_normalized[1].x]).max()
            max_y = np.array([segment.pixels_normalized[0].y, segment.pixels_normalized[1].y]).max()
            min_x = np.array([segment.pixels_normalized[0].x, segment.pixels_normalized[1].x]).min()
            min_y = np.array([segment.pixels_normalized[0].y, segment.pixels_normalized[1].y]).min()

            # self.log("\n***************************************************\n" + \
            #          "max_x: " + str(max_x) + ", max_y: " + str(max_y) + "\n" + \
            #          "min_x: " + str(min_x) + ", min_y: " + str(min_y) + \
            #          "\n***************************************************\n", "error")
            if (max_x>self.red_x_max) or (max_y>self.red_y_max) or (min_x<self.red_x_min) or (min_y<self.red_y_min):  # the point is not in the interested region
                continue

            #######################################
            red_segments.append([[segment.pixels_normalized[0].x, segment.pixels_normalized[0].y], [segment.pixels_normalized[1].x, segment.pixels_normalized[1].y]])
            #######################################

            # p1_lane = self.to_lane_frame(segment.points[0])
            # p2_lane = self.to_lane_frame(segment.points[1])
            # avg_x = 0.5 * (p1_lane[0] + p2_lane[0])
            # avg_y = 0.5 * (p1_lane[1] + p2_lane[1])
            # stop_line_x_accumulator += avg_x
            # stop_line_y_accumulator += avg_y  # TODO output covariance and not just mean
            # good_seg_count += 1.0

        ################################################
        # # self.log("\n##################################################\n", "error")
        stop_line_area = .0
        if not (red_segments == []):
            segment_x = []
            segment_y = []
            for segment in red_segments:
                segment_x.append(segment[0][0])
                segment_x.append(segment[1][0])
                segment_y.append(segment[0][1])
                segment_y.append(segment[1][1])
            #     # self.log("[["+str(segment[0][0])+","+str(segment[0][1])+"]"+ \
            #              # "["+str(segment[1][0])+","+str(segment[1][1])+"]]\n", "error")
            segment_x = np.array(segment_x)
            segment_y = np.array(segment_y)
            max_segment_x = segment_x.max()
            max_segment_y = segment_y.max()
            min_segment_x = segment_x.min()
            min_segment_y = segment_y.min()
            # self.log("\n***************************************************\n"+\
            #          "max_segment_x: "+str(max_segment_x)+", max_segment_y: "+str(max_segment_y)+"\n"+\
            #          "min_segment_x: "+str(min_segment_x)+", min_segment_y: "+str(min_segment_y)+\
            #          "\n***************************************************\n", "error")
            stop_line_area = (max_segment_x - min_segment_x) * (max_segment_y - min_segment_y)
            # self.log("\n##################################################\n" + \
            #          "stop_line_area: " + str(stop_line_area) + \
            #          "\n##################################################\n", "error")
        # else:
            # self.log("\n##################################################\n" + \
            #          "stop_line_area: 0" + \
            #          "\n##################################################\n", "error")
        ################################################

        stop_line_reading_msg = StopLineReading()
        stop_line_reading_msg.header.stamp = segment_list_msg.header.stamp

        #############################
        # self.log("good_seg_count: " + str(good_seg_count) + " self.min_segs.value: " + str(self.min_segs.value), "error")
        # self.log("phi_err too large/small, thresholding it!" + str(phi_err), "error")
        #############################

        # if good_seg_count < self.min_segs.value:

        ##################################################################
        self.stop_line_areas[0:-1] = self.stop_line_areas[1:]
        self.stop_line_areas[-1] = stop_line_area
        avg_stop_line_area = self.stop_line_areas.mean()
        # self.log("\n##################################################\n"+
        #          "avg_stop_line_area: " + str(avg_stop_line_area)+
        #          # "\nmin_stop_line_area: " + str(self.min_stop_line_area) +
        #          "\n##################################################\n", "error")
        ####################################################################

        if  avg_stop_line_area < self.min_stop_line_area:
            stop_line_reading_msg.stop_line_detected = False
            stop_line_reading_msg.at_stop_line = False
            self.pub_stop_line_reading.publish(stop_line_reading_msg)

            # ### CRITICAL: publish false to at stop line output_topic
            # msg = BoolStamped()
            # msg.header.stamp = stop_line_reading_msg.header.stamp
            # msg.data = False
            # self.pub_at_stop_line.publish(msg)
            # ### CRITICAL END

        else:
            stop_line_reading_msg.stop_line_detected = True

            ##########################################
            # self.log("the stop_line_detected should be True", "error")
            good_seg_count = 8
            ##########################################

            stop_line_point = Point()
            stop_line_point.x = stop_line_x_accumulator / good_seg_count
            stop_line_point.y = stop_line_y_accumulator / good_seg_count
            stop_line_reading_msg.stop_line_point = stop_line_point
            # Only detect redline if y is within max_y distance:

            ##########################################
            # self.log("stop_line_point.x = " + str(stop_line_point.x), "error")
            # self.log("np.abs(stop_line_point.y) = " + str(np.abs(stop_line_point.y)), "error")
            # self.log("self.stop_distance.value = " + str(self.stop_distance.value), "error")
            # self.log("self.max_y.value = " + str(self.max_y.value), "error")
            stop_line_reading_msg.at_stop_line = True
            ##########################################

            # stop_line_reading_msg.at_stop_line = (
            #     stop_line_point.x < self.stop_distance.value and np.abs(stop_line_point.y) < self.max_y.value
            # )

            self.pub_stop_line_reading.publish(stop_line_reading_msg)
            ##########################################
            # self.log("stop_line_reading_msg.at_stop_line: ", "error")
            # self.log(stop_line_reading_msg.at_stop_line, "error")
            ##########################################
            if stop_line_reading_msg.at_stop_line:
                ##########################################
                # self.log("if at_stop_line is true", "error")
                ##########################################
                msg = BoolStamped()
                msg.header.stamp = stop_line_reading_msg.header.stamp
                msg.data = True
                self.pub_at_stop_line.publish(msg)

        ####################################################################
        # self.log("\n#####################################\n" + \
        #          "stop_line_detected: " + str(stop_line_reading_msg.stop_line_detected) + \
        #          "\nat_stop_line：" + str(stop_line_reading_msg.at_stop_line) + \
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
    lane_filter_node = StopLineFilterNode(node_name="stop_line_filter")
    rospy.spin()
