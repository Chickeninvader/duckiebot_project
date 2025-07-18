#!/usr/bin/env python3
import numpy as np
import rospy

from duckietown.dtros import DTROS, NodeType, TopicType, DTParam, ParamType
from duckietown_msgs.msg import (
    Twist2DStamped,
    LanePose,
    WheelsCmdStamped,
    BoolStamped,
    FSMState,
    StopLineReading,
)

from lane_controller.controller import LaneController


class LaneControllerNode(DTROS):
    """Computes control action.
    The node compute the commands in form of linear and angular velocities, by processing the estimate error in
    lateral deviationa and heading.
    The configuration parameters can be changed dynamically while the node is running via ``rosparam set`` commands.
    Args:
        node_name (:obj:`str`): a unique, descriptive name for the node that ROS will use
    Configuration:
        ~v_bar (:obj:`float`): Nominal velocity in m/s
        ~k_d (:obj:`float`): Proportional term for lateral deviation
        ~k_theta (:obj:`float`): Proportional term for heading deviation
        ~k_Id (:obj:`float`): integral term for lateral deviation
        ~k_Iphi (:obj:`float`): integral term for lateral deviation
        ~d_thres (:obj:`float`): Maximum value for lateral error
        ~theta_thres (:obj:`float`): Maximum value for heading error
        ~d_offset (:obj:`float`): Goal offset from center of the lane
        ~integral_bounds (:obj:`dict`): Bounds for integral term
        ~d_resolution (:obj:`float`): Resolution of lateral position estimate
        ~phi_resolution (:obj:`float`): Resolution of heading estimate
        ~omega_ff (:obj:`float`): Feedforward part of controller
        ~verbose (:obj:`bool`): Verbosity level (0,1,2)
        ~stop_line_slowdown (:obj:`dict`): Start and end distances for slowdown at stop lines

    Publisher:
        ~car_cmd (:obj:`Twist2DStamped`): The computed control action
    Subscribers:
        ~lane_pose (:obj:`LanePose`): The lane pose estimate from the lane filter
        ~intersection_navigation_pose (:obj:`LanePose`): The lane pose estimate from intersection navigation
        ~wheels_cmd_executed (:obj:`WheelsCmdStamped`): Confirmation that the control action was executed
        ~stop_line_reading (:obj:`StopLineReading`): Distance from stopline, to reduce speed
        ~obstacle_distance_reading (:obj:`stop_line_reading`): Distancefrom obstacle virtual stopline, to reduce speed
    """

    def __init__(self, node_name):

        # Initialize the DTROS parent class
        super(LaneControllerNode, self).__init__(node_name=node_name, node_type=NodeType.PERCEPTION)

        # Add the node parameters to the parameters dictionary
        # TODO: MAKE TO WORK WITH NEW DTROS PARAMETERS
        self.params = dict()
        self.params["~v_bar"] = DTParam("~v_bar", param_type=ParamType.FLOAT, min_value=0.0, max_value=5.0)
        self.params["~k_d"] = DTParam("~k_d", param_type=ParamType.FLOAT, min_value=-100.0, max_value=100.0)
        self.params["~k_theta"] = DTParam(
            "~k_theta", param_type=ParamType.FLOAT, min_value=-100.0, max_value=100.0
        )
        self.params["~k_Id"] = DTParam("~k_Id", param_type=ParamType.FLOAT, min_value=-100.0, max_value=100.0)
        self.params["~k_Iphi"] = DTParam(
            "~k_Iphi", param_type=ParamType.FLOAT, min_value=-100.0, max_value=100.0
        )
        #self.params["~theta_thres"] = rospy.get_param("~theta_thres", None)
        #Breaking up the self.params["~theta_thres"] parameter for more finer tuning of phi
        self.params["~theta_thres_min"] = DTParam("~theta_thres_min", param_type=ParamType.FLOAT, min_value=-100.0, max_value=100.0)  #SUGGESTION mandatorizing the use of DTParam inplace of rospy.get_param for parameters in the entire dt-core repository as it allows active tuning while Robot is in action.
        self.params["~theta_thres_max"] = DTParam("~theta_thres_max", param_type=ParamType.FLOAT, min_value=-100.0, max_value=100.0) 
        self.params["~d_thres"] = rospy.get_param("~d_thres", None)
        self.params["~d_offset"] = rospy.get_param("~d_offset", None)
        self.params["~integral_bounds"] = rospy.get_param("~integral_bounds", None)
        self.params["~d_resolution"] = rospy.get_param("~d_resolution", None)
        self.params["~phi_resolution"] = rospy.get_param("~phi_resolution", None)
        self.params["~omega_ff"] = rospy.get_param("~omega_ff", None)
        self.params["~verbose"] = rospy.get_param("~verbose", None)
        self.params["~stop_line_slowdown"] = rospy.get_param("~stop_line_slowdown", None)

        # Need to create controller object before updating parameters, otherwise it will fail
        self.controller = LaneController(self.params)
        # self.updateParameters() # TODO: This needs be replaced by the new DTROS callback when it is implemented

        # Initialize variables

        self.is_switching_lane = None
        self.fsm_state = None
        self.wheels_cmd_executed = WheelsCmdStamped()
        self.pose_msg = LanePose()
        self.pose_initialized = False
        self.pose_msg_dict = dict()
        self.last_s = None
        self.stop_line_distance = None
        self.stop_line_detected = False
        self.at_stop_line = False
        self.obstacle_stop_line_distance = None
        self.obstacle_stop_line_detected = False
        self.at_obstacle_stop_line = False
        self.switch_lane_timeout = None
        self.pose_msg = None
        self.reversed_omega = False
        self.obstacle_avoidance_timeout = None
        self.is_finish_switch_lane = False
        self.switch_lane_start_time = None
        self.obstacle_avoidance_finish = False

        self.current_pose_source = "lane_filter"

        # Construct publishers
        self.pub_car_cmd = rospy.Publisher(
            "~car_cmd", Twist2DStamped, queue_size=1, dt_topic_type=TopicType.CONTROL
        )
        self.pub_finish_switch_lane = rospy.Publisher(
            "~finish_switch_lane", BoolStamped, queue_size=1
        )
        self.pub_finish_obstacle_avoidance = rospy.Publisher(
            "~finish_obstacle_avoidance", BoolStamped, queue_size=1
        )

        # Construct subscribers
        self.sub_lane_reading = rospy.Subscriber(
            "~lane_pose", LanePose, self.cbAllPoses, "lane_filter", queue_size=1
        )
        self.sub_intersection_navigation_pose = rospy.Subscriber(
            "~intersection_navigation_pose",
            LanePose,
            self.cbAllPoses,
            "intersection_navigation",
            queue_size=1,
        )
        self.sub_wheels_cmd_executed = rospy.Subscriber(
            "~wheels_cmd", WheelsCmdStamped, self.cbWheelsCmdExecuted, queue_size=1
        )
        self.sub_stop_line = rospy.Subscriber(
            "~stop_line_reading", StopLineReading, self.cbStopLineReading, queue_size=1
        )
        self.sub_obstacle_stop_line = rospy.Subscriber(
            "~obstacle_distance_reading", StopLineReading, self.cbObstacleStopLineReading, queue_size=1
        )
        self.sub_fsm_mode = rospy.Subscriber("~fsm_mode", FSMState, self.cbMode, queue_size=1)

        self.log("Initialized!")

    def cbObstacleStopLineReading(self, msg):
        """
        Callback storing the current obstacle distance, if detected.

        Args:
            msg(:obj:`StopLineReading`): Message containing information about the virtual obstacle stopline.
        """
        self.obstacle_stop_line_distance = np.sqrt(msg.stop_line_point.x**2 + msg.stop_line_point.y**2)
        self.obstacle_stop_line_detected = msg.stop_line_detected
        self.at_stop_line = msg.at_stop_line

    def cbStopLineReading(self, msg):
        """Callback storing current distance to the next stopline, if one is detected.

        Args:
            msg (:obj:`StopLineReading`): Message containing information about the next stop line.
        """
        self.stop_line_distance = np.sqrt(msg.stop_line_point.x**2 + msg.stop_line_point.y**2)
        self.stop_line_detected = msg.stop_line_detected
        self.at_obstacle_stop_line = msg.at_stop_line

    def cbMode(self, fsm_state_msg):

        self.fsm_state = fsm_state_msg.state  # String of current FSM state

        if self.fsm_state == "INTERSECTION_CONTROL":
            self.current_pose_source = "intersection_navigation"
        else:
            self.current_pose_source = "lane_filter"

        # if self.params["~verbose"] == 2:
        self.log("Pose source: %s" % self.current_pose_source)

    def cbAllPoses(self, input_pose_msg, pose_source):
        """Callback receiving pose messages from multiple topics.

        If the source of the message corresponds with the current wanted pose source, it computes a control command.

        Args:
            input_pose_msg (:obj:`LanePose`): Message containing information about the current lane pose.
            pose_source (:obj:`String`): Source of the message, specified in the subscriber.
        """

        if pose_source == self.current_pose_source:
            self.pose_msg_dict[pose_source] = input_pose_msg

            self.pose_msg = input_pose_msg

            self.getControlAction(self.pose_msg)

    def cbWheelsCmdExecuted(self, msg_wheels_cmd):
        """Callback that reports if the requested control action was executed.

        Args:
            msg_wheels_cmd (:obj:`WheelsCmdStamped`): Executed wheel commands
        """
        self.wheels_cmd_executed = msg_wheels_cmd

    def publishCmd(self, car_cmd_msg):
        """Publishes a car command message.

        Args:
            car_cmd_msg (:obj:`Twist2DStamped`): Message containing the requested control action.
        """
        self.pub_car_cmd.publish(car_cmd_msg)
        # self.log("Publishing car command: v = %f, omega = %f" % (car_cmd_msg.v, car_cmd_msg.omega))

    def getControlAction(self, pose_msg):
        """Callback that receives a pose message and updates the related control command.

        Using a controller object, computes the control action using the current pose estimate.

        Args:
            pose_msg (:obj:`LanePose`): Message containing information about the current lane pose.
        """
        v = 0
        omega = 0
        current_s = rospy.Time.now().to_sec()
        dt = None
        if self.last_s is not None:
            dt = current_s - self.last_s

        # Define constants for the lane switch durations
        SWITCH_LANE_LEFT_FIRST_PHASE = 1.2
        SWITCH_LANE_LEFT_SECOND_PHASE = 2.2
        SWITCH_LANE_RIGHT_FIRST_PHASE = 0.1
        SWITCH_LANE_RIGHT_SECOND_PHASE = 0.2

        # Introduce a dedicated state variable for lane switching
        if self.fsm_state in ["SWITCH_LANE_LEFT", "SWITCH_LANE_RIGHT"]:
            if not self.is_switching_lane and not self.is_finish_switch_lane:
                self.is_switching_lane = True
                self.switch_lane_start_time = rospy.Time.now()
                self.reversed_omega = False
                self.log(f"Entered SWITCH_LANE ({self.fsm_state}). Starting maneuver.")

            if self.is_switching_lane:
                elapsed_time = (rospy.Time.now() - self.switch_lane_start_time).to_sec()
                self.log(f"SWITCH_LANE ({self.fsm_state}): Elapsed time = {elapsed_time:.2f}s")

                v = 0.1
                omega = 0.0  # Initialize omega

                first_phase_duration = SWITCH_LANE_LEFT_FIRST_PHASE if self.fsm_state == "SWITCH_LANE_LEFT" else SWITCH_LANE_RIGHT_FIRST_PHASE
                second_phase_duration = SWITCH_LANE_LEFT_SECOND_PHASE if self.fsm_state == "SWITCH_LANE_LEFT" else SWITCH_LANE_RIGHT_SECOND_PHASE

                if elapsed_time < first_phase_duration:
                    omega = 3.8 if self.fsm_state == "SWITCH_LANE_LEFT" else -3.0
                elif elapsed_time < second_phase_duration:
                    omega = -1.0 if self.fsm_state == "SWITCH_LANE_LEFT" else 1.0
                    if not self.reversed_omega:
                        self.log(f"SWITCH_LANE ({self.fsm_state}): Reversing omega for stabilization.")
                        self.reversed_omega = True
                elif not self.is_finish_switch_lane:
                    self.is_switching_lane = False
                    self.reversed_omega = False
                    self.log("SWITCH_LANE: Maneuver completed. Publishing finish message.", 'info')

                    msg = BoolStamped()
                    msg.header.stamp = rospy.Time.now()
                    msg.data = True
                    for _ in range(4):  # Publish multiple times without blocking
                        self.pub_finish_switch_lane.publish(msg)
                        rospy.sleep(0.05)
                    car_control_msg = Twist2DStamped()
                    car_control_msg.header = pose_msg.header
                    self.pose_msg = pose_msg

                    # Add commands to car message
                    car_control_msg.v = 0
                    car_control_msg.omega = 0 # Multiply by 0.5 in sim

                    for _ in range(4):  # Publish multiple times without blocking
                        self.pub_car_cmd.publish(car_control_msg)
                        rospy.sleep(0.05)
                    self.is_finish_switch_lane = True
                    return  # Exit the callback after finishing the maneuver

                self.log(f"SWITCH_LANE ({self.fsm_state}): Setting omega = {omega:.2f}, v_ref = {v:.2f}")

        elif ((self.at_stop_line or self.at_obstacle_stop_line) and (self.current_pose_source == 'lane_filter')
                or self.fsm_state == "OBSTACLE_STOP"):
            v = 0
            omega = 0
            
        elif self.current_pose_source == 'lane_filter':

            if self.fsm_state == "OBSTACLE_AVOIDANCE":
                d_offset = self.params["~d_offset"]
                theta_thres_max = -self.params["~theta_thres_min"].value
                theta_thres_min = -self.params["~theta_thres_max"].value
            else:
                d_offset = self.params["~d_offset"]
                theta_thres_max = self.params["~theta_thres_max"].value
                theta_thres_min = self.params["~theta_thres_min"].value

            # Compute errors
            d_err = pose_msg.d - d_offset
            phi_err = pose_msg.phi

            # Thresholding errors
            if np.abs(d_err) > self.params["~d_thres"]:
                self.log("d_err too large, thresholding it!", "error")
                d_err = np.sign(d_err) * self.params["~d_thres"]

            if phi_err > theta_thres_max or phi_err < theta_thres_min:
                self.log("phi_err too large/small, thresholding it!", "error")
                phi_err = np.maximum(theta_thres_min, np.minimum(phi_err, theta_thres_max))

            # Compute control command
            wheels_cmd_exec = [self.wheels_cmd_executed.vel_left, self.wheels_cmd_executed.vel_right]
            if self.obstacle_stop_line_detected:
                v, omega = self.controller.compute_control_action(
                    d_err, phi_err, dt, wheels_cmd_exec, self.obstacle_stop_line_distance
                )
                v = v * 0.25
                omega = omega * 0.25
            else:
                v, omega = self.controller.compute_control_action(
                    d_err, phi_err, dt, wheels_cmd_exec, self.stop_line_distance
                )
            # For feedforward action (i.e. during intersection navigation)
            omega += self.params["~omega_ff"]

        elif self.current_pose_source == 'intersection_navigation':
            v = pose_msg.v_ref
            omega = -pose_msg.phi

        ## Current implementation do not handle timeout in fsm. Temporarily make time out in this node.

        if self.fsm_state not in ['SWITCH_LANE_LEFT', "SWITCH_LANE_RIGHT"]: # Assuming you have a function to check lane
            self.is_switching_lane = False
            self.switch_lane_start_time = None
            self.reversed_omega = False
            self.is_finish_switch_lane = False
            # self.log("SWITCH_LANE: Lane switch confirmed by perception. Resetting state.")

        if self.fsm_state in ['SWITCH_LANE_LEFT', "SWITCH_LANE_RIGHT"] and self.is_finish_switch_lane:
            return

        if self.fsm_state in ["OBSTACLE_AVOIDANCE"]:
            self.controller.reset_controller()
            # After 5s of obstacle avoidance, we publish msg to topic: "lane_controller_node/obstacle_avoidance_finish"
            if self.obstacle_avoidance_timeout is None and not self.obstacle_avoidance_finish:
                self.obstacle_avoidance_timeout = rospy.Time.now()
                self.log("Entered OBSTACLE_AVOIDANCE state. Starting timeout.")
            elapsed_time = (rospy.Time.now() - self.obstacle_avoidance_timeout).to_sec()
            self.log(f"OBSTACLE_AVOIDANCE: Elapsed time = {elapsed_time:.2f}s")
            if elapsed_time > 1:
                self.log("OBSTACLE_AVOIDANCE: Time out and in approximately straight lane. Publishing finish message.", 'info')

                msg = BoolStamped()
                msg.header.stamp = rospy.Time.now()
                msg.data = True
                for _ in range(4):
                    self.pub_finish_obstacle_avoidance.publish(msg)
                    rospy.sleep(0.05)
                self.obstacle_avoidance_finish = True
                rospy.sleep(1)
        else:
            self.obstacle_avoidance_timeout = None
        if self.fsm_state in ['SWITCH_LANE_RIGHT']:
            self.obstacle_avoidance_finish = False
            self.controller.reset_controller()

        # Initialize car control msg, add header from input message
        car_control_msg = Twist2DStamped()
        car_control_msg.header = pose_msg.header
        self.pose_msg = pose_msg

        # Add commands to car message
        car_control_msg.v = v * 2
        car_control_msg.omega = omega # Multiply by 0.5 in sim

        self.publishCmd(car_control_msg)
        self.last_s = current_s
        # self.log("Publishing car command: v = %f, omega = %f" % (car_control_msg.v, car_control_msg.omega))

    def cbParametersChanged(self):
        """Updates parameters in the controller object."""

        self.controller.update_parameters(self.params)


if __name__ == "__main__":
    # Initialize the node
    lane_controller_node = LaneControllerNode(node_name="lane_controller_node")
    # Keep it spinning
    rospy.spin()
