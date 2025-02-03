#!/usr/bin/env python3

import os
import rospy
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import Twist2DStamped

class TwistControlNode(DTROS):
    def __init__(self, node_name):
        # initialize the DTROS parent class
        super(TwistControlNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
        
        # static parameters
        vehicle_name = os.environ['VEHICLE_NAME']
        twist_topic = f"/{vehicle_name}/car_cmd_switch_node/cmd"
        
        # Initialize parameters
        self._v = 0.0
        self._omega = 0.0
        
        # construct publisher
        self._publisher = rospy.Publisher(twist_topic, Twist2DStamped, queue_size=1)
    
    def test_straight(self):
        """
        Test straight movement
        """
        self.log("Testing straight movement")
        self._v = 0.5
        self._omega = 0.0
        duration = 3.0
        
        self.execute_movement(duration)
        
    def test_left_turn(self):
        """
        Test left turn movement
        """
        self.log("Testing left turn")
        self._v = 0.3
        self._omega = 2.0
        duration = 5.3
        
        self.execute_movement(duration)
        
    def test_right_turn(self):
        """
        Test right turn movement
        """
        self.log("Testing right turn")
        self._v = 0.3
        self._omega = -3.4
        duration = 2.25
        
        self.execute_movement(duration)

    def execute_movement(self, duration):
        """
        Execute a single movement with current parameters
        """
        self.log(f"Executing movement: v={self._v}, omega={self._omega}, duration={duration}")
        
        start_time = rospy.Time.now()
        rate = rospy.Rate(10)  # 10 Hz
        running_duration = rospy.Duration(duration)
        
        movement_msg = Twist2DStamped(v=self._v, omega=self._omega)
        stop_msg = Twist2DStamped(v=0.0, omega=0.0)
        
        while not rospy.is_shutdown():
            current_time = rospy.Time.now()
            time_elapsed = current_time - start_time
            
            if time_elapsed < running_duration:
                self.log(f"Time elapsed: {time_elapsed.to_sec():.2f}/{duration:.2f}")
                self._publisher.publish(movement_msg)
            else:
                self.log("Movement completed, stopping")
                self._publisher.publish(stop_msg)
                break
                
            rate.sleep()

    def on_shutdown(self):
        """
        Ensure the robot stops on shutdown
        """
        stop = Twist2DStamped(v=0.0, omega=0.0)
        self._publisher.publish(stop)
        self.log("Node is shutting down, stopping robot")

if __name__ == '__main__':
    # create the node
    node = TwistControlNode(node_name='twist_control_test_node')
    
    try:
        # You can uncomment one of these to test individually
        node.test_straight()
        # node.test_left_turn()
        # node.test_right_turn()
        pass
        
    except rospy.ROSInterruptException:
        pass
    
    # keep spinning until shutdown
    rospy.spin()


# import os
# import rospy
# import numpy as np
# from duckietown.dtros import DTROS, NodeType
# from duckietown_msgs.msg import WheelsCmdStamped


# class WheelControlNode(DTROS):

#     def __init__(self, node_name):
#         # initialize the DTROS parent class
#         super(WheelControlNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
#         # static parameters
#         vehicle_name = os.environ['VEHICLE_NAME']
#         wheels_topic = f"/{vehicle_name}/wheels_driver_node/wheels_cmd"
#         # construct publisher
#         self._publisher = rospy.Publisher(wheels_topic, WheelsCmdStamped, queue_size=1)

#     def move_straight(self, velocity):
#         """
#         Move straight by setting the same velocity for both wheels.
#         :param velocity: Velocity for both left and right wheels.
#         """
#         self._vel_left = velocity * 1.2 
#         self._vel_right = velocity

#     def move_left(self, velocity):
#         """
#         Move straight by setting the same velocity for both wheels.
#         :param velocity: Velocity for both left and right wheels.
#         """
#         self._vel_left = velocity * 1.1 
#         self._vel_right = velocity * np.pi * 0.7

#     def move_right(self, velocity):
#         """
#         Move straight by setting the same velocity for both wheels.
#         :param velocity: Velocity for both left and right wheels.
#         """
#         self._vel_left = velocity * 1.2 * 2
#         self._vel_right = velocity / 2


#     def run(self):
#         # Set the publishing rate (10 Hz = 10 messages per second)
#         rate = rospy.Rate(10)  
#         start_time = rospy.Time.now()  # Record the start time
        
#         self.move_straight(velocity=0.1)  # Set to a positive value for forward movement
#         while rospy.Time.now() - start_time < rospy.Duration(5):  # Loop for 2 seconds
#             self.publish_wheel_command()
#             rate.sleep()

#         self.move_left(velocity=0.1)  # Set to a positive value for forward movement
#         while rospy.Time.now() - start_time < rospy.Duration(5):  # Loop for 2 seconds
#             self.publish_wheel_command()
#             rate.sleep()

#         self.move_right(velocity=0.1)  # Set to a positive value for forward movement
#         while rospy.Time.now() - start_time < rospy.Duration(3):  # Loop for 2 seconds
#             self.publish_wheel_command()
#             rate.sleep()
        
#         # Stop the robot after 2 seconds
#         self.move_straight(velocity=0.0)  # Stop the robot by setting velocity to 0
#         for _ in range(10):  # Send stop command for 1 second (10 Hz * 1s)
#             self.publish_wheel_command()
#         rate.sleep()


#     def publish_wheel_command(self):
#         """
#         Publish the wheel command with the current velocities.
#         """
#         message = WheelsCmdStamped(vel_left=self._vel_left, vel_right=self._vel_right)
#         self._publisher.publish(message)

#     def on_shutdown(self):
#         """
#         Stop the wheels when shutting down the node.
#         """
#         stop = WheelsCmdStamped(vel_left=0, vel_right=0)
#         self._publisher.publish(stop)


# if __name__ == '__main__':
#     # create the node
#     node = WheelControlNode(node_name='wheel_control_node')
#     # handle shutdown
#     rospy.on_shutdown(node.on_shutdown)
#     # run node
#     node.run()
