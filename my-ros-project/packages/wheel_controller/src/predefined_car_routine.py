import rospy
from duckietown_msgs.msg import Twist2DStamped

class PredefinedCarRoutine:
    # Fixed durations for each type of movement (in seconds)
    STRAIGHT_DURATION = 5.0
    TURN_RIGHT_DURATION = 1.5
    TURN_LEFT_DURATION = 3.3
    STOP_DURATION = 0.5
    
    def __init__(self, twist_publisher):
        """
        Initialize the PredefinedCarRoutine class.
        :param twist_publisher: Publisher for Twist2DStamped messages.
        """
        self._publisher = twist_publisher
        self._v = 0  # linear velocity
        self._omega = 0.00  # angular velocity

    def move_straight(self):
        """
        Move straight with specified velocity.
        :param velocity: Linear velocity for forward motion.
        """
        self._v = 0.25
        self._omega = -0.6
        self._publish_command()

    def turn_left(self):
        """
        Turn left with specified velocity.
        :param velocity: Base velocity for the maneuver.
        """
        self._v = 0.2  # Reduce forward velocity during turn
        self._omega = 2.0  # Positive omega for left turn
        self._publish_command()

    def turn_right(self, velocity):
        """
        Turn right with specified velocity.
        :param velocity: Base velocity for the maneuver.
        """
        self._v = 0.3  # Reduce forward velocity during turn
        self._omega = -3.4  # Negative omega for right turn
        self._publish_command()

    def stop(self):
        """
        Stop the vehicle by setting both velocities to zero.
        """
        self._v = 0
        self._omega = 0
        self._publish_command()

    def _publish_command(self):
        """
        Publish the current velocities as a Twist2DStamped message.
        """
        message = Twist2DStamped()
        message.v = self._v
        message.omega = self._omega
        self._publisher.publish(message)

    def execute_intersection_routine(self, direction):
        """
        Execute a routine to navigate an intersection.
        :param direction: Direction to go ('straight', 'left', or 'right') or stop.
        """
        rate = rospy.Rate(10)  # 10 Hz control loop
        start_time = rospy.Time.now()
        
        # Set the movement and duration based on direction
        if direction == 'straight':
            self.move_straight()
            duration = self.STRAIGHT_DURATION
        elif direction == 'left':
            self.turn_left()
            duration = self.TURN_LEFT_DURATION
        elif direction == 'right':
            self.turn_right()
            duration = self.TURN_RIGHT_DURATION
        elif direction == 'stop':
            self.stop()
            duration = self.STOP_DURATION
        else:
            raise ValueError("Invalid direction. Use 'straight', 'left', or 'right'.")

        # Continue the movement for the fixed duration
        while rospy.Time.now() - start_time < rospy.Duration(duration):
            self._publish_command()
            rate.sleep()

        # Stop after completing the maneuver
        self.stop()