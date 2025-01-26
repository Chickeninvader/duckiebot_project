import rospy
from lane_controller.controller import LaneController
from fsm.predefine_car_routine import PredefinedCarRoutine
class FSM:
    '''
    This class handle state transistion:
    State1: normal moving
    State2: stop sign detected
    State4: traffic light check
    State6: intersection routine
    '''
    def __init__(self):
        self._state = "state1"

        self._v = 0
        self._omega = 0

        self._is_stop_sign = False
        self._stop_sign_buffer = queue(10)

        self._at_stop_line = False

        self._is_red_light = False
        self._red_light_buffer = queue(10)

        self._is_close_car = False
        self._last_state_print_time = rospy.time()

    def _handle_state(self, v, omega):
        current_time = rospy.time()

        # Print and publish state every 1 second
        if current_time - self._last_state_print_time >= 1.0:
            rospy.loginfo(f"Current state: {self._state}")
            self._last_state_print_time = current_time

        if self._state == "state1":
            self._handle_state1(current_time)
        elif self._state == "state2":
            self._handle_state2(current_time)
        elif self._state == "state4":
            self._handle_state4(current_time)
        elif self._state == "state6":
            self._handle_state6(current_time)

    def _update_info(kwargs):
        """
        Parameter is update here
        """
        pass

    def _handle_state1(self, current_time):
        self.publish_twist()
        if self._stop_sign_detected > 3:
            self._state = "state2"
            self._state2_start_time = current_time
            self._state2_phase = "running"
            self.log(f"Stop sign detected {self._stop_sign_detected} times. Entering state2 running phase")
   
    def _handle_state2(self, current_time):
        self.log(f"State2 stopping phase: Time elapsed = {current_time - self._state2_start_time:.2f}s")
        if current_time - self._state2_start_time >= 2.0:
            self._state = "state1"
            self._stop_sign_detected = 0
            self.log("State2: Stop completed. Returning to state1")

    def _handle_state4(self, current_time):
        self._predefined_routine.execute_intersection_routine(direction="stop")
        self.log(f"State4: Checking traffic light. Red light detections = {self._red_light_detected}, Time elapsed = {current_time - self._traffic_light_check_start:.2f}s")
        if self._traffic_light_check_start and (current_time - self._traffic_light_check_start >= 1.5):
            if self._red_light_detected < 3:
                self._state = "state6"
                self._traffic_light_check_start = None
                self.log(f"State4: No red light detected: {self._red_light_detected}. Transitioning to state61")
            else:
                self._traffic_light_check_start = current_time
                self.log(f"State4: Red light detected: {self._red_light_detected}. Resetting check timer")
                self._red_light_detected = 0

    def _handle_state6(self):
        self.log("State61: Executing intersection routine")
        self.intersection_routine()
        rospy.sleep(1)
        self._state = "state1"
        self.log("State6: Intersection routine first phase completed. Returning to state1")

    def publish_FSM(self):
        # Implement logic to publish FSM states if necessary
        pass

    def publish_twist(self):
        # Implement logic to publish twist commands
        pass

    def intersection_routine(self):
        # Implement logic for intersection routine
        PredefinedCarRoutine.move_straight()

    def reset_variable(self):
        # Implement reset logic
        self._v = 0
        self._omega = 0

        self._is_stop_sign = False
        self._stop_sign_buffer = queue(10)

        self._at_stop_line = False

        self._is_red_light = False
        self._red_light_buffer = queue(10)

        self._is_close_car = False
        self._last_state_print_time = rospy.time()

