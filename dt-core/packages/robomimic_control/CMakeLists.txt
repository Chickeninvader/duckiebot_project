cmake_minimum_required(VERSION 2.8.3)
project(robomimic_control)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  duckietown_msgs
  roscpp
  rospy
  sensor_msgs
  message_runtime
)

catkin_python_setup()

catkin_package(
  CATKIN_DEPENDS cv_bridge duckietown_msgs roscpp rospy sensor_msgs message_runtime
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

