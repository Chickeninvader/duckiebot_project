cmake_minimum_required(VERSION 2.8.3)
project(line_seg_detector_improve)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  image_transport
  roscpp
  rospy
  duckietown_msgs
  geometry_msgs
  sensor_msgs
  std_msgs
)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
)
