#!/bin/bash

source /environment.sh

# initialize launch file
dt-launchfile-init

# launch subscriber
dt-exec roslaunch --wait vehicle_detection vehicle_detection_node.launch
dt-exec roslaunch --wait vehicle_detection vehicle_filter_node.launch

# wait for app to end
dt-launchfile-join