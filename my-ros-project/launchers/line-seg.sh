#!/bin/bash

source /environment.sh

# initialize launch file
dt-launchfile-init

# launch app
rosrun line_seg_detector_improve line_seg_detector_node.py


# wait for app to end
dt-launchfile-join