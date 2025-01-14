#!/bin/bash

source /environment.sh

# initialize launch file
dt-launchfile-init

# launch traffic light
rosrun traffic_light traffic_light_node.py


# wait for app to end
dt-launchfile-join