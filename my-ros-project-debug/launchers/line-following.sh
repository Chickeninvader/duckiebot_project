#!/bin/bash

source /environment.sh

# initialize launch file
dt-launchfile-init

# launch app
dt-exec roslaunch --wait my_demo lane_following.launch


# wait for app to end
dt-launchfile-join