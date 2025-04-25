# Duckiebot Lane-following Project Setup Guide

## Project Overview

This repository contains routines to collect data, train, and test lane-follwing task for Duckiebot:

## Prerequisites

- Duckiebot running and connected to your network, make sure it is charge to more than 50%
- DTS (Duckietown Shell) installed
- ROS environment properly configured

## Instructions (execute this at the directory where the project is)

### 1. Data collection

# Physical duckiebot: 
```bash
# Build on the robot at dt-core directory
dts devel build -f

# Copy calibration file
scp -r duckie@robotname.local:/data/config/calibration/ /data/config/calibration/

# Run dt-core interactively
dts devel run -R robotname  --mount /path/to/dt-core --cmd /bin/bash

# Within the terminal, launch this custom lane-following:
launcher/darl.sh
# Hold A to run lane-following routine, S to stop, arrow key to navigate the robot when it stops execute routine

# Open another terminal,
docker exec -it dts-run-dt-core bash 

# launch this data collection when you want to start record data. Press ctrl+C to stop. Data will save at /data/logs/robot_name_date_time.bag at your local computer:
launcher/record_minimal_logs.sh


```

### 2. Model training
```bash
# Build locally on your computer
dts devel build -f

# Copy calibration files from robot to your computer
sudo scp -r duckie@robotname.local:/data/config/calibration/ /data/config/calibration/
```

### 3. Testing
```bash
# Build locally on your computer
dts devel build -f

# Copy calibration files from robot to your computer
sudo scp -r duckie@robotname.local:/data/config/calibration/ /data/config/calibration/
```

## Execution Order

For proper system operation, launch the projects in the following sequence:

1. **Object Detection**
   - Launch and wait until the node publishes initialization message
   - Verify that object detection is running properly

2. **dt-core (Line Detection)**
   - Launch after object detection is initialized
   - Wait until the first line segment is detected
   - Confirm successful line detection

## Troubleshooting

- If calibration copy fails, verify robot's network connection and hostname
- If the bot cannot detect the line segment properly, make sure to change the config file of the line-detector node in dt-core
- Ensure all dependencies are installed before building
- Check robot's logs if any module fails to initialize
- The robot is not publish useful debug functionality to ensure better latency. You can comment proper debug/visualization part if there is issues, whether the bot is not run in the lane or it cannot detect objects properly 

## Notes

- Always maintain the execution order for proper system functionality
- Monitor system resources during operation
