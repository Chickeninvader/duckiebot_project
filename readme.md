# Duckiebot Multi-Project Setup Guide

## Project Overview

This repository contains three interconnected projects for Duckiebot:
1. Object Detection
2. Line Detection (dt-core)
3. My ROS Project (core bot execution)

## Prerequisites

- Duckiebot running and connected to your network
- DTS (Duckietown Shell) installed
- SSH access to your Duckiebot
- ROS environment properly configured

## Build Instructions

### 1. Object Detection Project
```bash
# Build on the robot
dts devel build -H robotname -f
```

### 2. Line Detection (dt-core)
```bash
# Build locally on your computer
dts devel build -f

# Copy calibration files from robot to your computer
sudo scp -r duckie@robotname.local:/data/config/calibration/ /data/config/calibration/
```

### 3. My ROS Project
```bash
# Build on the robot
dts devel build -H robotname -f
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

3. **my-ros-project**
   - Launch last after both object and line detection are running
   - This is the core bot execution module

## Troubleshooting

- If calibration copy fails, verify robot's network connection and hostname
- Ensure all dependencies are installed before building
- Check robot's logs if any module fails to initialize

## Notes

- Always maintain the execution order for proper system functionality
- Monitor system resources during operation

## Issues:
