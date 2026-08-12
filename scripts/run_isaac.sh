#!/usr/bin/env bash
nohup env -u PYTHONPATH ROS_DISTRO=humble RMW_IMPLEMENTATION=rmw_fastrtps_cpp LD_LIBRARY_PATH=/home/a/isaacsim/exts/isaacsim.ros2.bridge/humble/lib /home/a/isaacsim/isaac-sim.sh > "$HOME/Desktop/shihoon/isaac_bunker_project/logs/isaac_sim.log" 2>&1 < /dev/null &
