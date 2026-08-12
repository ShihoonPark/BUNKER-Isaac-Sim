# BUNKER Isaac Sim

Isaac Sim based simulation, trajectory planning, and control project for the AgileX BUNKER MINI 2.0 tracked robot.

## Current Environment

- Ubuntu 22.04.5 LTS
- NVIDIA GeForce RTX 3080 10GB
- NVIDIA Driver 580.173.02
- NVIDIA Isaac Sim 5.1.0
- ROS 2 Humble

## Current Progress

- Isaac Sim startup environment configured
- ROS 2 Bridge verified with Isaac Sim internal Humble libraries
- PhysX rigid-body and collision test completed
- Base physics scene created
- AgileX BUNKER MINI URDF/STL asset inspected and imported into Isaac Sim
- BUNKER track geometry analysis in progress

## Project Structure

~~~text
isaac_bunker_project/
├── scenes/
│   └── base_physics.usd
├── scripts/
│   └── run_isaac.sh
├── assets/
└── logs/
~~~

## External BUNKER Asset

The AgileX simulation repository is not committed to this repository. Clone it under `assets/` when needed:

~~~bash
cd assets
git clone https://github.com/agilexrobotics/ugv_gazebo_sim.git
~~~

## Launch Isaac Sim

~~~bash
./scripts/run_isaac.sh
~~~

## Notes

- Large ROS bags, point clouds, logs, downloaded external repositories, and generated Isaac Sim imports are excluded from Git.
- The imported BUNKER URDF contains a single rigid `base_link`; tracked-vehicle drive/contact modeling is being developed separately.
