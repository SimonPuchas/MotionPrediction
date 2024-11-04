cmake_minimum_required(VERSION 3.0.2)
project(datasetcreator)

find_package(catkin REQUIRED COMPONENTS
  rospy
  gazebo_msgs
  geometry_msgs
  tf
)

catkin_package(
  CATKIN_DEPENDS rospy gazebo_msgs geometry_msgs tf
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(
  PROGRAMS
    src/scripts/robot_controller.py
    src/scripts/data_recorder.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(
  DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)