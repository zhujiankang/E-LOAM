cmake_minimum_required(VERSION 3.0.2)
project(e_loam)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

find_package(OpenMP)

if (OPENMP_FOUND)

set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")


endif()


find_package(catkin REQUIRED COMPONENTS
sensor_msgs
roscpp
std_msgs
nav_msgs
geometry_msgs
pcl_conversions
ndt_omp
tf
)


############### dependencies ######################
# Eigen
include_directories("/usr/include/eigen3")

#PCL
find_package(PCL REQUIRED QUIET)
include_directories(${PCL_INCLUDE_DIRS})

# OpenCV
find_package(OpenCV 3.1 REQUIRED QUIET)
include_directories(${OpenCV_INCLUDE_DIRS})

# Ceres 
find_package(Ceres REQUIRED QUIET)
include_directories(${CERES_INCLUDE_DIRS})

#Gtsam
find_package(GTSAM REQUIRED QUIET)
include_directories(${GTSAM_INCLUDE_DIR})

include_directories(include ${catkin_INCLUDE_DIRS})


set(THIRD_PARTY_LIBS
${OpenCV_LIBS}
${PCL_LIBRARIES}
)
  
catkin_package(
 INCLUDE_DIRS include
)

link_directories(
  include
  ${PCL_LIBRARY_DIRS}
)
add_executable(feature_extraction_node src/feature_extraction.cpp)
target_link_libraries(feature_extraction_node ${THIRD_PARTY_LIBS} ${catkin_LIBRARIES})

add_executable(lidar_odometry_loam_node src/lidar_odometry_loam.cpp )
target_link_libraries(lidar_odometry_loam_node ${THIRD_PARTY_LIBS} ${catkin_LIBRARIES} ${CERES_LIBRARIES})

add_executable(lidar_odometry_ndt_node src/lidar_odometry_ndt.cpp )
add_dependencies(lidar_odometry_ndt_node ndt_omp) #这里的align改成你的app
target_link_libraries(lidar_odometry_ndt_node ${THIRD_PARTY_LIBS} ${catkin_LIBRARIES} ${CERES_LIBRARIES})


add_executable(lidar_mapping_node src/lidar_mapping.cpp)
# add_dependencies(test_feature_extraction_node ndt_omp) #这里的align改成你的app
target_link_libraries(lidar_mapping_node ${THIRD_PARTY_LIBS} gtsam ${catkin_LIBRARIES})