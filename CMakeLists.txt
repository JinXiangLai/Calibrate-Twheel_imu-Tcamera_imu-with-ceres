cmake_minimum_required(VERSION 3.10)
project(IMU_Wheel_Calibration)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 14)

set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O2 -Wall -g -ggdb")

IF (NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF ()

# 查找依赖库
find_package(Eigen3 REQUIRED)
find_package(Ceres 2 REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(
  ${OpenCV_INCLUDE_DIRS}
  ${Eigen_INCLUDE_DIRS}
  ${Ceres_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}
)

# 预先指定链接库
link_libraries(
  Eigen3::Eigen
  #ceres::ceres # 这样写就行不通
  Ceres::ceres # 若是使用ceres会与opencv的冲突，所以必须要命名空间，这里的命名空间就是存放libceres.a的文件夹名字
  ${OpenCV_LIBS}
)

# 添加可执行文件
add_executable(imu_wheel_calibration imuWheelCalibrate.cpp)
add_executable(imu_camera_calibration TcbCalibrate.cpp)
add_executable(optimizaLandmarkPosition optimizaLandmarkPosition.cpp)
add_executable(larndmark_radius_to_evaluate_depth larndmark_radius_to_evaluate_depth.cpp)