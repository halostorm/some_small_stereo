cmake_minimum_required(VERSION 2.8.3)
project(640_stereo)

# 设置编译模式
set( CMAKE_BUILD_TYPE "Release" )
set( CMAKE_CXX_FLAGS "-std=c++11" )

find_package( OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable( 640_stereo 640_stereo.cc)
target_link_libraries( 640_stereo ${OpenCV_LIBS})











