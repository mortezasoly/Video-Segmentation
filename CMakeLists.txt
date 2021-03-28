cmake_minimum_required( VERSION 3.0 FATAL_ERROR)
project(segment)
find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIR})
add_executable(segment segmentation.cpp)
target_link_libraries( segment ${OpenCV_LIBS} "${TORCH_LIBRARIES}")
set_property( TARGET segment PROPERTY CXX_STANDARD 14)

