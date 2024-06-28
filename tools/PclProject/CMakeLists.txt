cmake_minimum_required(VERSION 3.10)
project(PointCloudProject VERSION 1.0)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find packages
find_package(PkgConfig REQUIRED)
pkg_check_modules(OPENCV REQUIRED opencv4)
find_package(PCL 1.8 REQUIRED COMPONENTS common io visualization segmentation surface)

# Include directories
include_directories(${OPENCV_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS})

# Definitions and compiler flags
add_definitions(${PCL_DEFINITIONS} ${OPENCV_CFLAGS_OTHER})

# Specify the target executable
add_executable(${PROJECT_NAME} main.cpp)

# Link libraries
target_link_libraries(${PROJECT_NAME} ${OPENCV_LIBRARIES} ${PCL_LIBRARIES})

# Ensure the compiler knows where to find the header files and link libraries
target_include_directories(${PROJECT_NAME} PUBLIC ${OPENCV_INCLUDE_DIRS})

