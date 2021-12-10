cmake_minimum_required(VERSION 3.20)
project(slam)

set(CMAKE_CXX_STANDARD 20)

find_package(Threads REQUIRED)
find_package( OpenCV REQUIRED )
include_directories(
        "include/"
        ${OpenCV_INCLUDE_DIRS}
        "/usr/include/python3.9"
        "/usr/include/matplotlib-cpp"
        "/usr/lib/python3.9/site-packages/numpy/core/include"
)

add_executable(slam vfc.cpp
        src/main.cpp
        src/Frame.cpp
        src/FeatureDetector.cpp
        src/FeatureMatcher.cpp
        src/MatchFilterer.cpp
        src/PoseEstimator.cpp
        src/OpenCVVisualizer.cpp src/FramePreprocessor.cpp include/Helpers.hpp)

add_executable(old Ransac.hpp vfc.cpp main4.cpp Frame.cpp)
target_link_libraries(slam ${OpenCV_LIBS} python3.9 Threads::Threads)
target_link_libraries(old ${OpenCV_LIBS} python3.9)

