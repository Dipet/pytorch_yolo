cmake_minimum_required(VERSION 3.5)

set(PROJECT_NAME new_pedestrian_tracker)
project(${PROJECT_NAME} LANGUAGES CXX)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package (Threads)
set (CMAKE_CXX_FLAGS -pthread)

file (GLOB_RECURSE SOURCES *.c*)
file (GLOB_RECURSE HEADERS *.h)

find_package(OpenCV REQUIRED)
if(NOT OpenCV_FOUND)
    message(WARNING "OPENCV is disabled or not found, " ${PROJECT_NAME} " skipped")
    return()
endif()

add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} stdc++fs)

set(InferenceEngine_DIR "/home/druzhinin/HDD/GitRepos/dldt/inference-engine/build_cpu")
find_package(InferenceEngine)
include_directories(${InferenceEngine_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${InferenceEngine_LIBRARIES} IE::ie_cpu_extension)

