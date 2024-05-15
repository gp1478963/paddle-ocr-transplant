cmake_minimum_required(VERSION 3.13)
project(paddlev2_inference)

set(CMAKE_CXX_STANDARD 11)

option(WITH_SYSTEM_OPENCV_LIB "是否使用系统的opencv库，如果为OFF，则编译dep/opencv-4.8.1.tar.gz, 默认OFF" OFF)
option(BUILD_SPLIT_BIN_LIB "是否分开编译动态库与可执行文件，分开编译用于打包发布动态库，一起编译用于动态库调试，默认为ON" ON)

if (${WITH_SYSTEM_OPENCV_LIB})
    #set(OpenCV_DIR "/home/hk/gp/opencv-4.8.1/build_tree")
    set(OpenCV_DIR "D:\\libs\\cv\\opencv\\build\\x64\\vc16\\lib")
    #set(OpenCV_DIR "/home/hk/opencv-4.8.1/oo")
    find_package(OpenCV REQUIRED)
    link_directories("D:\\cv\\opencv\\build\\x64\\vc16\\lib")
    include_directories("/home/hk/opencv-4.8.1/build_true/include")
    message(STATUS ${OpenCV_LIBS})
else()
    set(OPENCV_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR}/external/cv2)

    include_directories($<IF:$<STREQUAL:${CMAKE_HOST_SYSTEM_NAME},Linux>,${OPENCV_BUILD_DIR}/include/opencv4,${OPENCV_BUILD_DIR}/include>)
    link_directories($<IF:$<STREQUAL:${CMAKE_HOST_SYSTEM_NAME},Linux>,${OPENCV_BUILD_DIR}/lib,${OPENCV_BUILD_DIR}/x86/vc15/bin>)
    link_directories($<IF:$<STREQUAL:${CMAKE_HOST_SYSTEM_NAME},Linux>,${OPENCV_BUILD_DIR}/lib,${OPENCV_BUILD_DIR}/x86/vc15/lib>)
endif ()


if(UNIX)
    set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
    set(CMAKE_INSTALL_RPATH ".")
endif()


# 设置 OpenCV 静态库的路径
#set(OpenCV_DIR ${CMAKE_BINARY_DIR}/opencv/src/opencv-build)
#find_package(OpenCV REQUIRED NO_MODULE PATHS ${OpenCV_DIR} NO_DEFAULT_PATH)
#include_directories(${OpenCV_INCLUDE_DIRS})

if (${BUILD_SPLIT_BIN_LIB})
    add_library(paddlev2_inference SHARED
            predict_ocr.cpp
            predict_config.hpp
            predict_detection.hpp
            predict_recogntion.hpp
            clipper/clipper.cpp
            predict_algorithm.hpp
            predict_interface.h )
else()
    add_executable(paddlev2_inference
            predict_ocr.cpp
            predict_config.hpp
            predict_detection.hpp
            predict_recogntion.hpp
            clipper/clipper.cpp
            predict_algorithm.hpp
            predict_interface.h )
    target_compile_definitions(paddlev2_inference PRIVATE PADDLE_OCR_EXPORT_MAIN)
endif ()



MESSAGE("WITH_SYSTEM_OPENCV_LIB:${WITH_SYSTEM_OPENCV_LIB}")

if (${WITH_SYSTEM_OPENCV_LIB})
else()
    add_subdirectory(dep)
    add_dependencies(paddlev2_inference opencv_project)
endif()

#target_link_libraries(paddlev2_inference PRIVATE ${OpenCV_LIBS})
target_include_directories(paddlev2_inference PRIVATE
        #${OpenCV_INCLUDE_DIRS}
        polyclipping)

if (${WITH_SYSTEM_OPENCV_LIB})
    target_include_directories(paddlev2_inference PUBLIC  ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(paddlev2_inference PUBLIC ${OpenCV_LIBS})
endif ()

target_compile_definitions(paddlev2_inference PRIVATE DEEPNETAPI_EXPORT  USE_STDCOUT_STREAM)



if (${BUILD_SPLIT_BIN_LIB})
    message(STATUS "BUILD paddlev2_inference_bin")
add_executable(paddlev2_inference_bin predict_interface.h main.cpp)
target_link_libraries(paddlev2_inference_bin PRIVATE paddlev2_inference)
target_compile_definitions(paddlev2_inference_bin PRIVATE USE_STDCOUT_STREAM)
target_link_libraries(paddlev2_inference_bin PRIVATE ${OpenCV_LIBS})
endif ()
