include(ExternalProject)

ExternalProject_Add(
        opencv_project
        PREFIX ${CMAKE_BINARY_DIR}/external
        URL ${CMAKE_CURRENT_SOURCE_DIR}/opencv-4.8.1.tar.gz
        CMAKE_ARGS
        -DBUILD_opencv_world=ON
        -DBUILD_opencv_calib3d=OFF
        -DBUILD_opencv_contrib=OFF
        -DBUILD_opencv_features2d=OFF
        -DBUILD_opencv_ml=OFF
        -DBUILD_opencv_video=OFF
        -DBUILD_opencv_videoio=OFF
        -DBUILD_opencv_flann=OFF
        -DBUILD_opencv_gpu=OFF
        -DBUILD_opencv_gapi=OFF
        -DBUILD_opencv_highgui=OFF
        -DBUILD_opencv_photo=OFF
        -DBUILD_opencv_gpu=OFF
        -DOPENCV_FORCE_3RDPARTY_BUILD=ON
#        -DBUILD_SHARED_LIBS=OFF
        -DOPJ_STATIC=ON
        -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/external/cv2
        BUILD_ALWAYS 1

)


# 假设 OpenCV 被构建在以下目录
set(OPENCV_BUILD_DIR ${CMAKE_BINARY_DIR}/external/cv2)

include_directories(${OPENCV_BUILD_DIR}/include)
link_directories($<IF:$<STREQUAL:${CMAKE_HOST_SYSTEM_NAME},Linux>,${OPENCV_BUILD_DIR}/lib,${OPENCV_BUILD_DIR}/x86/vc15/bin>)
target_link_libraries(paddlev2_inference PRIVATE $<IF:$<STREQUAL:${CMAKE_HOST_SYSTEM_NAME},Linux>,opencv_world,opencv_world481>)

# 添加你的项目的可执行文件
#add_executable(my_project main.cpp)
#
## 将 OpenCV 作为依赖链接到你的项目
#add_dependencies(my_project my_opencv_project)
#target_link_libraries(my_project PRIVATE opencv_core opencv_imgproc opencv_highgui)
