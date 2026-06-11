# okvis_xfeat_frontend — XFeat-on-TensorRT feature frontend bridge.
#
# Bridges mowe_camera_core (capture) → TensorRT (XFeat) → okvis::xfeat features.
# Buildable two ways:
#   * as an OKVIS submodule: add_subdirectory() from the top CMakeLists behind
#     option(USE_MOWE_XFEAT) — Eigen3::Eigen already exists there.
#   * standalone: cmake -S okvis_xfeat_frontend -B build  (finds Eigen3 itself).
#
# mowe_camera_core is found via its vanilla CMake package config (ament-free).
cmake_minimum_required(VERSION 3.16)
project(okvis_xfeat_frontend CXX)

option(USE_TENSORRT "Build with the real TensorRT/CUDA inference path" OFF)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

# --- dependencies ---
find_package(mowe_camera_core CONFIG REQUIRED)

add_library(okvis_xfeat_frontend
  src/XFeatFrontend.cpp
  src/TensorRTEngine.cpp)
add_library(okvis::XFeatFrontend ALIAS okvis_xfeat_frontend)

target_include_directories(okvis_xfeat_frontend
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(okvis_xfeat_frontend
  PUBLIC
    mowe_camera::core)
target_compile_features(okvis_xfeat_frontend PUBLIC cxx_std_17)

if(USE_TENSORRT)
  enable_language(CUDA)
  # Jetson Orin Nano (Ampere) is compute capability 8.7.
  if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 87)
  endif()
  find_package(CUDAToolkit REQUIRED)
  find_library(NVINFER_LIB nvinfer REQUIRED)

  # The CUDA preprocess kernel is only compiled in this configuration.
  target_sources(okvis_xfeat_frontend PRIVATE src/preprocess.cu)
  set_target_properties(okvis_xfeat_frontend PROPERTIES
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")

  target_link_libraries(okvis_xfeat_frontend
    PRIVATE CUDA::cudart ${NVINFER_LIB})
  target_compile_definitions(okvis_xfeat_frontend PRIVATE OKVIS_XFEAT_USE_TENSORRT=1)
  message(STATUS "okvis_xfeat_frontend: TensorRT inference ENABLED (sm_${CMAKE_CUDA_ARCHITECTURES})")
else()
  message(STATUS "okvis_xfeat_frontend: stub mode (no TensorRT) — features empty")
endif()

# Skeleton end-to-end demo (camera → frontend). No ROS.
add_executable(xfeat_frontend_demo src/xfeat_frontend_demo.cpp)
target_link_libraries(xfeat_frontend_demo PRIVATE okvis_xfeat_frontend)
