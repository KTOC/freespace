# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.

cmake_minimum_required(VERSION 2.8.12 FATAL_ERROR)

#-------------------------------------------------------------------------------
# Set basic configurations
#-------------------------------------------------------------------------------
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include(SamplesSetBuildType) # Set the build type before project is created

set(SDK_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(SDK_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

#-------------------------------------------------------------------------------
# Root project
#-------------------------------------------------------------------------------
project(DriveworksSDK-Samples C CXX)

#-------------------------------------------------------------------------------
# System packages
#-------------------------------------------------------------------------------
find_package(CUDA REQUIRED)
find_package(Threads REQUIRED)
find_package(TensorRT REQUIRED)
find_package(CuDNN REQUIRED)

#-------------------------------------------------------------------------------
# Basic configuration
#-------------------------------------------------------------------------------
include(ArchConfiguration)
include(CommonConfiguration)
include(SamplesConfiguration)
include(Samples3rdparty)

include(SamplesInstallConfiguration)
include(ExpandDependencyTree)
include(IncludeTargetDirectories)
include(UploadVibrante)
set(SDK_BINARY_DIR ${CMAKE_BINARY_DIR})

#-------------------------------------------------------------------------------
# Driveworks SDK
#-------------------------------------------------------------------------------
find_package(Driveworks REQUIRED)
include_directories(${Driveworks_INCLUDE_DIR})

list(APPEND Driveworks_LIBRARIES ${CUDA_LIBRARIES} ${CUDA_cublas_LIBRARY})

if(LINUX)
    list(APPEND Driveworks_LIBRARIES ${CMAKE_THREAD_LIBS_INIT} dl rt)
elseif(VIBRANTE)
    list(APPEND Driveworks_LIBRARIES ${vibrante_LIBRARIES} ${vibrante_Xlibs_LIBRARIES} nvmedia ${EGL_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})
endif()

#-------------------------------------------------------------------------------
# Samples
#-------------------------------------------------------------------------------
if(CMAKE_CROSSCOMPILING)
  set(DRIVEWORKS_DATAPATH "../data")
else()
  set(DRIVEWORKS_DATAPATH "${CMAKE_CURRENT_LIST_DIR}/../data")
endif()
include_directories(${SDK_BINARY_DIR}/configured/samples)
configure_file(src/framework/DataPath.hpp.in
               ${SDK_BINARY_DIR}/configured/samples/framework/DataPath.hpp)

           #set(SAMPLES framework;egomotion;sensors;features;mapping;rigconfiguration;renderer;sfm;dnn;laneDetection;colorcorrection;isp;rectifier;ipc;hello_world;image;stereo;freespace;drivenet;maps;template;icp;lidar_accumulator;calibration;vehicleio)
set(SAMPLES framework;newfreespace)
#if dnn is needed

file(GLOB ADDITIONAL_SAMPLES "CMakeListsSamples*.txt")
foreach(ADDITIONAL_SAMPLE ${ADDITIONAL_SAMPLES})
  include(${ADDITIONAL_SAMPLE})
endforeach()

foreach(SAMPLE ${SAMPLES})
    add_subdirectory(src/${SAMPLE})
endforeach()

if(WIN32)
  # Handle runtime library dependence
  get_filename_component(DriveworksLibPath ${Driveworks_LIBRARY} DIRECTORY)
  set(LIBRARY_DIR ${DriveworksLibPath}/../bin)

  file(GLOB dll_files "${LIBRARY_DIR}/*.dll")
  foreach(file ${dll_files})
        file(COPY "${file}" DESTINATION "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Debug")
        file(COPY "${file}" DESTINATION "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Release")
        install(FILES ${LIBRARY_DIR}/${file}
                DESTINATION "${CMAKE_INSTALL_PREFIX}/${SDK_SAMPLE_DESTINATION}"
        )
  endforeach()
endif()

#-------------------------------------------------------------------------------
# Cuda dependencies
#-------------------------------------------------------------------------------
# Create a target to update cuda dependencies
add_cuda_dependencies_target(update_cuda_deps ON)
set_property(TARGET update_cuda_deps PROPERTY FOLDER "Support")
