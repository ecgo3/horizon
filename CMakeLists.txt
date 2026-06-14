cmake_minimum_required(VERSION 3.24)

project(horizon
  VERSION 0.1.0
  DESCRIPTION "CUDA/OpenGL Schwarzschild black hole particle and lensing engine"
  LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(HORIZON_ENABLE_DOUBLE_DOUBLE "Enable double-double horizon diagnostics in CUDA code" OFF)
option(HORIZON_BUILD_VALIDATION "Build the Schwarzschild validation suite" ON)

find_package(CUDAToolkit REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(glfw3 REQUIRED)

if(TARGET glfw)
  set(HORIZON_GLFW_TARGET glfw)
elseif(TARGET glfw3::glfw)
  set(HORIZON_GLFW_TARGET glfw3::glfw)
else()
  message(FATAL_ERROR "glfw3 was found, but no usable imported target was exported")
endif()

add_library(horizon_core
  src/cuda/SchwarzschildKernels.cu
  src/render/Camera.cpp
  src/render/ShaderProgram.cpp
  src/io/CsvExporter.cpp
  src/CudaSimulation.cpp)

target_include_directories(horizon_core
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src)

target_link_libraries(horizon_core
  PUBLIC
    CUDA::cudart
    CUDA::cuda_driver
    OpenGL::GL
    GLEW::GLEW
    ${HORIZON_GLFW_TARGET})

target_compile_features(horizon_core PUBLIC cxx_std_20)

if(HORIZON_ENABLE_DOUBLE_DOUBLE)
  target_compile_definitions(horizon_core PUBLIC HORIZON_ENABLE_DOUBLE_DOUBLE=1)
endif()

set_target_properties(horizon_core PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
  CUDA_ARCHITECTURES "75;80;86;89;90")

target_compile_options(horizon_core PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --prec-div=true --prec-sqrt=true -lineinfo>
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>)

add_executable(horizon src/main.cpp)
target_link_libraries(horizon PRIVATE horizon_core)

if(HORIZON_BUILD_VALIDATION)
  enable_testing()
  add_executable(horizon_validation tests/validation.cu)
  target_include_directories(horizon_validation PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_link_libraries(horizon_validation PRIVATE CUDA::cudart)
  set_target_properties(horizon_validation PROPERTIES
    CUDA_ARCHITECTURES "75;80;86;89;90")
  add_test(NAME schwarzschild_validation COMMAND horizon_validation)
endif()

add_custom_command(TARGET horizon POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders
    $<TARGET_FILE_DIR:horizon>/shaders)
