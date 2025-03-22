cmake_minimum_required(VERSION 3.31.6)

# Explicitly tell CMake where to find CUDA
set(CMAKE_CUDA_COMPILER /usr/local/cuda-12.8/bin/nvcc)

# Set CUDA architectures before defining the project
set(CMAKE_CUDA_ARCHITECTURES 89)

# Define languages with CUDA required
project(DrumForge VERSION 0.1.0 LANGUAGES CXX CUDA)

# Set CUDA-related flags
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set C++ standard for the rest of the code
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)
find_package(GLEW REQUIRED)
find_package(glm REQUIRED)

# Define source files
set(SOURCES
    src/main.cpp
    src/membrane.cpp
    src/shader.cpp
    src/airspace.cpp
    src/camera.cpp
)

# CUDA source files
set(CUDA_SOURCES
    src/membrane_cuda.cu
    src/cuda_test.cu
)

# Define header files
set(HEADERS
    include/membrane.h
    include/shader.h
    include/airspace.h
    include/camera.h
    include/membrane_cuda.h
)

# Enable separate compilation for CUDA
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# Main executable
add_executable(drumforge ${SOURCES} ${HEADERS} ${CUDA_SOURCES})

# CUDA test executable
add_executable(cuda_test src/test_cuda_main.cpp src/cuda_test.cu)

# Include directories
target_include_directories(drumforge PRIVATE 
    ${CMAKE_SOURCE_DIR}/include
    /usr/local/cuda-12.8/include
)

target_include_directories(cuda_test PRIVATE 
    ${CMAKE_SOURCE_DIR}/include
    /usr/local/cuda-12.8/include
)

# Link libraries
target_link_libraries(drumforge PRIVATE 
    OpenGL::GL
    glfw
    GLEW::GLEW
    GLU
    cuda
    cudart
)

target_link_libraries(cuda_test PRIVATE 
    cuda
    cudart
)