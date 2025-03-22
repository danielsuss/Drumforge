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

# Enable separate compilation for CUDA
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# Set C++ standard for the rest of the code
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)
find_package(GLEW REQUIRED)
find_package(glm REQUIRED)

# Legacy source files
set(SOURCES_LEGACY
    legacy/src/main.cpp
    legacy/src/membrane.cpp
    legacy/src/shader.cpp
    legacy/src/airspace.cpp
    legacy/src/camera.cpp
)

# Legacy header files
set(HEADERS_LEGACY
    legacy/include/membrane.h
    legacy/include/shader.h
    legacy/include/airspace.h
    legacy/include/camera.h
)

# CUDA test source files
set(SOURCES_CUDA_TEST
    cuda_test/src/cuda_test.cu
    cuda_test/src/test_cuda_main.cpp
)

# Parallel source files
set(SOURCES_PARALLEL
    parallel/src/main.cpp
    parallel/src/cuda_memory_manager.cu
    parallel/src/simulation_manager.cpp
)

# Parallel header files
set(HEADERS_PARALLEL
    parallel/include/cuda_memory_manager.h
    parallel/include/component_interface.h
    parallel/include/simulation_manager.h
)

# Legacy executable
add_executable(drumforge_legacy ${SOURCES_LEGACY} ${HEADERS_LEGACY})

# CUDA test executable
add_executable(cuda_test ${SOURCES_CUDA_TEST})

# Parallel executable
add_executable(drumforge_parallel ${SOURCES_PARALLEL} ${HEADERS_PARALLEL})

# Legacy include directories
target_include_directories(drumforge_legacy PRIVATE 
    ${CMAKE_SOURCE_DIR}/legacy/include
)

# CUDA test include directories
target_include_directories(cuda_test PRIVATE
    /usr/local/cuda-12.8/include
)

# Parallel include directories
target_include_directories(drumforge_parallel PRIVATE
    ${CMAKE_SOURCE_DIR}/parallel/include
    /usr/local/cuda-12.8/include
)

# Legacy link libraries
target_link_libraries(drumforge_legacy PRIVATE 
    OpenGL::GL
    glfw
    GLEW::GLEW
    GLU
)

# CUDA test link libraries
target_link_libraries(cuda_test PRIVATE 
    cuda
    cudart
)

# Parallel link libraries
target_link_libraries(drumforge_parallel PRIVATE 
    cuda
    cudart
    OpenGL::GL
    GLEW::GLEW
    glfw
)