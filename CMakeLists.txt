cmake_minimum_required(VERSION 3.31.6)
project(DrumForge VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find required packages
find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)
find_package(GLEW REQUIRED)
find_package(glm REQUIRED)

# CUDA configuration
set(CMAKE_CUDA_ARCHITECTURES 89) # Adjust based on your GPU architecture
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -O3)

# Define source files
set(SOURCES
    src/main.cpp
    src/membrane.cpp
    src/shader.cpp
    src/airspace.cpp
    src/camera.cpp
)

set(CUDA_SOURCES
    src/membrane_cuda.cu
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
set(CUDA_SEPARABLE_COMPILATION ON)

# Main executable
add_executable(drumforge ${SOURCES} ${HEADERS})

set_source_files_properties(${CUDA_SOURCES} PROPERTIES LANGUAGE CUDA)

# Include directories
target_include_directories(drumforge PRIVATE 
    ${CMAKE_SOURCE_DIR}/include
)

# Link libraries
target_link_libraries(drumforge PRIVATE 
    OpenGL::GL
    glfw
    GLEW::GLEW
    GLU
)

target_link_libraries(drumforge PRIVATE cuda cudart)