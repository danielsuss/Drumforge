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

# ImGui source files
set(IMGUI_SOURCES
    vendor/imgui/imgui.cpp
    vendor/imgui/imgui_demo.cpp
    vendor/imgui/imgui_draw.cpp
    vendor/imgui/imgui_tables.cpp
    vendor/imgui/imgui_widgets.cpp
    vendor/imgui/backends/imgui_impl_glfw.cpp
    vendor/imgui/backends/imgui_impl_opengl3.cpp
)

# Legacy header files
set(HEADERS_LEGACY
    legacy/include/membrane.h
    legacy/include/shader.h
    legacy/include/airspace.h
    legacy/include/camera.h
)

# Legacy source files
set(SOURCES_LEGACY
    legacy/src/main.cpp
    legacy/src/membrane.cpp
    legacy/src/shader.cpp
    legacy/src/airspace.cpp
    legacy/src/camera.cpp
)

# CUDA test source files
set(SOURCES_CUDA_TEST
    cuda_test/src/cuda_test.cu
    cuda_test/src/test_cuda_main.cpp
)

# Parallel header files
set(HEADERS_PARALLEL
    parallel/include/cuda_memory_manager.h
    parallel/include/component_interface.h
    parallel/include/simulation_manager.h
    parallel/include/cuda_gl_buffer.h
    parallel/include/membrane_component.h
    parallel/include/membrane_kernels.cuh
    parallel/include/camera.h
    parallel/include/input_handler.h
    parallel/include/visualization_manager.h
    parallel/include/gui_manager.h
    parallel/include/air_cavity_component.h
    parallel/include/air_cavity_kernels.cuh
)

# Parallel source files
set(SOURCES_PARALLEL
    parallel/src/main.cpp
    parallel/src/cuda_memory_manager.cu
    parallel/src/simulation_manager.cpp
    parallel/src/cuda_gl_buffer.cu
    parallel/src/membrane_kernels.cu
    parallel/src/membrane_component.cpp
    parallel/src/camera.cpp
    parallel/src/input_handler.cpp
    parallel/src/visualization_manager.cpp
    parallel/src/gui_manager.cpp
    parallel/src/air_cavity_component.cpp
)

# Legacy executable
add_executable(drumforge_legacy ${SOURCES_LEGACY} ${HEADERS_LEGACY})

# CUDA test executable
add_executable(cuda_test ${SOURCES_CUDA_TEST})

# Parallel executable
add_executable(drumforge_parallel ${SOURCES_PARALLEL} ${HEADERS_PARALLEL} ${IMGUI_SOURCES})

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
    ${CMAKE_SOURCE_DIR}/vendor/imgui
    ${CMAKE_SOURCE_DIR}/vendor/imgui/backends
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