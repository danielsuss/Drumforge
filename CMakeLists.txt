cmake_minimum_required(VERSION 3.31.6)
project(DrumForge VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)

# Define source files
set(SOURCES
    src/main.cpp
    src/membrane.cpp
)

# Define header files
set(HEADERS
    include/membrane.h
)

# Main executable
add_executable(drumforge ${SOURCES} ${HEADERS})

# Include directories
target_include_directories(drumforge PRIVATE 
    ${CMAKE_SOURCE_DIR}/include
)

# Link libraries
target_link_libraries(drumforge PRIVATE 
    OpenGL::GL
    glfw
)