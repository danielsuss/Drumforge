cmake_minimum_required(VERSION 3.31.6)
project(DrumForge VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Main executable
add_executable(drumforge src/main.cpp)