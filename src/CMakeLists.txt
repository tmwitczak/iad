# Set CMAKE main properties
cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

# Create project
project(iad-2a
        VERSION 0.1
        LANGUAGES CXX)

# Set C++ properties
set(CMAKE_CXX_STANDARD 17)

# Set path to Eigen's include directories
set(EIGEN_INCLUDE_DIRECTORIES
        "C:\\Eigen")

# Add include directories
include_directories(
        ${EIGEN_INCLUDE_DIRECTORIES})

# Add sources and create executable
add_executable(iad-2a
        main.cpp
        multi-layer-perceptron.cpp
        multi-layer-perceptron.hpp
        single-layer-perceptron.cpp
        single-layer-perceptron.hpp
        training-example.hpp)
