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
    "C:/Eigen")

# Set path to Cereal's include directories
set(CEREAL_INCLUDE_DIRECTORIES
    "C:/includes")

# Add sources and create executable
add_executable(iad-2a
               main.cpp
               multi-layer-perceptron.cpp
               multi-layer-perceptron.hpp
               perceptron-layer.cpp
               perceptron-layer.hpp
               training-example.hpp activation-function.cpp activation-function.hpp cloneable.hpp sigmoid.cpp sigmoid.hpp rectified-linear-unit.cpp rectified-linear-unit.hpp parametric-rectified-linear-unit.cpp parametric-rectified-linear-unit.hpp)

# Add include directories
target_include_directories(iad-2a PRIVATE
                           ${EIGEN_INCLUDE_DIRECTORIES}
                           + ${CEREAL_INCLUDE_DIRECTORIES}
                           + ${CEREAL_INCLUDE_DIRECTORIES}/archives)