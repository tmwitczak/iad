# Set CMAKE main properties
cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

# Create project
project(iad-2a
        VERSION 0.1
        LANGUAGES CXX)

# Set C++ properties
set(CMAKE_CXX_STANDARD 17)

# Set path to Cereal's include directories
set(CEREAL_INCLUDE_DIRECTORIES
    "C:/includes")

# Add sources and create executable target
add_executable(iad-2a
               main.cpp
               multi-layer-perceptron.cpp
               multi-layer-perceptron.hpp
               perceptron-layer.cpp
               perceptron-layer.hpp
               training-example.hpp activation-function.cpp activation-function.hpp cloneable.hpp sigmoid.cpp sigmoid.hpp rectified-linear-unit.cpp rectified-linear-unit.hpp parametric-rectified-linear-unit.cpp parametric-rectified-linear-unit.hpp)

# Add Python
find_package(Python 3.7.3 REQUIRED COMPONENTS Development)
target_link_libraries(iad-2a ${Python_LIBRARIES})
target_include_directories(iad-2a PRIVATE ${Python_INCLUDE_DIRS})

# Add Eigen3
find_package (Eigen3 3.3.7 REQUIRED NO_MODULE)
target_link_libraries (iad-2a Eigen3::Eigen)


# Add include directories
target_include_directories(iad-2a PRIVATE
                           ${CEREAL_INCLUDE_DIRECTORIES}
                           + ${CEREAL_INCLUDE_DIRECTORIES}/archives)