# //////////////////////////////////////////////// | Set CMake main properties #
cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

# /////////////////////////////////////////////////////////// | Create project #
project(iad-2a
        VERSION 0.1
        LANGUAGES CXX)

# Set C++ properties
set(CMAKE_CXX_STANDARD 17)

# Add sources and create executable target
add_executable(iad-2a
               main.cpp
               multi-layer-perceptron.cpp
               multi-layer-perceptron.hpp
               perceptron-layer.cpp
               perceptron-layer.hpp
               training-example.hpp
               activation-function.cpp
               activation-function.hpp
               cloneable.hpp
               sigmoid.cpp
               sigmoid.hpp
               rectified-linear-unit.cpp
               rectified-linear-unit.hpp
               parametric-rectified-linear-unit.cpp
               parametric-rectified-linear-unit.hpp)

set_target_properties(iad-2a PROPERTIES
                      RUNTIME_OUTPUT_DIRECTORY ${CMAKE_HOME_DIRECTORY})

#target_compile_options(iad-2a -Wall -Wextra -Wpedantic -Werror)

# Add Eigen3
find_package(Eigen3 REQUIRED)
target_link_libraries(iad-2a Eigen3::Eigen)

# Add cereal
find_package(cereal REQUIRED)
target_link_libraries(iad-2a cereal)