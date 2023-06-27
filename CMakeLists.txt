cmake_minimum_required(VERSION 3.11...3.21)

project(XoscarCollective)

if(NOT ${PYTHON_EXECUTABLE})
  find_package(Python COMPONENTS Interpreter Development)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")

include_directories(${CMAKE_SOURCE_DIR})

add_subdirectory(third_party/fmt)
add_subdirectory(third_party/pybind11)
add_subdirectory(cpp)
