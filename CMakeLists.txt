cmake_minimum_required(VERSION 3.11...3.21)

project(XoscarCollective)

if(NOT DEFINED PYTHON_PATH)
  find_package(Python COMPONENTS Interpreter Development)
else()
  set(PYTHON_EXECUTABLE ${PYTHON_PATH})
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")

include_directories(${CMAKE_SOURCE_DIR})

add_subdirectory(third_party/fmt)
add_subdirectory(third_party/pybind11)
add_subdirectory(cpp)
