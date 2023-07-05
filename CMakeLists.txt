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
#find python3 include dir
execute_process(
  COMMAND python3-config --includes
  OUTPUT_VARIABLE PYTHON_INCLUDE_DIRS
  OUTPUT_STRIP_TRAILING_WHITESPACE)
# Split the string
string(REGEX MATCHALL "[^ ]+" STR_LIST "${PYTHON_INCLUDE_DIRS}")
list(GET STR_LIST 0 FIRST_PATH)
string(SUBSTRING ${FIRST_PATH} 2 -1 PYTHON_INCLUDE)
# Set include directories
include_directories(${PYTHON_INCLUDE})

add_subdirectory(third_party/fmt)
add_subdirectory(third_party/pybind11)
add_subdirectory(third_party/gloo)
add_subdirectory(cpp)
