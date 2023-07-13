cmake_minimum_required(VERSION 3.11...3.21)

project(XoscarCollective)

if(NOT DEFINED PYTHON_PATH)
  find_package(Python COMPONENTS Interpreter Development)
else()
  set(PYTHON_EXECUTABLE ${PYTHON_PATH})
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")

if(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
  option(USE_LIBUV "Build tcp transport on linux" OFF)
else()
  option(USE_LIBUV "Build libuv transport on others" ON)
endif()

include_directories(${CMAKE_SOURCE_DIR})
#find python3 include dir
execute_process(COMMAND python -c "import sysconfig; print(sysconfig.get_path('include'))"
                OUTPUT_VARIABLE PYTHON_INCLUDE_PATH)

# Set include directories
include_directories(${PYTHON_INCLUDE_PATH})

add_subdirectory(third_party/fmt)
add_subdirectory(third_party/pybind11)
add_subdirectory(third_party/gloo)

# set c++11 for gloo
set_target_properties(
  gloo
  PROPERTIES CXX_STANDARD 11
             CXX_STANDARD_REQUIRED ON
             CXX_EXTENSIONS OFF)

# copy config.h to cpp/gloo/include
file(COPY python/${BUILD_TMP_DIR}/third_party/gloo/gloo/config.h
     DESTINATION ${CMAKE_SOURCE_DIR}/cpp/collective/gloo/include)
add_subdirectory(cpp)
