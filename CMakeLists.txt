cmake_minimum_required(VERSION 3.18...3.21)

project(XoscarCollective)
if(DEFINED PYTHON_PATH)
  set(Python_EXECUTABLE "${PYTHON_PATH}")
endif()
if(Python_FIND_ABI MATCHES "ON$")
  cmake_minimum_required(VERSION 3.30)
  # Windows headers do not define this for the free-threaded build.
  add_compile_definitions(Py_GIL_DISABLED=1)
endif()
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
set(PYBIND11_FINDPYTHON ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")

if(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
  option(USE_LIBUV "Build tcp transport on linux" OFF)
else()
  option(USE_LIBUV "Build libuv transport on others" ON)
endif()

if(MSVC)
  add_compile_options(/utf-8)
  message(STATUS "Done setting /utf-8 for MSVC")
endif()

include_directories(${CMAKE_SOURCE_DIR})
# Set include directories
include_directories(${Python_INCLUDE_DIRS})
if(${CMAKE_SYSTEM_NAME} MATCHES "Windows")
  set(libuv_dir ${CMAKE_SOURCE_DIR}/third_party/libuv/build/uvlib)
  if(NOT EXISTS ${libuv_dir})
    execute_process(
      COMMAND
        cmd /c
        "echo %cd% && cd ..\\..\\..\\..\\..\\third_party\\libuv && mkdir build && cd build && mkdir uvlib && cmake .. -DCMAKE_INSTALL_PREFIX=uvlib -DCMAKE_POLICY_VERSION_MINIMUM=3.10 && msbuild.exe INSTALL.vcxproj"
    )
    message(STATUS "Done creating libuv_dir = ${libuv_dir}")
  endif()
  set(libuv_ROOT ${CMAKE_SOURCE_DIR}/third_party/libuv/build/uvlib)
  set(uv_HEADER_PATH ${CMAKE_SOURCE_DIR}/third_party/libuv/include)
  include_directories(${uv_HEADER_PATH})
  #copy uv.dll to /python/xoscar/collective
  file(COPY ${CMAKE_SOURCE_DIR}/third_party/libuv/build/uvlib/bin/uv.dll
       DESTINATION ${CMAKE_SOURCE_DIR}/python/xoscar/collective)
  add_definitions(-DNOMINMAX)
endif()

add_subdirectory(third_party/fmt)
add_subdirectory(third_party/pybind11)
add_subdirectory(third_party/gloo)

# set c++11 for gloo
set_target_properties(
  gloo
  PROPERTIES CXX_STANDARD 11
             CXX_STANDARD_REQUIRED ON
             CXX_EXTENSIONS OFF)
if(NOT DEFINED BUILD_TMP_DIR)
  file(GLOB TMP_DIRS "python/build/temp*")
  foreach(TMP_DIR ${TMP_DIRS})
    set(BUILD_TMP_DIR ${TMP_DIR}/xoscar_pygloo)
  endforeach()
else()
  set(BUILD_TMP_DIR python/${BUILD_TMP_DIR})
endif()
# copy config.h to cpp/gloo/include
file(COPY ${BUILD_TMP_DIR}/third_party/gloo/gloo/config.h
     DESTINATION ${CMAKE_SOURCE_DIR}/cpp/collective/gloo/include)

add_subdirectory(cpp)
