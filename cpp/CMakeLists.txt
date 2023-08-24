cmake_minimum_required(VERSION 3.11...3.21)

project(XoscarCollective)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
file(GLOB TMP_DIRS "../python/build/lib*")
foreach(TMP_DIR ${TMP_DIRS})
  message(${TMP_DIR})
  set(LIBRARY_OUTPUT_DIRECTORY ${TMP_DIR}/xoscar/collective)
endforeach()

include_directories(${CMAKE_SOURCE_DIR}/cpp/collective/rendezvous/include)
include_directories(${CMAKE_SOURCE_DIR}/cpp/collective/gloo/include)
include_directories(../third_party/gloo)
include_directories(../third_party/fmt/include)

add_subdirectory(collective/rendezvous)
add_subdirectory(collective/gloo)

pybind11_add_module(xoscar_pygloo collective/gloo/main.cc)
if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  include(FindPkgConfig)
  pkg_search_module(libuv REQUIRED libuv>=1.26)
  find_file(
    libuv_LIBRARY
    NAMES libuv.a libuv_a.a
    PATHS ${libuv_LIBDIR}
    NO_DEFAULT_PATH)
  if(NOT EXISTS ${libuv_LIBRARY})
    message(FATAL_ERROR "Unable to find static libuv library in " ${libuv_LIBDIR})
  endif()
  add_library(uv_s INTERFACE IMPORTED)
  set_target_properties(uv_s PROPERTIES INTERFACE_INCLUDE_DIRECTORIES ${libuv_INCLUDE_DIRS}
                                        INTERFACE_LINK_LIBRARIES ${libuv_LIBRARY})
  target_link_libraries(xoscar_pygloo PRIVATE GlooLib gloo StoreLib fmt::fmt uv_s)
else()
  target_link_libraries(xoscar_pygloo PRIVATE GlooLib gloo StoreLib fmt::fmt)
endif()
set_target_properties(xoscar_pygloo PROPERTIES LIBRARY_OUTPUT_DIRECTORY
                                               ${LIBRARY_OUTPUT_DIRECTORY})
