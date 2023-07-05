cmake_minimum_required(VERSION 3.11...3.21)

project(XoscarCollective)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
if(NOT DEFINED LIBRARY_OUTPUT_DIRECTORY_STORE)
  set(LIBRARY_OUTPUT_DIRECTORY_STORE ${CMAKE_SOURCE_DIR}/python/xoscar/collective/rendezvous)
endif()

if(NOT DEFINED LIBRARY_OUTPUT_DIRECTORY_GLOO)
  set(LIBRARY_OUTPUT_DIRECTORY_GLOO ${CMAKE_SOURCE_DIR}/python/xoscar/collective/gloo)
endif()

include_directories(${CMAKE_SOURCE_DIR}/cpp/collective/rendezvous/include)
include_directories(${CMAKE_SOURCE_DIR}/cpp/collective/gloo/include)
include_directories(../third_party/gloo)

add_subdirectory(collective/rendezvous)
add_subdirectory(collective/gloo)

pybind11_add_module(xoscar_store collective/rendezvous/src/bind_tcp_store.cpp)
target_link_libraries(xoscar_store PRIVATE StoreLib fmt::fmt)
set_target_properties(xoscar_store PROPERTIES LIBRARY_OUTPUT_DIRECTORY
                                              ${LIBRARY_OUTPUT_DIRECTORY_STORE})

pybind11_add_module(xoscar_pygloo collective/gloo/main.cc)
target_link_libraries(xoscar_pygloo PRIVATE GlooLib gloo)
set_target_properties(xoscar_pygloo PROPERTIES LIBRARY_OUTPUT_DIRECTORY
                                               ${LIBRARY_OUTPUT_DIRECTORY_GLOO})
