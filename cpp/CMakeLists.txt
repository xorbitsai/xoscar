cmake_minimum_required(VERSION 3.11...3.21)

project(XoscarCollective)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
if(NOT DEFINED LIBRARY_OUTPUT_DIRECTORY)
  set(LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/python/xoscar/collective/rendezvous)
endif()

include_directories(${CMAKE_SOURCE_DIR}/cpp/collective/rendezvous/include)

add_subdirectory(collective/rendezvous)

pybind11_add_module(xoscar_store collective/rendezvous/src/bind_tcp_store.cpp)
target_link_libraries(xoscar_store PRIVATE StoreLib fmt::fmt)
set_target_properties(xoscar_store PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${LIBRARY_OUTPUT_DIRECTORY})
