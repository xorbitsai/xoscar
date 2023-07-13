cmake_minimum_required(VERSION 3.11...3.21)

project(
  XoscarGloo
  VERSION 0.0.1
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)

include_directories(include)
include_directories(../rendezvous/include)
include_directories(../../../third_party/pybind11/include)

add_library(
  GlooLib
  include/collective.h
  include/rendezvous.h
  include/transport.h
  include/config.h
  src/allgather.cc
  src/allreduce.cc
  src/barrier.cc
  src/broadcast.cc
  src/gather.cc
  src/recv.cc
  src/reduce_scatter.cc
  src/reduce.cc
  src/rendezvous.cc
  src/scatter.cc
  src/send.cc
  src/transport.cc
  src/all_to_all.cc)
