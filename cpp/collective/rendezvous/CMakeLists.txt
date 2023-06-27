cmake_minimum_required(VERSION 3.11...3.21)

project(
  XoscarRendezvous
  VERSION 0.0.1
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)

include_directories(include)
include_directories(../../../third_party/fmt/include)

add_library(
  StoreLib
  include/error.h
  include/exception.h
  src/exception.cpp
  include/socket.h
  src/socket.cpp
  include/store.hpp
  src/store.cpp
  include/tcp_store.hpp
  src/tcp_store.cpp
  include/unix_sock_utils.hpp
  include/utils.hpp)
