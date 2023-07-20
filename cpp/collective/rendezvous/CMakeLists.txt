cmake_minimum_required(VERSION 3.11...3.21)

project(
  XoscarRendezvous
  VERSION 0.0.1
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

include_directories(include)

add_library(
  StoreLib
  include/error.h
  include/win32-headers.h
  include/CallOnce.h
  include/WinSockUtils.hpp
  include/exception.h
  src/exception.cpp
  include/socket.h
  src/socket.cpp
  include/tcp_store.hpp
  src/tcp_store.cpp
  include/unix_sock_utils.hpp
  include/utils.hpp
  include/Backtrace.h
  src/Backtrace.cpp)
