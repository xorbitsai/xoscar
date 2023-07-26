cmake_minimum_required(VERSION 3.11...3.21)

project(
  XoscarRendezvous
  VERSION 0.0.1
  LANGUAGES CXX)

# there will be an error that
# "Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.36.32532\include\format"
# on windows if c++ standard is set to 20
if(${CMAKE_SYSTEM_NAME} MATCHES "Windows")
  set(CMAKE_CXX_STANDARD 17)
else()
  set(CMAKE_CXX_STANDARD 20)
endif()

include_directories(include)

add_library(
  StoreLib
  include/error.h
  include/call_once.h
  include/win_sock_utils.hpp
  include/exception.h
  src/exception.cpp
  include/socket.h
  src/socket.cpp
  include/tcp_store.hpp
  src/tcp_store.cpp
  include/unix_sock_utils.hpp
  include/utils.hpp)
