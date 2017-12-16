
# I developed this with version 3.1.1.  It may work with earlier versions too.
cmake_minimum_required(VERSION 3.0)

project(parMH)

enable_language(CXX)
enable_language(C)

include_directories(${CMAKE_BINARY_DIR}/include)

add_subdirectory(external)

