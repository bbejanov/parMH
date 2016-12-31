
# I developed this with version 3.1.1.  It may work with earlier versions too.
cmake_minimum_required(VERSION 3.0)

project(parMH)

enable_language(CXX)
enable_language(C)

ADD_SUBDIRECTORY(external)
include_directories(RngStreams_INCLUDE_DIRS)

