
include(ExternalProject)
ExternalProject_Add(RngStreams
    PREFIX  "${CMAKE_CURRENT_BINARY_DIR}"
    DOWNLOAD_NO_PROGRESS   1
    URL     "http://statmath.wu.ac.at/software/RngStreams/rngstreams-1.0.1.tar.gz"
    CONFIGURE_COMMAND "../RngStreams/configure" "--prefix=${CMAKE_BINARY_DIR}" "--enable-shared"
)

