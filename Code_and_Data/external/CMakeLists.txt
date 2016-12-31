
# Let's try to find RngStreams, or else download and build it 

set(RNGSTREAMS_ROOT ${PROJECT_BINARY_DIR}/rngstreams CACHE PATH "RngStreams prefix")

find_path(RngStreams_INCLUDE_DIR
    NAMES RngStream.h
    PATHS ${RNGSTREAMS_ROOT}/include 
)

find_library(RngStreams_LIBRARY
    NAMES rngstreams
    PATHS ${RNGSTREAMS_ROOT}/lib
)

IF(RngStreams_INCLUDE_DIR AND RngStreams_LIBRARY)
    set(RngStreams_FOUND true CACHE BOOL "We have RngStreams")
ENDIF(RngStreams_INCLUDE_DIR AND RngStreams_LIBRARY)

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
#  Not found - let's download and build it.
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

if(NOT RngStreams_FOUND)
    # We are going to setup a rule to build RngStreams.
    # We look for the configure script
    find_program(RNGSTREAMS_CONFIGURE 
        NAMES configure 
        PATHS "${CMAKE_CURRENT_SOURCE_DIR}/rngstreams-1.0.1"
        ENV RNGSTREAMS_SRC_DIR ENGSTREAMS_SOURCE_DIR
        NO_DEFAULT_PATH
    )
    if(NOT RNGSTREAMS_CONFIGURE) 
        # Do we have the source tar.gz file? If not download it
        find_file(RNGSTREAMS_SRC_TAR
            NAMES rngstreams-1.0.1.tar.gz
            PATHS "${CMAKE_CURRENT_SOURCE_DIR}"
            NO_DEFAULT_PATH
            )
        if(NOT RNGSTREAMS_SRC_TAR)
            message("Downloading RngStreams")
            file(DOWNLOAD 
                "http://statmath.wu.ac.at/software/RngStreams/rngstreams-1.0.1.tar.gz"
                "${CMAKE_CURRENT_SOURCE_DIR}/rngstreams-1.0.1.tar.gz"
                SHOW_PROGRESS
                )
            find_file(RNGSTREAMS_SRC_TAR
                NAMES rngstreams-1.0.1.tar.gz
                PATHS "${CMAKE_CURRENT_SOURCE_DIR}"
                NO_DEFAULT_PATH
                )
        endif(NOT RNGSTREAMS_SRC_TAR)
        if(NOT RNGSTREAMS_SRC_TAR)
            message(FATAL_ERROR "Couldn't download RngStreams")
        endif(NOT RNGSTREAMS_SRC_TAR)
        # We have the tar file.  Untar
        message("Extracting RngStreams source")
        execute_process(
            COMMAND ${CMAKE_COMMAND} -E tar xzf ${RNGSTREAMS_SRC_TAR}
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} 
            )
        find_program(RNGSTREAMS_CONFIGURE 
            NAMES configure 
            PATHS "${CMAKE_CURRENT_SOURCE_DIR}/rngstreams-1.0.1"
            ENV RNGSTREAMS_SRC_DIR ENGSTREAMS_SOURCE_DIR
            NO_DEFAULT_PATH
        )        
        unset(RNGSTREAMS_SRC_TAR CACHE)
    endif(NOT RNGSTREAMS_CONFIGURE)

    if(NOT RNGSTREAMS_CONFIGURE)
        message(FATAL_ERROR "Failed to locate RngStreams source directory")
    endif(NOT RNGSTREAMS_CONFIGURE)

    message("Building RngStreams")
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/build_rngstreams
    )
    execute_process(
        COMMAND ${RNGSTREAMS_CONFIGURE} --prefix=${PROJECT_BINARY_DIR}/rngstreams --enable-shared
        OUTPUT_FILE "/dev/null"
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/build_rngstreams
        )
    execute_process(
        COMMAND make
        OUTPUT_FILE "/dev/null"
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/build_rngstreams
        )
    execute_process(
        COMMAND make install
        OUTPUT_FILE "/dev/null"
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/build_rngstreams
        )

    find_path(RngStreams_INCLUDE_DIR
        NAMES RngStream.h
        PATHS ${RNGSTREAMS_ROOT}/include 
    )
    find_library(RngStreams_LIBRARY
        NAMES rngstreams
        PATHS ${RNGSTREAMS_ROOT}/lib
    )

    IF(RngStreams_INCLUDE_DIR AND RngStreams_LIBRARY)
        set(RngStreams_FOUND true CACHE BOOL "We have RngStreams")
    ENDIF(RngStreams_INCLUDE_DIR AND RngStreams_LIBRARY)

    unset(RNGSTREAMS_CONFIGURE CACHE)
endif(NOT RngStreams_FOUND)

set(RngStreams_INCLUDE_DIRS ${RngStreams_INCLUDE_DIR} CACHE PATH "RngStreams include")

if(RngStreams_FOUND)
    message("Found RngStreams in ${RNGSTREAMS_ROOT}")
else(RngStreams_FOUND)
    message(FATAL_ERROR "Failed to locate or build RngStreams")
endif(RngStreams_FOUND)

