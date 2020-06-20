cmake_minimum_required(VERSION 3.17)

set(CMAKE_CXX_COMPILER "/usr/local/bin/g++-9" CACHE STRING "g++ compiler" FORCE)
set(CMAKE_C_COMPILER "/usr/local/bin/gcc-9" CACHE STRING "gcc compiler" FORCE)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(MIRROR_TEST LANGUAGES C CXX)

find_package(FFTW3 REQUIRED)

find_library (
    FFTW3_LIB
    NAMES fftw3 libfftw3 # what to look for
    HINTS ${FFTW3_LIBRARY_DIRS} # where to look
    NO_DEFAULT_PATH # do not search system default paths
)

find_library (
    FFTW3_THREAD_LIB
    NAMES fftw3_threads libfftw3_threads # what to look for
    HINTS ${FFTW3_LIBRARY_DIRS} # where to look
    NO_DEFAULT_PATH # do not search system default paths
)

find_package(TBB REQUIRED)

# find packages and include files
find_package(Eigen3 REQUIRED)

set(EXEC_NAMES parabola design_ringfocus propagation_test plane test)

foreach(EXEC_NAME ${EXEC_NAMES})

    add_executable(${EXEC_NAME} ${EXEC_NAME}/main.cpp)
    target_include_directories(${EXEC_NAME} PRIVATE
        ${FFTW3_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIRS}
        /usr/local/Cellar/tbb/2020_U2/include
        external/grid/include
        external/unit/include
        )
    target_link_libraries(${EXEC_NAME}
        ${FFTW3_LIB} ${FFTW3_THREAD_LIB} ${TBB_IMPORTED_TARGETS}
        )

    target_compile_options(${EXEC_NAME} PRIVATE
            -g -O3 -Wfatal-errors)

endforeach()

## exec for ellipoidal mirror
set(ELLIPSE_SRC_NAMES simple knifeedge_scan foucault_test)
foreach(SRC_NAME ${ELLIPSE_SRC_NAMES})

    set(EXEC_NAME "ellipse_${SRC_NAME}")

    add_executable(${EXEC_NAME} ellipse/${SRC_NAME}.cpp)
    target_include_directories(${EXEC_NAME} PRIVATE
        ${FFTW3_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIRS}
        /usr/local/Cellar/tbb/2020_U2/include
        external/grid/include
        external/unit/include
        )
    target_link_libraries(${EXEC_NAME}
        ${FFTW3_LIB} ${FFTW3_THREAD_LIB} ${TBB_IMPORTED_TARGETS}
        )

    target_compile_options(${EXEC_NAME} PRIVATE
            -g -O3 -Wfatal-errors)

endforeach()
