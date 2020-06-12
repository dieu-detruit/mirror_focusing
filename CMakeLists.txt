cmake_minimum_required(VERSION 3.17)

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

# find packages and include files
find_package(Eigen3 REQUIRED)

find_library(
       OpenMP_LIB
       NAMES omp libomp
    )

set(EXEC_NAMES parabola design_ringfocus propagation_test)

foreach(EXEC_NAME ${EXEC_NAMES})

    add_executable(${EXEC_NAME} ${EXEC_NAME}/main.cpp)
    target_include_directories(${EXEC_NAME} PRIVATE
        ${FFTW3_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIRS}
        external/grid/include
        external/unit/include
        )
    target_link_libraries(${EXEC_NAME}
        ${FFTW3_LIB} ${FFTW3_THREAD_LIB} ${OpenMP_LIB}
        )

    target_compile_options(${EXEC_NAME} PRIVATE
            -g -O3 -Wfatal-errors)

endforeach()
