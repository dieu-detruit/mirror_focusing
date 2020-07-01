cmake_minimum_required(VERSION 3.17)

set(CMAKE_CXX_COMPILER "/usr/local/bin/g++-10" CACHE STRING "g++ compiler" FORCE)
set(CMAKE_C_COMPILER "/usr/local/bin/gcc-10" CACHE STRING "gcc compiler" FORCE)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(MIRROR_TEST LANGUAGES C CXX)

include(external/wavefield/wavefield.cmake)

message("WAVEFIELD_INCLUDE_DIRS: ${WAVEFIELD_INCLUDE_DIRS}")
message("WAVEFIELD_DEPENDENT_LIBS: ${WAVEFIELD_DEPENDENT_LIBS}")

# find packages and include files
find_package(Eigen3 REQUIRED)

set(EXEC_NAMES parabola design_ringfocus propagation_test plane  test)

foreach(EXEC_NAME ${EXEC_NAMES})

    add_executable(${EXEC_NAME} ${EXEC_NAME}/main.cpp)
    target_include_directories(${EXEC_NAME} PRIVATE
        ${EIGEN3_INCLUDE_DIRS}
        ${WAVEFIELD_INCLUDE_DIRS}
        external/unit/include
        external/autodiff
        )
    target_link_libraries(${EXEC_NAME}
        ${WAVEFIELD_DEPENDENT_LIBS}
        )

    target_compile_options(${EXEC_NAME} PRIVATE
            -g -O3 -Wfatal-errors)

endforeach()

## exec for ellipoidal mirror
set(ELLIPSE_SRC_NAMES simple knifeedge_scan foucault_test hybrid_propagation raytrace_test)
foreach(SRC_NAME ${ELLIPSE_SRC_NAMES})

    set(EXEC_NAME "ellipse_${SRC_NAME}")

    add_executable(${EXEC_NAME} ellipse/${SRC_NAME}.cpp)
    target_include_directories(${EXEC_NAME} PRIVATE
        ${WAVEFIELD_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIRS}
        external/unit/include
        external/autodiff
        )
    target_link_libraries(${EXEC_NAME}
        ${WAVEFIELD_DEPENDENT_LIBS}
        )

    target_compile_options(${EXEC_NAME} PRIVATE
            -g -O3)

endforeach()
