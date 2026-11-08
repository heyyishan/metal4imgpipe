cmake_minimum_required(VERSION 3.25)
project(Metal4ImagePipeline LANGUAGES CXX OBJCXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OBJCXX_STANDARD 23)
set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)

# macOS deployment target for Metal 4 / Apple Silicon
set(CMAKE_OSX_DEPLOYMENT_TARGET "15.0" CACHE STRING "Minimum macOS version")
set(CMAKE_OSX_ARCHITECTURES "arm64")

# Compile Metal shaders
set(METAL_SHADER_DIR "${CMAKE_SOURCE_DIR}/src/shaders")
set(METAL_AIR_DIR "${CMAKE_BINARY_DIR}/metal_air")
set(METAL_LIB_OUTPUT "${CMAKE_BINARY_DIR}/default.metallib")

file(MAKE_DIRECTORY ${METAL_AIR_DIR})
file(GLOB METAL_SOURCES "${METAL_SHADER_DIR}/*.metal")

set(METAL_AIR_FILES)
foreach(METAL_SRC ${METAL_SOURCES})
    get_filename_component(METAL_NAME ${METAL_SRC} NAME_WE)
    set(AIR_FILE "${METAL_AIR_DIR}/${METAL_NAME}.air")
    add_custom_command(
        OUTPUT ${AIR_FILE}
        COMMAND xcrun -sdk macosx metal
            -std=metal3.2
            -O2
            -I "${METAL_SHADER_DIR}"
            -c "${METAL_SRC}"
            -o "${AIR_FILE}"
        DEPENDS ${METAL_SRC} "${METAL_SHADER_DIR}/common.h"
        COMMENT "Compiling ${METAL_NAME}.metal -> .air"
    )
    list(APPEND METAL_AIR_FILES ${AIR_FILE})
endforeach()

add_custom_command(
    OUTPUT ${METAL_LIB_OUTPUT}
    COMMAND xcrun -sdk macosx metallib ${METAL_AIR_FILES} -o ${METAL_LIB_OUTPUT}
    DEPENDS ${METAL_AIR_FILES}
    COMMENT "Linking metal library"
)
add_custom_target(MetalShaders ALL DEPENDS ${METAL_LIB_OUTPUT})

# Core library
set(CORE_SOURCES
    src/core/gpu_context.mm
    src/core/frame_pool.mm
    src/core/filter_graph.mm
    src/core/pipeline_engine.mm
    src/core/benchmark.mm
    src/filters/bilateral_filter.mm
    src/filters/unsharp_mask.mm
    src/filters/color_grading.mm
    src/filters/edge_smooth.mm
    src/display/display_link.mm
)

add_library(m4pipe_core STATIC ${CORE_SOURCES})
target_include_directories(m4pipe_core PUBLIC
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/core
    ${CMAKE_SOURCE_DIR}/src/filters
    ${CMAKE_SOURCE_DIR}/src/display
)
target_link_libraries(m4pipe_core PUBLIC
    "-framework Metal"
    "-framework MetalKit"
    "-framework Foundation"
    "-framework QuartzCore"
    "-framework CoreGraphics"
    "-framework IOSurface"
    "-framework Accelerate"
    "-framework AppKit"
    "-framework MetalPerformanceShaders"
)
target_compile_options(m4pipe_core PUBLIC -fobjc-arc)
add_dependencies(m4pipe_core MetalShaders)

# Main executable
add_executable(m4pipe src/main.mm)
target_link_libraries(m4pipe PRIVATE m4pipe_core)
target_compile_options(m4pipe PRIVATE -fobjc-arc)

# Python bindings (optional)
find_package(Python3 COMPONENTS Interpreter Development)
find_package(pybind11 QUIET)
if(pybind11_FOUND AND Python3_FOUND)
    pybind11_add_module(m4pipe_py src/python/bindings.mm)
    target_link_libraries(m4pipe_py PRIVATE m4pipe_core)
    target_compile_options(m4pipe_py PRIVATE -fobjc-arc)
    message(STATUS "Python bindings enabled")
else()
    message(STATUS "pybind11 not found, Python bindings disabled. Install with: pip install pybind11")
endif()

# Test executable
add_executable(m4pipe_test tests/test_pipeline.mm tests/test_filters.mm)
target_link_libraries(m4pipe_test PRIVATE m4pipe_core)
target_compile_options(m4pipe_test PRIVATE -fobjc-arc)
