cmake_minimum_required(VERSION 3.22)
project(qspec_cpp LANGUAGES C CXX)

# C++ standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Source files
set(QSPEC_SOURCES
    src/qspec_cpp.cpp
    src/Utility.cpp
    src/Physics.cpp
    src/Light.cpp
    src/Matter.cpp
    src/Interaction.cpp
    src/wignerSymbols-cpp.cpp
    src/King.cpp
)

# Include directories (header-only)
set(QSPEC_INCLUDES
    src/
    src/_lib/boost_1_90_0
    src/_lib/eigen-5.0.0
    src/_lib/fmt-master/include
)

# Export macro
add_compile_definitions(QSPEC_EXPORTS)

# Build shared library
add_library(qspec_cpp SHARED ${QSPEC_SOURCES})
target_include_directories(qspec_cpp PRIVATE ${QSPEC_INCLUDES})

# MSVC: UTF-8 source encoding
if(MSVC)
    target_compile_options(qspec_cpp PRIVATE /utf-8)
endif()

# GCC/G++: Force static runtime on GCC/G++ (no GCC/G++ SO dependencies)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(qspec_cpp PRIVATE -fPIC)
    target_link_options(qspec_cpp PRIVATE -static-libstdc++ -static-libgcc)
endif()

# Output DLL name
set_target_properties(qspec_cpp PROPERTIES
    OUTPUT_NAME qspec_cpp
)

# Force static runtime on MSVC (no MSVC DLL dependencies)
if (MSVC)
    set_target_properties(qspec_cpp PROPERTIES
        MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

# Visibility settings (only affect Unix)
if(UNIX)
    set_target_properties(qspec_cpp PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN YES
    )
endif()

if(APPLE)
    # Build for both x86_64 and arm64
    set(CMAKE_OSX_ARCHITECTURES "x86_64;arm64")
    set_target_properties(qspec_cpp PROPERTIES
        POSITION_INDEPENDENT_CODE ON
    )
endif()
