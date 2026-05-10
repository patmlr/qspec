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
    src/_lib/fmt-main/include
)

# Export macro
add_compile_definitions(QSPEC_EXPORTS)

# Build shared library
add_library(qspec_cpp SHARED ${QSPEC_SOURCES})
target_include_directories(qspec_cpp PRIVATE ${QSPEC_INCLUDES})

target_compile_definitions(qspec_cpp PRIVATE
    $<$<CONFIG:Release>:EIGEN_NO_DEBUG>
    $<$<CONFIG:Release>:EIGEN_DONT_PARALLELIZE>
    $<$<CONFIG:Release>:EIGEN_FAST_MATH>
)

if(MSVC)
    target_compile_options(qspec_cpp PRIVATE
        /utf-8

        # AVX2 only for x64 release builds
        $<$<AND:$<CONFIG:Release>,$<STREQUAL:${CMAKE_VS_PLATFORM_NAME},x64>>:/arch:AVX2>
    )

    # ARM64: NEON is implicit
    if(CMAKE_VS_PLATFORM_NAME STREQUAL "ARM64")
        target_compile_definitions(qspec_cpp PRIVATE
            $<$<CONFIG:Release>:EIGEN_VECTORIZE_NEON>
        )
    endif()

else()
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64)$")
        target_compile_options(qspec_cpp PRIVATE
            $<$<CONFIG:Release>:-msse4.2>
        )

    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
        # ARM64 NEON is implicit
        target_compile_definitions(qspec_cpp PRIVATE
            $<$<CONFIG:Release>:EIGEN_VECTORIZE_NEON>
        )
    endif()
endif()

# GCC/G++: Force static runtime on GCC/G++ (no GCC/G++ SO dependencies)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(qspec_cpp PRIVATE -fPIC)
    target_link_options(qspec_cpp PRIVATE -static-libstdc++ -static-libgcc)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(qspec_cpp PRIVATE -fPIC)
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
