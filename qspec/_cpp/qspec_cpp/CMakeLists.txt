cmake_minimum_required(VERSION 3.22)
project(qspec_cpp LANGUAGES C CXX)

# C++ standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Source files
set(QSPEC_SOURCES
    src/qspec_cpp.cpp
    src/Cpu.cpp
    src/Utility.cpp
    src/Physics.cpp
    src/Light.cpp
    src/Matter.cpp
    src/Interaction.cpp
    src/wignerSymbols-cpp.cpp
    src/King.cpp
)

# Header-only dependencies. Uses the copies in src/_lib/ if present (local development),
# otherwise downloads pinned releases (CI and pip builds from the sdist).
include(FetchContent)

function(qspec_find_dependency name local_dir url sha256)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${local_dir}")
        set(${name}_DIR "${CMAKE_CURRENT_SOURCE_DIR}/${local_dir}" PARENT_SCOPE)
        return()
    endif()
    message(STATUS "${local_dir} not found, downloading ${name}")
    FetchContent_Declare(${name}
        URL "${url}"
        URL_HASH SHA256=${sha256}
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        SOURCE_SUBDIR _none_  # Only download and extract, do not add_subdirectory().
    )
    FetchContent_MakeAvailable(${name})
    set(${name}_DIR "${${name}_SOURCE_DIR}" PARENT_SCOPE)
endfunction()

qspec_find_dependency(boost src/_lib/boost_1_90_0
    https://github.com/boostorg/boost/releases/download/boost-1.90.0/boost-1.90.0-b2-nodocs.tar.xz
    9e6bee9ab529fb2b0733049692d57d10a72202af085e553539a05b4204211a6f)
qspec_find_dependency(eigen src/_lib/eigen-5.0.0
    https://gitlab.com/libeigen/eigen/-/archive/5.0.0/eigen-5.0.0.tar.gz
    315c881e19e17542a7d428c5aa37d113c89b9500d350c433797b730cd449c056)
qspec_find_dependency(fmt src/_lib/fmt-main
    https://github.com/fmtlib/fmt/releases/download/12.1.0/fmt-12.1.0.zip
    695fd197fa5aff8fc67b5f2bbc110490a875cdf7a41686ac8512fb480fa8ada7)

# Include directories (header-only)
set(QSPEC_INCLUDES
    src/
    ${boost_DIR}
    ${eigen_DIR}
    ${fmt_DIR}/include
)

# Export macro
add_compile_definitions(QSPEC_EXPORTS)

# std::thread (see src/Parallel.h)
find_package(Threads REQUIRED)

# Adds a shared library target with all settings except the CPU architecture flags.
function(qspec_add_library name)
    add_library(${name} SHARED ${QSPEC_SOURCES})
    target_include_directories(${name} PRIVATE ${QSPEC_INCLUDES})
    target_link_libraries(${name} PRIVATE Threads::Threads)

    target_compile_definitions(${name} PRIVATE
        $<$<CONFIG:Release>:EIGEN_NO_DEBUG>
        $<$<CONFIG:Release>:EIGEN_DONT_PARALLELIZE>
        $<$<CONFIG:Release>:EIGEN_FAST_MATH>
    )

    if(MSVC)
        target_compile_options(${name} PRIVATE /utf-8)

        # Force static runtime on MSVC (no MSVC DLL dependencies)
        set_target_properties(${name} PROPERTIES
            MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    endif()

    # GCC/G++: Force static runtime on GCC/G++ (no GCC/G++ SO dependencies)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        target_compile_options(${name} PRIVATE -fPIC)
        target_link_options(${name} PRIVATE -static-libstdc++ -static-libgcc)
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        target_compile_options(${name} PRIVATE -fPIC)
    endif()

    # Visibility settings (only affect Unix)
    if(UNIX)
        set_target_properties(${name} PROPERTIES
            POSITION_INDEPENDENT_CODE ON
            CXX_VISIBILITY_PRESET hidden
            VISIBILITY_INLINES_HIDDEN YES
        )
    endif()

    set(QSPEC_TARGETS ${QSPEC_TARGETS} ${name} PARENT_SCOPE)
endfunction()

# Target CPU architecture
if(MSVC)
    set(QSPEC_ARCH "${CMAKE_CXX_COMPILER_ARCHITECTURE_ID}")
elseif(APPLE AND CMAKE_OSX_ARCHITECTURES)
    set(QSPEC_ARCH "${CMAKE_OSX_ARCHITECTURES}")
else()
    set(QSPEC_ARCH "${CMAKE_SYSTEM_PROCESSOR}")
endif()
string(TOLOWER "${QSPEC_ARCH}" QSPEC_ARCH)

if(QSPEC_ARCH MATCHES "^(x64|x86_64|amd64)$")
    # x86-64: A baseline build (x86-64-v2, SSE4.2) that runs on all CPUs, and an AVX2 build (x86-64-v3)
    # that qspec/_cpp/cpp.py loads instead if the CPU supports it (see cpu_supports_avx2() in src/Cpu.cpp).
    qspec_add_library(qspec_cpp)
    qspec_add_library(qspec_cpp_avx2)

    if(MSVC)
        target_compile_options(qspec_cpp PRIVATE $<$<CONFIG:Release>:/arch:SSE4.2>)
        target_compile_options(qspec_cpp_avx2 PRIVATE $<$<CONFIG:Release>:/arch:AVX2>)
    else()
        target_compile_options(qspec_cpp PRIVATE $<$<CONFIG:Release>:-march=x86-64-v2>)
        target_compile_options(qspec_cpp_avx2 PRIVATE $<$<CONFIG:Release>:-march=x86-64-v3>)
    endif()

else()
    qspec_add_library(qspec_cpp)

    if(QSPEC_ARCH MATCHES "^(arm64|aarch64)$")
        # ARM64: NEON is implicit
        target_compile_definitions(qspec_cpp PRIVATE
            $<$<CONFIG:Release>:EIGEN_VECTORIZE_NEON>
        )
    endif()
endif()

# Install location inside the Python wheel (used by scikit-build-core, see pyproject.toml).
# qspec/_cpp/cpp.py loads the library from qspec/_cpp/bin/.
install(TARGETS ${QSPEC_TARGETS}
    RUNTIME DESTINATION qspec/_cpp/bin  # .dll
    LIBRARY DESTINATION qspec/_cpp/bin  # .so, .dylib
    ARCHIVE EXCLUDE_FROM_ALL            # .lib import library, not needed by ctypes
)
