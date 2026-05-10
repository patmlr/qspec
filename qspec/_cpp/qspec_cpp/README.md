qspec_cpp
=========

This is the C++ shared library project for the [*qspec*](https://patmlr.github.io/qspec/) Python package. *qspec_cpp* uses CMake to create a .dll (Windows), .so (Linux) or .dylib (MacOS) file. See below for the compilation requirements.

## How to compile

- Move the [Boost](https://www.boost.org/) (1.90.0), [Eigen](https://libeigen.gitlab.io/) (5.0.0) and [fmt](https://github.com/fmtlib/fmt) (12.1.0) libraries into `./src/_lib`.


### Linux
- Install `>= gcc 13.3.0, g++ 13.3.0`, `CMake` and `Ninja`.
- Run `CMake`
  - `cmake --preset linux-gcc-x64`
  - `cmake --build --preset linux-gcc-x64`
- Run `file libqspec_cpp.so` to check for 64-bit.
- Run `ldd libqspec_cpp.so` to check that there are no dependencies on `gcc` or `g++`.

### Windows
- Install [Visual Studio 18 2026](https://code.visualstudio.com/Download) and [CMake](https://cmake.org/).
- Run `CMake`
  - `cmake --preset windows-msvc-x64`
  - `cmake --build --preset windows-msvc-x64-release`
- In `Developer Powershell for vs` run:
  - `dumpbin /headers qspec_cpp.dll | findstr machine` to check for 64-bit.
  - `dumpbin /dependents qspec_cpp.dll` to check that there are no dependencies on `MSVCP140*.dll` and `VCRUNTIME140*.dll`.

### MacOS
- Install `Clang 17.0.0`, `CMake`, `Ninja`
- Run `CMake`
  - `cmake --preset macos-clang-arm64`
  - `cmake --build --preset macos-clang-arm64`
