qspec_cpp
=========

This is the C++ shared library project for the [*qspec*](https://patmlr.github.io/qspec/) Python package. *qspec_cpp* uses CMake to create a .dll (Windows), .so (Linux) or .dylib (MacOS) file. See below for the compilation requirements.

## Python wheels

The Python wheels are built by GitHub Actions (`.github/workflows/wheels.yml`) using [cibuildwheel](https://cibuildwheel.pypa.io/) and [scikit-build-core](https://scikit-build-core.readthedocs.io/), for Windows, Linux and macOS on x64 and arm64.
The library does not use the Python C API, so each wheel is tagged `py3-none-<platform>` and works with all Python 3 versions.
Each wheel only contains the binary for its platform, installed into `qspec/_cpp/bin/`.

On x86-64, CMake builds the library twice: `qspec_cpp` for all CPUs (x86-64-v2, SSE4.2) and `qspec_cpp_avx2` (x86-64-v3: AVX2, FMA, BMI), which is 1.5-2x faster for the master equation.
When importing *qspec*, `qspec/_cpp/cpp.py` loads the baseline build, which checks the CPU (`cpu_supports_avx2()` in `src/Cpu.cpp`), and switches to the AVX2 build if it is supported.
Set the environment variable `QSPEC_DISABLE_AVX2=1` to always use the baseline build.

A local `pip install .` from the repository does not compile anything on Windows (x64, arm64) and Linux x64. It packages the committed binaries from `./build/` instead.
CMake only runs in cibuildwheel, when installing from the sdist, on macOS and Linux aarch64 (no committed binaries), or when the environment variable `QSPEC_BUILD_CPP=1` is set.
Compiling takes several minutes and downloads Boost (~50 MB), Eigen and fmt if they are not in `./src/_lib`.
pip hides the build output, so use `pip install -v .` to follow the download and compilation progress.

To build a wheel with a freshly compiled library for the current machine locally, run `QSPEC_BUILD_CPP=1 pip wheel . --no-deps -w dist` in the repository root.

## How to compile

- Move the [Boost](https://www.boost.org/) (1.90.0), [Eigen](https://libeigen.gitlab.io/) (5.0.0) and [fmt](https://github.com/fmtlib/fmt) (12.1.0) libraries into `./src/_lib`.
  If they are missing, CMake downloads them automatically (see `qspec_find_dependency` in `CMakeLists.txt`).


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
