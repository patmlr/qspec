qspec_cpp
=========

This is the C++ shared library project for the [*qspec*](https://patmlr.github.io/qspec/) Python package. *qspec_cpp* uses CMake to create a .dll (Windows), .so (Linux) or .dylib (MacOS) file. See below for the compilation requirements.

## How to compile

- Include the [Boost](https://www.boost.org/) (1.90.0), [Eigen](https://libeigen.gitlab.io/) (5.0.0) and [fmt](https://github.com/fmtlib/fmt) (12.1.0) libraries in `./src/_lib`.


### Recommendation on Linux
- Install `>= gcc 13.3.0, g++ 13.3.0` and `Ninja`.

### Recommendation on Windows
- Install >= Visual Studio 18 2026.
- In `Developer Powershell for vs` run:
  - `dumpbin /headers qspec_cpp.dll | findstr machine` to check for 64-bit.
  - `dumpbin /dependents qspec_cpp.dll` to check that there are no dependencies on `MSVCP140*.dll` and `VCRUNTIME140*.dll`.
