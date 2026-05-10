# qspec — macOS port notes

Write-up of what was needed to compile `qspec_cpp` v0.5.0 on macOS, for the maintainer.

## Environment

- macOS 26.3.1 (Darwin 25.3.0), Apple Silicon (arm64)
- Apple Clang 17.0.0 (`/usr/bin/clang++`, ships with Xcode CLT)
- CMake 4.3.2, Ninja 1.13 (both via the Python.framework distribution — no Homebrew)
- Python 3.14
- Dependencies pulled from upstream into `src/_lib/`:
  - Boost 1.90.0 (`https://archives.boost.io/release/1.90.0/source/boost_1_90_0.tar.gz`)
  - Eigen 5.0.0 (`https://gitlab.com/libeigen/eigen/-/archive/5.0.0/eigen-5.0.0.tar.gz`)
  - fmt — cloned from `https://github.com/fmtlib/fmt.git` (default branch)

## Result

`build/macosx-arm64/libqspec_cpp.dylib`, 824 KB, arm64-only (single-arch — see note below). Smoke test passes: `Interaction.rates()` on a Ca⁺ S→P scan returns the expected `(n_delta, atom.size, nt)` array.

```
$ otool -L libqspec_cpp.dylib
libqspec_cpp.dylib:
    @rpath/libqspec_cpp.dylib (compatibility version 0.0.0, current version 0.0.0)
    /usr/lib/libc++.1.dylib
    /usr/lib/libSystem.B.dylib
```

No surprise dependencies; symbols hidden by default via existing `CXX_VISIBILITY_PRESET hidden`.

## Issues encountered & patches applied

### 1. `-static-libstdc++ -static-libgcc` rejected by Apple Clang

`CMakeLists.txt` lines 69–72 set these link flags whenever the compiler ID is `GNU` *or* `Clang`. Apple Clang accepts only `-static-libstdc++` silently in some versions but in 17.0 it errors on both since libstdc++ is not on the system at all (Apple ships only libc++). The flags also make no sense on macOS — there is no GCC runtime to statically link.

**Patch (applied locally):**

```cmake
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(qspec_cpp PRIVATE -fPIC)
    if(NOT APPLE)
        target_link_options(qspec_cpp PRIVATE -static-libstdc++ -static-libgcc)
    endif()
endif()
```

**Suggestion for upstream:** the `if(NOT APPLE)` guard above is the minimal change. Equivalently, gate on `CMAKE_CXX_COMPILER_ID STREQUAL "GNU"` only (the linux-only intent), since Linux Clang would not link statically against GNU libstdc++ either.

### 2. `std::execution::par` / `std::execution::par_unseq` not in libc++

Apple's libc++ ships `<execution>` but does **not** implement the parallel execution policies (this has been a known gap for years; only `std::execution::seq` is available, and its symbol is sometimes also missing depending on the SDK). All five call sites fail to compile:

- `src/Matter.cpp:1029` — `std::for_each(std::execution::par_unseq, …)`
- `src/Interaction.cpp:1156` — `std::for_each(std::execution::par, …)`
- `src/Interaction.cpp:1247`
- `src/Interaction.cpp:1347`
- `src/Interaction.cpp:1467`

The compiler's recovery from the missing policy generates noisy follow-on errors about implicit captures of `__begin2`/`__end2`/`n`/`dopri5` in lambdas. Those are red herrings — they vanish once the policy argument is dropped.

**Patch applied locally:** delete the leading `std::execution::par_unseq, ` / `std::execution::par, ` from each call, leaving the standard sequential `std::for_each(begin, end, fn)`. The surrounding `std::thread worker([…]{ for_each(…); })` still runs the body on a worker thread, but the inner parallelism over `n_vec` is gone.

**Performance impact:** macOS-only regression. Rate / Schrödinger / master / mc_master scans over `delta` (or trajectories for MC) are now serialized inside the worker thread. For a typical 100-point detuning scan this is roughly an `n_cores`-fold slowdown on Apple Silicon.

**Suggested upstream fixes (any one would work):**

- **Easiest:** wrap the policy in a macro and define it to nothing on `__APPLE__`:
  ```cpp
  #ifdef __APPLE__
  #define QSPEC_EXEC_PAR
  #define QSPEC_EXEC_PAR_UNSEQ
  #else
  #include <execution>
  #define QSPEC_EXEC_PAR std::execution::par,
  #define QSPEC_EXEC_PAR_UNSEQ std::execution::par_unseq,
  #endif
  std::for_each(QSPEC_EXEC_PAR n_vec.begin(), n_vec.end(), …);
  ```
- **Better:** use [oneTBB](https://github.com/uxlfoundation/oneTBB) (`tbb::parallel_for_each`) on all platforms; it's the same API the GNU libstdc++ parallel STL is built on, builds cleanly on macOS, and gives you predictable behavior across compilers.
- **Apple-native:** Grand Central Dispatch (`dispatch_apply`) — zero new deps, but Mac-only.

### 3. Python loader rejects Darwin

`qspec/_cpp/cpp.py:57-68` — `_get_platform()` raises before the `return "macos"`, and `_load_dll()` has no `macos` branch.

**Patch applied locally:**

```python
def _get_platform() -> str:
    system = platform.system()
    if system.lower() == "windows":
        return "windows"
    if system.lower() == "linux":
        return "linux"
    if system.lower() == "darwin":
        return "macos"
    raise OSError(f"Unsupported platform: {system}")


def _load_dll() -> ctypes.CDLL:
    # …
    elif plat == "macos":
        dll_path = os.path.join(dll_path, f"macosx-{arch}")
        prefix = "lib"
        suffix = ".dylib"
    else:
        raise OSError(f"Unsupported platform: {plat}")
```

The `macosx-{arch}` directory name matches your existing `CMakePresets.json` `binaryDir` (`build/macosx-x64`, `build/macosx-arm64`). No other change needed.

### 4. fmt `master` branch no longer exists

The CMakeLists references `src/_lib/fmt-master/include`. Cloning fmt with `git clone --branch master https://github.com/fmtlib/fmt.git` fails — upstream fmt renamed `master` → `main`. Plain `git clone` works and gives a directory you can rename to `fmt-master`.

**Suggested upstream fix:** update the README's fmt instruction (or pin a tagged release like `12.1.0` and document the tag).

## Things that worked unchanged

- The `macos-clang-arm64` and `macos-clang-x64` configure presets in `CMakePresets.json` worked as-is.
- `set(CMAKE_OSX_ARCHITECTURES "x86_64;arm64")` did not cause issues — but I built single-arch (`-arch arm64` only emerged in the compile commands because I'm on arm64 and configured the arm64 preset). For a true universal `.dylib`, `cmake --preset macos-clang-arm64` while leaving the `OSX_ARCHITECTURES` line should produce a fat binary — I did not test this.
- C++23 mode compiled fine after the libc++ workaround. No `<format>` / `<ranges>` issues.
- `Eigen: NO SIMD` warning fires on macOS arm64 because `CMakeLists.txt` only emits `-msse4.2` on `CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64"`. NEON is implicit on arm64 — no flag needed — but Eigen's macro detection apparently doesn't trip without an explicit hint. May be worth adding an `aarch64`/`arm64` branch that defines `EIGEN_VECTORIZE_NEON` or just leaves Eigen's auto-detection to do its thing in `-O3`.
  - Note: the existing `elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")` branch sets `-mfpu=neon`, which is a 32-bit ARM GCC flag and is rejected by Apple Clang on aarch64. If you ever want to enable that branch on macOS, drop the flag — NEON is unconditional on `arm64`/`aarch64`.

## Reproduction (clean)

```bash
cd qspec/qspec/_cpp/qspec_cpp/src
mkdir -p _lib && cd _lib

# Boost 1.90.0
curl -L -o boost.tar.gz \
  https://archives.boost.io/release/1.90.0/source/boost_1_90_0.tar.gz
tar -xzf boost.tar.gz && rm boost.tar.gz

# Eigen 5.0.0
curl -L -o eigen.tar.gz \
  https://gitlab.com/libeigen/eigen/-/archive/5.0.0/eigen-5.0.0.tar.gz
tar -xzf eigen.tar.gz && rm eigen.tar.gz

# fmt (default branch)
git clone --depth 1 https://github.com/fmtlib/fmt.git fmt-master

cd ../..
# (apply the three patches above)
cmake --preset macos-clang-arm64
cmake --build --preset macos-clang-arm64
```

## Smoke test

```python
import sys, numpy as np
sys.path.insert(0, "qspec")
import qspec.simulate as sim

s = sim.gen_electronic_ls_state(0., s=0.5, l=0, j=0.5, label="s")
p = sim.gen_electronic_ls_state(7.55222766e8, 0.5, 1, 0.5, label="p")
ca = sim.Atom(s + p, sim.DecayMap([("s","p")], [140.]))
las = sim.Laser(7.55222766e8, intensity=500.,
                polarization=sim.Polarization([0,1,0]))
inter = sim.Interaction(ca, [las])
n_t = inter.rates(np.linspace(0, 0.1, 51), delta=np.linspace(-50, 50, 11))
print(n_t.shape)   # (11, 4, 51)
```

Returns the expected `(n_delta, atom.size, nt)` shape.
