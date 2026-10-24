"""
Smoke test for built wheels (run by cibuildwheel, see pyproject.toml).

Checks that the matching build of the C++ library is loaded from the wheel and that a simple rate-equation simulation
runs. On x86-64, the test reruns itself with QSPEC_DISABLE_AVX2=1 to also test the baseline build.
"""

import os
import platform
import subprocess
import sys

import numpy as np

import qspec.simulate as sim
from qspec._cpp import c_bool, dll

print(f"Loaded {dll._name}")
lib_dir, lib_name = os.path.split(dll._name)
assert os.path.basename(lib_dir) == "bin", "C++ library was not loaded from the wheel."

x86_64 = platform.machine().lower() in {"x86_64", "amd64"}
avx2_disabled = os.environ.get("QSPEC_DISABLE_AVX2", "") not in {"", "0"}
if x86_64:
    assert any("qspec_cpp_avx2" in f for f in os.listdir(lib_dir)), "AVX2 build missing from the x86-64 wheel."
    dll.cpu_supports_avx2.restype = c_bool
    expect_avx2 = not avx2_disabled and dll.cpu_supports_avx2()
    assert ("qspec_cpp_avx2" in lib_name) == expect_avx2, f"Expected the {'AVX2' if expect_avx2 else 'baseline'} build."

s = sim.gen_electronic_ls_state(0.0, s=0.5, l=0, j=0.5, label="s")
p = sim.gen_electronic_ls_state(7.55222766e8, 0.5, 1, 0.5, label="p")
ca = sim.Atom(s + p, sim.DecayMap([("s", "p")], [140.0]))
las = sim.Laser(7.55222766e8, intensity=500.0, polarization=sim.Polarization([0, 1, 0]))
inter = sim.Interaction(ca, [las])
n_t = inter.rates(np.linspace(0, 0.1, 51), delta=np.linspace(-50, 50, 11))

assert n_t.shape == (11, 4, 51), n_t.shape
assert np.all(np.isfinite(n_t)), "Simulation returned non-finite values."
print("Smoke test passed.", flush=True)

if x86_64 and not avx2_disabled:
    subprocess.run([sys.executable, __file__], env={**os.environ, "QSPEC_DISABLE_AVX2": "1"}, check=True)
