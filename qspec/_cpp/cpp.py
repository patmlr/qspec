"""
PyCLS.cpp.cpp
=============

Python/C++ interface.
"""

import ctypes
import os
import platform
from _ctypes import CFuncPtr
from enum import Enum
from typing import Any

from numpy import ctypeslib

__all__ = [
    "POINTER",
    "C",
    "c_bool",
    "c_bool_p",
    "c_char_p",
    "c_complex",
    "c_complex_p",
    "c_double",
    "c_double_p",
    "c_float",
    "c_float_p",
    "c_int",
    "c_int32",
    "c_int32_p",
    "c_int_p",
    "c_size_t",
    "c_size_t_p",
    "c_void_p",
    "dll",
    "matrix3cd_p",
    "set_argtypes",
    "set_restype",
    "vector3cd_p",
    "vector3d_p",
]


def _get_architecture() -> str:
    """"""
    machine = platform.machine().lower()

    if machine in {"x86_64", "amd64"}:
        return "x64"
    if machine in {"arm64", "aarch64"}:
        return "arm64"

    raise OSError(f"Unsupported Python architecture: {machine}")


def _get_platform() -> str:
    """"""
    system = platform.system()
    if system.lower() == "windows":
        return "windows"
    if system.lower() == "linux":
        return "linux"
    if system.lower() == "darwin":
        return "macos"

    raise OSError(f"Unsupported platform: {system}")


def _load_dll() -> ctypes.CDLL:
    file_path = os.path.abspath(os.path.dirname(__file__))

    arch = _get_architecture()
    plat = _get_platform()
    config = "Debug" if False else "Release"

    dll_path = os.path.join(file_path, "qspec_cpp", "build")

    if plat == "windows":
        dll_path = os.path.join(dll_path, f"windows-{arch}", config)
        prefix = ""
        suffix = ".dll"

    elif plat == "linux":
        if arch == "arm64":
            raise OSError(f"Python architecture {arch} not supported on Linux.")

        dll_path = os.path.join(dll_path, f"linux-{arch}")
        prefix = "lib"
        suffix = ".so"

    elif plat == "macos":
            dll_path = os.path.join(dll_path, f"macosx-{arch}")
            prefix = "lib"
            suffix = ".dylib"

    else:
        raise OSError(f"Unsupported platform: {plat}")

    dll_name = f"{prefix}qspec_cpp{suffix}"

    try:
        return ctypes.CDLL(os.path.join(dll_path, dll_name))
    except FileNotFoundError:
        dll_path = os.path.join(file_path, "bin")
        return ctypes.CDLL(os.path.join(dll_path, dll_name))


def set_argtypes(func: CFuncPtr, argtypes: tuple[Any, ...]) -> None:
    func.argtypes = argtypes


def set_restype(func: CFuncPtr, restype: Any) -> None:
    func.restype = restype


class c_complex(ctypes.Structure):
    """
    Complex number, compatible with std::complex layout.
    """

    _fields_ = [("real", ctypes.c_double), ("imag", ctypes.c_double)]

    def __init__(self, z: complex) -> None:
        """
        :param z: A complex scalar.
        """
        super().__init__()
        self.real = z.real
        self.imag = z.imag

    def to_complex(self) -> complex:
        """
        Convert to Python complex.

        :returns: A Python-type complex scalar.
        """
        return self.real + 1.0j * self.imag


cast = ctypes.cast
POINTER = ctypes.POINTER

c_void_p = ctypes.c_void_p
c_bool = ctypes.c_bool
c_bool_p = POINTER(ctypes.c_bool)
c_int = ctypes.c_int
c_int_p = POINTER(ctypes.c_int)
c_int32 = ctypes.c_int32
c_int32_p = POINTER(ctypes.c_int32)
c_size_t = ctypes.c_size_t
c_size_t_p = POINTER(ctypes.c_size_t)
c_float = ctypes.c_float
c_float_p = POINTER(ctypes.c_float)
c_double = ctypes.c_double
c_double_p = POINTER(ctypes.c_double)
c_complex_p = POINTER(c_complex)
c_char_p = ctypes.c_char_p


vector3d_p = ctypeslib.ndpointer(dtype=float, shape=(3,))
vector3cd_p = ctypeslib.ndpointer(dtype=complex, shape=(3,))
matrix3cd_p = ctypeslib.ndpointer(dtype=complex, shape=(3, 3))


class C(Enum):
    CppClassHandler = POINTER(ctypes.c_char)
    PolarizationHandler = POINTER(ctypes.c_char)
    LaserHandler = POINTER(ctypes.c_char)
    EnvironmentHandler = POINTER(ctypes.c_char)
    StateHandler = POINTER(ctypes.c_char)
    DecayMapHandler = POINTER(ctypes.c_char)
    AtomHandler = POINTER(ctypes.c_char)
    InteractionHandler = POINTER(ctypes.c_char)
    MultivariateNormalHandler = POINTER(ctypes.c_char)


dll = _load_dll()


""" simulate """
# Polarization
dll.polarization_construct.restype = C.PolarizationHandler.value
dll.polarization_destruct.argtypes = (C.PolarizationHandler.value,)

dll.polarization_init.argtypes = (C.PolarizationHandler.value, vector3cd_p, vector3d_p, c_bool)
dll.polarization_def_q_axis.argtypes = (C.PolarizationHandler.value, vector3d_p, c_bool)

dll.polarization_get_q_axis.argtypes = (C.PolarizationHandler.value,)
dll.polarization_get_q_axis.restype = vector3d_p

dll.polarization_get_x.argtypes = (C.PolarizationHandler.value,)
dll.polarization_get_x.restype = vector3cd_p

dll.polarization_get_q.argtypes = (C.PolarizationHandler.value,)
dll.polarization_get_q.restype = vector3cd_p

# Laser
dll.laser_construct.restype = C.LaserHandler.value
dll.laser_destruct.argtypes = (C.LaserHandler.value,)

dll.laser_init.argtypes = (C.LaserHandler.value, c_double, c_double, C.PolarizationHandler.value, c_double_p)

dll.laser_get_freq.argtypes = (C.LaserHandler.value,)
dll.laser_get_freq.restype = c_double
dll.laser_set_freq.argtypes = (C.LaserHandler.value, c_double)

dll.laser_get_intensity.argtypes = (C.LaserHandler.value,)
dll.laser_get_intensity.restype = c_double
dll.laser_set_intensity.argtypes = (C.LaserHandler.value, c_double)

dll.laser_get_polarization.argtypes = (C.LaserHandler.value,)
dll.laser_get_polarization.restype = C.PolarizationHandler.value
dll.laser_set_polarization.argtypes = (C.LaserHandler.value, C.PolarizationHandler.value)

dll.laser_get_k.argtypes = (C.LaserHandler.value,)
dll.laser_get_k.restype = vector3d_p
dll.laser_set_k.argtypes = (C.LaserHandler.value, vector3d_p)

dll.laser_get_kpol.argtypes = (C.LaserHandler.value, c_bool, c_size_t, c_double_p)


# Environment
dll.environment_construct.restype = C.EnvironmentHandler.value
dll.environment_destruct.argtypes = (C.EnvironmentHandler.value,)

dll.environment_get_E.argtypes = (C.EnvironmentHandler.value,)
dll.environment_get_E.restype = c_double
dll.environment_get_e_E.argtypes = (C.EnvironmentHandler.value,)
dll.environment_get_e_E.restype = vector3d_p
dll.environment_set_E.argtypes = (C.EnvironmentHandler.value, vector3d_p)
dll.environment_set_E_double.argtypes = (C.EnvironmentHandler.value, c_double)

dll.environment_get_B.argtypes = (C.EnvironmentHandler.value,)
dll.environment_get_B.restype = c_double
dll.environment_get_e_B.argtypes = (C.EnvironmentHandler.value,)
dll.environment_get_e_B.restype = vector3d_p
dll.environment_set_B.argtypes = (C.EnvironmentHandler.value, vector3d_p)
dll.environment_set_B_double.argtypes = (C.EnvironmentHandler.value, c_double)


# State
dll.state_construct.restype = C.StateHandler.value
dll.state_destruct.argtypes = (C.StateHandler.value,)

dll.state_init.argtypes = (
    C.StateHandler.value,
    c_double,
    c_double_p,
    c_double_p,
    c_double,
    c_double,
    c_double,
    c_double,
    c_bool,
    c_double_p,
    c_size_t,
    vector3d_p,
    c_double,
    c_double,
    c_char_p,
)

dll.state_reset.argtypes = (C.StateHandler.value,)

dll.state_get_freq_j.argtypes = (C.StateHandler.value,)
dll.state_get_freq_j.restype = c_double
dll.state_set_freq_j.argtypes = (C.StateHandler.value, c_double)

dll.state_get_freq.argtypes = (C.StateHandler.value,)
dll.state_get_freq.restype = c_double
dll.state_set_freq.argtypes = (C.StateHandler.value, c_double)

dll.state_get_j.argtypes = (C.StateHandler.value,)
dll.state_get_j.restype = c_double

dll.state_get_i.argtypes = (C.StateHandler.value,)
dll.state_get_i.restype = c_double

dll.state_get_f.argtypes = (C.StateHandler.value,)
dll.state_get_f.restype = c_double

dll.state_get_m.argtypes = (C.StateHandler.value,)
dll.state_get_m.restype = c_double

dll.state_get_hyper_const.argtypes = (C.StateHandler.value,)
dll.state_get_hyper_const.restype = vector3d_p
dll.state_set_hyper_const.argtypes = (C.StateHandler.value, vector3d_p)

dll.state_get_gj.argtypes = (C.StateHandler.value,)
dll.state_get_gj.restype = c_double
dll.state_set_gj.argtypes = (C.StateHandler.value, c_double)

dll.state_get_gi.argtypes = (C.StateHandler.value,)
dll.state_get_gi.restype = c_double
dll.state_set_gi.argtypes = (C.StateHandler.value, c_double)

dll.state_get_label.argtypes = (C.StateHandler.value,)
dll.state_get_label.restype = c_char_p
dll.state_set_label.argtypes = (C.StateHandler.value, c_char_p)


# DecayMap
dll.decaymap_construct.restype = C.DecayMapHandler.value
dll.decaymap_destruct.argtypes = (C.DecayMapHandler.value,)

dll.decaymap_add_decay.argtypes = (
    C.DecayMapHandler.value,
    c_char_p,
    c_char_p,
    c_double_p,
    c_size_t,
    c_double_p,
    c_size_t,
    c_bool,
)

dll.decaymap_get_label.argtypes = (C.DecayMapHandler.value, c_size_t, c_size_t)
dll.decaymap_get_label.restype = c_char_p

dll.decaymap_get_a_i.argtypes = (C.DecayMapHandler.value, c_char_p, c_char_p)
dll.decaymap_get_a_i.restype = c_double

dll.decaymap_get_ae_ik.argtypes = (C.DecayMapHandler.value, c_char_p, c_char_p, c_size_t)
dll.decaymap_get_ae_ik.restype = c_double

dll.decaymap_get_am_ik.argtypes = (C.DecayMapHandler.value, c_char_p, c_char_p, c_size_t)
dll.decaymap_get_am_ik.restype = c_double

dll.decaymap_get_size.argtypes = (C.DecayMapHandler.value,)
dll.decaymap_get_size.restype = c_size_t

dll.decaymap_get_k_em_max.argtypes = (C.DecayMapHandler.value,)
dll.decaymap_get_k_em_max.restype = c_size_t

dll.decaymap_set_k_em_max.argtypes = (C.DecayMapHandler.value, c_size_t)


# Atom
dll.atom_construct.restype = C.AtomHandler.value
dll.atom_destruct.argtypes = (C.AtomHandler.value,)

dll.atom_update.argtypes = (C.AtomHandler.value,)
dll.atom_set_env.argtypes = (C.AtomHandler.value, C.EnvironmentHandler.value)

dll.atom_add_state.argtypes = (C.AtomHandler.value, C.StateHandler.value)
dll.atom_clear_states.argtypes = (C.AtomHandler.value,)

dll.atom_get_decay_map.argtypes = (C.AtomHandler.value,)
dll.atom_get_decay_map.restype = C.DecayMapHandler.value
dll.atom_set_decay_map.argtypes = (C.AtomHandler.value, C.DecayMapHandler.value)

dll.atom_get_gamma.argtypes = (C.DecayMapHandler.value, c_size_t)
dll.atom_get_gamma.restype = c_double

dll.atom_get_mass.argtypes = (C.AtomHandler.value,)
dll.atom_get_mass.restype = c_double
dll.atom_set_mass.argtypes = (C.AtomHandler.value, c_double)

dll.atom_get_size.argtypes = (C.AtomHandler.value,)
dll.atom_get_size.restype = c_size_t

dll.atom_get_gs_size.argtypes = (C.AtomHandler.value,)
dll.atom_get_gs.restype = c_size_t
dll.atom_get_gs.argtypes = (C.AtomHandler.value,)
dll.atom_get_gs.restype = c_size_t_p

dll.atom_get_ek.argtypes = (C.AtomHandler.value, c_size_t)

dll.atom_get_mk.argtypes = (C.AtomHandler.value, c_size_t)

dll.atom_get_emk.argtypes = (C.AtomHandler.value, c_size_t)

dll.atom_get_d_em.argtypes = (C.AtomHandler.value, c_size_t)

dll.atom_get_L0.argtypes = (C.AtomHandler.value,)
dll.atom_get_L0.restype = c_double_p

dll.atom_get_L1.argtypes = (C.AtomHandler.value,)
dll.atom_get_L1.restype = c_double_p

dll.atom_scattering_rate_4pi.argtypes = (
    C.AtomHandler.value,
    c_double_p,
    c_size_t_p,
    c_size_t,
    c_complex_p,
    c_size_t,
    c_bool,
    c_size_t_p,
    c_size_t,
    c_size_t_p,
    c_size_t,
)

dll.atom_scattering_rate_k.argtypes = (
    C.AtomHandler.value,
    c_double_p,
    c_size_t_p,
    c_size_t,
    c_complex_p,
    c_size_t,
    c_bool,
    c_double_p,
    c_size_t,
    c_size_t_p,
    c_size_t,
    c_size_t_p,
    c_size_t,
)

dll.atom_scattering_rate_k_tp.argtypes = (
    C.AtomHandler.value,
    c_double_p,
    c_size_t_p,
    c_size_t,
    c_complex_p,
    c_size_t,
    c_bool,
    c_double_p,
    c_double_p,
    c_size_t,
    c_size_t_p,
    c_size_t,
    c_size_t_p,
    c_size_t,
)

dll.atom_scattering_rate_qk.argtypes = (
    C.AtomHandler.value,
    c_double_p,
    c_size_t_p,
    c_size_t,
    c_complex_p,
    c_size_t,
    c_bool,
    c_double_p,
    c_complex_p,
    c_size_t,
    c_size_t_p,
    c_size_t,
    c_size_t_p,
    c_size_t,
)

dll.atom_scattering_rate_qk_xb.argtypes = (
    C.AtomHandler.value,
    c_double_p,
    c_size_t_p,
    c_size_t,
    c_complex_p,
    c_size_t,
    c_bool,
    c_double_p,
    c_size_t,
    c_size_t,
    c_size_t_p,
    c_size_t,
    c_size_t_p,
    c_size_t,
)

dll.atom_scattering_rate_qk_tp.argtypes = (
    C.AtomHandler.value,
    c_double_p,
    c_size_t_p,
    c_size_t,
    c_complex_p,
    c_size_t,
    c_bool,
    c_double_p,
    c_double_p,
    c_complex_p,
    c_size_t,
    c_size_t_p,
    c_size_t,
    c_size_t_p,
    c_size_t,
)

dll.atom_scattering_rate_qk_tp_xb.argtypes = (
    C.AtomHandler.value,
    c_double_p,
    c_size_t_p,
    c_size_t,
    c_complex_p,
    c_size_t,
    c_bool,
    c_double_p,
    c_double_p,
    c_size_t,
    c_size_t,
    c_size_t_p,
    c_size_t,
    c_size_t_p,
    c_size_t,
)


# Interaction
dll.interaction_construct.restype = C.InteractionHandler.value
dll.interaction_destruct.argtypes = (C.InteractionHandler.value,)

dll.interaction_resonance_info.argtypes = (C.InteractionHandler.value,)

dll.interaction_update.argtypes = (C.InteractionHandler.value,)
dll.interaction_update.restype = c_int32

dll.interaction_get_environment.argtypes = (C.InteractionHandler.value,)
dll.interaction_get_environment.restype = C.EnvironmentHandler.value
dll.interaction_set_environment.argtypes = (C.InteractionHandler.value, C.EnvironmentHandler.value)

dll.interaction_set_atom.argtypes = (C.InteractionHandler.value, C.AtomHandler.value)

dll.interaction_add_laser.argtypes = (C.InteractionHandler.value, C.LaserHandler.value)
dll.interaction_clear_lasers.argtypes = (C.InteractionHandler.value,)
dll.interaction_get_lasers_size.argtypes = (C.InteractionHandler.value,)
dll.interaction_get_lasers_size.restype = c_size_t
dll.interaction_get_laser.argtypes = (C.InteractionHandler.value, c_size_t)
dll.interaction_get_laser.restype = C.LaserHandler.value

dll.interaction_get_delta_max.argtypes = (C.InteractionHandler.value,)
dll.interaction_get_delta_max.restype = c_double
dll.interaction_set_delta_max.argtypes = (C.InteractionHandler.value, c_double)

dll.interaction_get_controlled.argtypes = (C.InteractionHandler.value,)
dll.interaction_get_controlled.restype = c_bool
dll.interaction_set_controlled.argtypes = (C.InteractionHandler.value, c_bool)

dll.interaction_get_dense.argtypes = (C.InteractionHandler.value,)
dll.interaction_get_dense.restype = c_bool
dll.interaction_set_dense.argtypes = (C.InteractionHandler.value, c_bool)

dll.interaction_get_dt.argtypes = (C.InteractionHandler.value,)
dll.interaction_get_dt.restype = c_double
dll.interaction_set_dt.argtypes = (C.InteractionHandler.value, c_double)

dll.interaction_get_dt_max.argtypes = (C.InteractionHandler.value,)
dll.interaction_get_dt_max.restype = c_double
dll.interaction_set_dt_max.argtypes = (C.InteractionHandler.value, c_double)

dll.interaction_get_atol.argtypes = (C.InteractionHandler.value,)
dll.interaction_get_atol.restype = c_double
dll.interaction_set_atol.argtypes = (C.InteractionHandler.value, c_double)

dll.interaction_get_rtol.argtypes = (C.InteractionHandler.value,)
dll.interaction_get_rtol.restype = c_double
dll.interaction_set_rtol.argtypes = (C.InteractionHandler.value, c_double)

dll.interaction_get_n_history.argtypes = (C.InteractionHandler.value,)
dll.interaction_get_n_history.restype = c_int
dll.interaction_get_history.argtypes = (C.InteractionHandler.value,)

dll.interaction_get_loop.argtypes = (C.InteractionHandler.value,)
dll.interaction_get_loop.restype = c_bool

dll.interaction_get_time_dependent.argtypes = (C.InteractionHandler.value,)
dll.interaction_get_time_dependent.restype = c_bool
dll.interaction_set_time_dependent.argtypes = (C.InteractionHandler.value, c_bool)

dll.interaction_get_summap.argtypes = (C.InteractionHandler.value,)
dll.interaction_get_rabi.argtypes = (C.InteractionHandler.value, c_size_t)
dll.interaction_get_atommap.argtypes = (C.InteractionHandler.value,)
dll.interaction_get_deltamap.argtypes = (C.InteractionHandler.value,)
dll.interaction_get_delta.argtypes = (C.InteractionHandler.value,)

dll.interaction_get_hamiltonian.argtypes = (
    C.InteractionHandler.value,
    c_double_p,
    c_double_p,
    c_double_p,
    c_complex_p,
    c_size_t,
    c_size_t,
)
dll.interaction_get_hamiltonian.restype = c_void_p

dll.interaction_rates.argtypes = (
    C.InteractionHandler.value,
    c_double_p,
    c_double_p,
    c_double_p,
    c_double_p,
    c_double_p,
    c_size_t,
    c_size_t,
    c_bool,
)
dll.interaction_rates.restype = c_void_p

dll.interaction_schroedinger.argtypes = (
    C.InteractionHandler.value,
    c_double_p,
    c_double_p,
    c_double_p,
    c_complex_p,
    c_complex_p,
    c_size_t,
    c_size_t,
)
dll.interaction_schroedinger.restype = c_void_p

dll.interaction_master.argtypes = (
    C.InteractionHandler.value,
    c_double_p,
    c_double_p,
    c_double_p,
    c_complex_p,
    c_complex_p,
    c_size_t,
    c_size_t,
)
dll.interaction_master.restype = c_void_p

dll.interaction_mc_master.argtypes = (
    C.InteractionHandler.value,
    c_double_p,
    c_double_p,
    c_double_p,
    c_complex_p,
    c_bool,
    c_complex_p,
    c_size_t,
    c_size_t,
)
dll.interaction_mc_master.restype = c_void_p


# ScatteringRate
dll.sr_generate_y.argtypes = (c_complex_p, c_complex_p, c_complex_p, c_size_t_p, c_size_t_p, c_double_p)
dll.sr_generate_y.restype = c_void_p


# King
dll.multivariatenormal_construct.argtypes = (c_double_p, c_double_p, c_size_t)
dll.multivariatenormal_construct.restype = C.MultivariateNormalHandler.value
dll.multivariatenormal_destruct.argtypes = (C.MultivariateNormalHandler.value,)

dll.multivariatenormal_size.restype = c_size_t
dll.multivariatenormal_rvs.restype = c_void_p

dll.gen_collinear.argtypes = (
    c_double_p,
    c_double_p,
    c_double_p,
    c_size_t_p,
    c_size_t,
    c_size_t,
    c_size_t_p,
    c_bool,
    c_size_t,
    c_bool,
)
dll.gen_collinear.restype = ctypes.c_size_t
