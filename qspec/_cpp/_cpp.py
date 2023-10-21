# -*- coding: utf-8 -*-
"""
PyCLS.cpp.cpp

Created on 23.10.2021

@author: Patrick Mueller

Python/C++ interface.
"""

import os
import ctypes
from numpy import ctypeslib


def set_argtypes(func, argtypes):
    func.argtypes = argtypes


def set_restype(func, restype):
    func.restype = restype


# noinspection PyPep8Naming
class c_complex(ctypes.Structure):
    """
    Complex number, compatible with std::complex layout.
    """
    _fields_ = [('real', ctypes.c_double), ('imag', ctypes.c_double)]

    def __init__(self, z):
        """
        :param z: A complex scalar.
        """
        super().__init__()
        self.real = z.real
        self.imag = z.imag

    def to_complex(self):
        """
        Convert to Python complex.

        :returns: A Python-type complex scalar.
        """
        return self.real + 1.j * self.imag


c_bool = ctypes.c_bool
c_bool_p = ctypes.POINTER(ctypes.c_bool)
c_int = ctypes.c_int
c_int_p = ctypes.POINTER(ctypes.c_int)
c_size_t = ctypes.c_size_t
c_size_t_p = ctypes.POINTER(ctypes.c_size_t)
c_float = ctypes.c_float
c_float_p = ctypes.POINTER(ctypes.c_float)
c_double = ctypes.c_double
c_double_p = ctypes.POINTER(ctypes.c_double)
c_complex_p = ctypes.POINTER(c_complex)
c_char_p = ctypes.c_char_p

vector3d_p = ctypeslib.ndpointer(dtype=float, shape=(3, ))
vector3cd_p = ctypeslib.ndpointer(dtype=complex, shape=(3, ))
matrix3cd_p = ctypeslib.ndpointer(dtype=complex, shape=(3, 3))

PolarizationHandler = ctypes.POINTER(ctypes.c_char)
LaserHandler = ctypes.POINTER(ctypes.c_char)
EnvironmentHandler = ctypes.POINTER(ctypes.c_char)
StateHandler = ctypes.POINTER(ctypes.c_char)
DecayMapHandler = ctypes.POINTER(ctypes.c_char)
AtomHandler = ctypes.POINTER(ctypes.c_char)
InteractionHandler = ctypes.POINTER(ctypes.c_char)
ResultHandler = ctypes.POINTER(ctypes.c_char)
SpectrumHandler = ctypes.POINTER(ctypes.c_char)

dll_path = os.path.abspath(os.path.dirname(__file__))
x64 = r'\x64' if ctypes.sizeof(ctypes.c_void_p) == 8 else ''
dll_path = os.path.join(dll_path, r'src\qspec_cpp{}\Release'.format(x64))
dll_name = 'qspec_cpp.dll'
dll = ctypes.CDLL(os.path.join(dll_path, dll_name))


""" simulate """
# Polarization
dll.polarization_construct.restype = PolarizationHandler
dll.polarization_destruct.argtypes = (PolarizationHandler, )

dll.polarization_init.argtypes = (PolarizationHandler, vector3cd_p, vector3d_p, c_bool)
dll.polarization_def_q_axis.argtypes = (PolarizationHandler, vector3d_p, c_bool)

dll.polarization_get_q_axis.argtypes = (PolarizationHandler, )
dll.polarization_get_q_axis.restype = vector3d_p

dll.polarization_get_x.argtypes = (PolarizationHandler, )
dll.polarization_get_x.restype = vector3cd_p

dll.polarization_get_q.argtypes = (PolarizationHandler, )
dll.polarization_get_q.restype = vector3cd_p

# Laser
dll.laser_construct.restype = LaserHandler
dll.laser_destruct.argtypes = (LaserHandler, )

dll.laser_init.argtype = (LaserHandler, c_double, c_double, PolarizationHandler)

dll.laser_get_freq.argtypes = (LaserHandler, )
dll.laser_get_freq.restype = c_double
dll.laser_set_freq.argtypes = (LaserHandler, c_double)

dll.laser_get_intensity.argtypes = (LaserHandler, )
dll.laser_get_intensity.restype = c_double
dll.laser_set_intensity.argtypes = (LaserHandler, c_double)

dll.laser_get_polarization.argtypes = (LaserHandler, )
dll.laser_get_polarization.restype = PolarizationHandler
dll.laser_set_polarization.argtypes = (LaserHandler, PolarizationHandler)

dll.laser_get_k.argtypes = (LaserHandler, )
dll.laser_get_k.restype = vector3d_p
dll.laser_set_k.argtypes = (LaserHandler, vector3d_p)


# Environment
dll.environment_construct.restype = EnvironmentHandler
dll.environment_destruct.argtypes = (EnvironmentHandler, )

dll.environment_get_E.argtypes = (EnvironmentHandler, )
dll.environment_get_E.restype = c_double
dll.environment_get_e_E.argtypes = (EnvironmentHandler, )
dll.environment_get_e_E.restype = vector3d_p
dll.environment_set_E.argtypes = (EnvironmentHandler, vector3d_p)
dll.environment_set_E_double.argtypes = (EnvironmentHandler, c_double)

dll.environment_get_B.argtypes = (EnvironmentHandler, )
dll.environment_get_B.restype = c_double
dll.environment_get_e_B.argtypes = (EnvironmentHandler, )
dll.environment_get_e_B.restype = vector3d_p
dll.environment_set_B.argtypes = (EnvironmentHandler, vector3d_p)
dll.environment_set_B_double.argtypes = (EnvironmentHandler, c_double)


# State
dll.state_construct.restype = StateHandler
dll.state_destruct.argtypes = (StateHandler, )

dll.state_init.argtypes = (StateHandler, c_double, c_double, c_double, c_double, c_double, c_double,
                           c_double, vector3d_p, c_double, c_char_p)

dll.state_update.argtypes = (StateHandler, )
dll.state_update_env.argtypes = (StateHandler, EnvironmentHandler)

dll.state_get_freq_j.argtypes = (StateHandler, )
dll.state_get_freq_j.restype = c_double
dll.state_set_freq_j.argtypes = (StateHandler, c_double)

dll.state_get_freq.argtypes = (StateHandler, )
dll.state_get_freq.restype = c_double

dll.state_get_s.argtypes = (StateHandler, )
dll.state_get_s.restype = c_double

dll.state_get_l.argtypes = (StateHandler, )
dll.state_get_l.restype = c_double

dll.state_get_j.argtypes = (StateHandler, )
dll.state_get_j.restype = c_double

dll.state_get_i.argtypes = (StateHandler, )
dll.state_get_i.restype = c_double

dll.state_get_f.argtypes = (StateHandler, )
dll.state_get_f.restype = c_double

dll.state_get_m.argtypes = (StateHandler, )
dll.state_get_m.restype = c_double

dll.state_get_hyper_const.argtypes = (StateHandler, )
dll.state_get_hyper_const.restype = vector3d_p
dll.state_set_hyper_const.argtypes = (StateHandler, vector3d_p)

dll.state_get_g.argtypes = (StateHandler, )
dll.state_get_g.restype = c_double
dll.state_set_g.argtypes = (StateHandler, c_double)

dll.state_get_label.argtypes = (StateHandler, )
dll.state_get_label.restype = c_char_p
dll.state_set_label.argtypes = (StateHandler, c_char_p)


# DecayMap
dll.decaymap_construct.restype = DecayMapHandler
dll.decaymap_destruct.argtypes = (DecayMapHandler, )

dll.decaymap_add_decay.argtypes = (DecayMapHandler, c_char_p, c_char_p, c_double)

dll.decaymap_get_label.argtypes = (DecayMapHandler, c_size_t, c_size_t)
dll.decaymap_get_label.restype = c_char_p

dll.decaymap_get_a.argtypes = (DecayMapHandler, )

dll.decaymap_get_size.argtypes = (DecayMapHandler, )
dll.decaymap_get_size.restype = c_size_t

dll.decaymap_get_item.argtypes = (DecayMapHandler, c_char_p, c_char_p)
dll.decaymap_get_item.restype = c_double


# Atom
dll.atom_construct.restype = AtomHandler
dll.atom_destruct.argtypes = (AtomHandler, )

dll.atom_update.argtypes = (AtomHandler, )

dll.atom_add_state.argtypes = (AtomHandler, StateHandler)
dll.atom_clear_states.argtypes = (AtomHandler, )

dll.atom_get_decay_map.argtypes = (AtomHandler, )
dll.atom_get_decay_map.restype = DecayMapHandler
dll.atom_set_decay_map.argtypes = (AtomHandler, DecayMapHandler)

dll.atom_get_mass.argtypes = (AtomHandler, )
dll.atom_get_mass.restype = c_double
dll.atom_set_mass.argtypes = (AtomHandler, c_double)

dll.atom_get_size.argtypes = (AtomHandler, )
dll.atom_get_size.restype = c_size_t

dll.atom_get_gs_size.argtypes = (AtomHandler, )
dll.atom_get_gs.restype = c_size_t
dll.atom_get_gs.argtypes = (AtomHandler, )
dll.atom_get_gs.restype = c_size_t_p

dll.atom_get_m_dipole.argtypes = (AtomHandler, c_size_t)
dll.atom_get_m_dipole.restype = c_double_p

dll.atom_get_L0.argtypes = (AtomHandler, )
dll.atom_get_L0.restype = c_double_p

dll.atom_get_L1.argtypes = (AtomHandler, )
dll.atom_get_L1.restype = c_double_p


# Interaction
dll.interaction_construct.restype = InteractionHandler
dll.interaction_destruct.argtypes = (InteractionHandler, )

dll.interaction_update.argtypes = (InteractionHandler, )

dll.interaction_get_environment.argtypes = (InteractionHandler, )
dll.interaction_get_environment.restype = EnvironmentHandler
dll.interaction_set_environment.argtypes = (InteractionHandler, EnvironmentHandler)

dll.interaction_set_atom.argtypes = (InteractionHandler, AtomHandler)

dll.interaction_add_laser.argtypes = (InteractionHandler, LaserHandler)
dll.interaction_clear_lasers.argtypes = (InteractionHandler, )
dll.interaction_get_lasers_size.argtyps = (InteractionHandler, )
dll.interaction_get_lasers_size.restype = c_size_t
dll.interaction_get_laser.argtypes = (InteractionHandler, c_size_t)
dll.interaction_get_laser.restype = LaserHandler

dll.interaction_get_delta_max.argtypes = (InteractionHandler, )
dll.interaction_get_delta_max.restype = c_double
dll.interaction_set_delta_max.argtypes = (InteractionHandler, c_double)

dll.interaction_get_controlled.argtypes = (InteractionHandler, )
dll.interaction_get_controlled.restype = c_bool
dll.interaction_set_controlled.argtypes = (InteractionHandler, c_bool)

dll.interaction_get_dt.argtypes = (InteractionHandler, )
dll.interaction_get_dt.restype = c_double
dll.interaction_set_dt.argtypes = (InteractionHandler, c_double)

dll.interaction_get_n_history.argtypes = (InteractionHandler, )
dll.interaction_get_n_history.restype = c_int
dll.interaction_get_history.argtypes = (InteractionHandler, )

dll.interaction_get_loop.argtypes = (InteractionHandler, )
dll.interaction_get_loop.restype = c_bool

dll.interaction_get_time_dependent.argtypes = (InteractionHandler, )
dll.interaction_get_time_dependent.restype = c_bool
dll.interaction_set_time_dependent.argtypes = (InteractionHandler, c_bool)

dll.interaction_get_summap.argtypes = (InteractionHandler, )
dll.interaction_get_rabi.argtypes = (InteractionHandler, c_size_t)
dll.interaction_get_atommap.argtypes = (InteractionHandler, )
dll.interaction_get_deltamap.argtypes = (InteractionHandler, )
dll.interaction_get_delta.argtypes = (InteractionHandler, )

dll.interaction_rates.argtypes = \
    (InteractionHandler, c_double_p, c_double_p, c_double_p, c_double_p, c_double_p, c_size_t, c_size_t)
dll.interaction_rates.restype = ctypes.c_void_p

dll.interaction_schroedinger.argtypes = \
    (InteractionHandler, c_double_p, c_double_p, c_double_p, c_complex_p, c_complex_p, c_size_t, c_size_t)
dll.interaction_schroedinger.restype = ctypes.c_void_p

dll.interaction_master.argtypes = \
    (InteractionHandler, c_double_p, c_double_p, c_double_p, c_complex_p, c_complex_p, c_size_t, c_size_t)
dll.interaction_master.restype = ctypes.c_void_p

dll.interaction_mc_schroedinger.argtypes = \
    (InteractionHandler, c_double_p, c_double_p, c_double_p, c_complex_p, c_bool, c_complex_p, c_size_t, c_size_t)
dll.interaction_mc_schroedinger.restype = ctypes.c_void_p


# @ScatteringRate
dll.sr_generate_y.argtypes = (c_complex_p, c_complex_p, c_complex_p, c_int_p, c_int_p, c_double_p)
dll.sr_generate_y.restype = ctypes.c_void_p
