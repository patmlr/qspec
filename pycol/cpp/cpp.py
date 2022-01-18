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
StateHandler = ctypes.POINTER(ctypes.c_char)
DecayMapHandler = ctypes.POINTER(ctypes.c_char)
AtomHandler = ctypes.POINTER(ctypes.c_char)
InteractionHandler = ctypes.POINTER(ctypes.c_char)
ResultHandler = ctypes.POINTER(ctypes.c_char)
SpectrumHandler = ctypes.POINTER(ctypes.c_char)

dll_path = os.path.abspath(os.path.dirname(__file__))
dll_path = os.path.join(dll_path, r'src\pycol_simulate\x64\Release')
dll_name = 'pycol_simulate.dll'
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


# State
dll.state_construct.restype = StateHandler
dll.state_destruct.argtypes = (StateHandler, )

dll.state_init.argtypes = (StateHandler, c_double, c_double, c_double, c_double, c_double, c_double,
                           c_double, vector3d_p, c_double, c_char_p)

dll.state_update.argtypes = (StateHandler, )

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


# Result
dll.result_construct.restype = ResultHandler
dll.result_destruct.argtypes = (ResultHandler, )

dll.result_get_x.argtypes = (ResultHandler, )
dll.result_get_x_size.argtypes = (ResultHandler, )
dll.result_get_x_size.restype = c_size_t

dll.result_get_y.argtypes = (ResultHandler, )
dll.result_get_y_size.argtypes = (ResultHandler, )
dll.result_get_y_size.restype = c_size_t

dll.result_get_v.argtypes = (ResultHandler, )
dll.result_get_v_size.argtypes = (ResultHandler, )
dll.result_get_v_size.restype = c_size_t


# Spectrum
dll.spectrum_get_m_size.argtypes = (SpectrumHandler, )
dll.spectrum_get_m_size.restype = c_size_t
dll.spectrum_get_x_size.argtypes = (SpectrumHandler, )
dll.spectrum_get_x_size.restype = c_size_t
dll.spectrum_get_t_size.argtypes = (SpectrumHandler, )
dll.spectrum_get_t_size.restype = c_size_t
dll.spectrum_get_y_size.argtypes = (SpectrumHandler, )
dll.spectrum_get_y_size.restype = c_size_t
dll.spectrum_get_x.argtypes = (SpectrumHandler, )
dll.spectrum_get_t.argtypes = (SpectrumHandler, )
dll.spectrum_get_y.argtypes = (SpectrumHandler, )

# Interaction
dll.interaction_construct.restype = InteractionHandler
dll.interaction_destruct.argtypes = (InteractionHandler, )

dll.interaction_update.argtypes = (InteractionHandler, )

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

dll.interaction_get_summap.argtypes = (InteractionHandler, )
dll.interaction_get_atommap.argtypes = (InteractionHandler, )
dll.interaction_get_deltamap.argtypes = (InteractionHandler, )

dll.interaction_rate_equations.argtypes = (InteractionHandler, c_double)
dll.interaction_rate_equations.restype = ResultHandler
dll.interaction_rate_equations_y0.argtypes = (InteractionHandler, c_double, c_double_p)
dll.interaction_rate_equations_y0.restype = ResultHandler

dll.interaction_schroedinger.argtypes = (InteractionHandler, c_double)
dll.interaction_schroedinger.restype = ResultHandler
dll.interaction_schroedinger_y0.argtypes = (InteractionHandler, c_double, c_complex_p)
dll.interaction_schroedinger_y0.restype = ResultHandler

dll.interaction_master.argtypes = (InteractionHandler, c_double)
dll.interaction_master.restype = ResultHandler
dll.interaction_master_y0.argtypes = (InteractionHandler, c_double, c_complex_p)
dll.interaction_master_y0.restype = ResultHandler

dll.interaction_master_mc.argtypes = (InteractionHandler, c_double, c_size_t, c_bool)
dll.interaction_master_mc.restype = ResultHandler
dll.interaction_master_mc_v.argtypes = (InteractionHandler, c_double_p, c_size_t, c_double, c_bool)
dll.interaction_master_mc_v.restype = ResultHandler
dll.interaction_master_mc_y0.argtypes = (InteractionHandler, c_double, c_size_t, c_complex_p, c_size_t, c_bool)
dll.interaction_master_mc_y0.restype = ResultHandler
dll.interaction_master_mc_v_y0.argtypes = \
    (InteractionHandler, c_double_p, c_size_t, c_double, c_complex_p, c_size_t, c_bool)
dll.interaction_master_mc_v_y0.restype = ResultHandler

dll.interaction_mean_v.argtypes = (InteractionHandler, c_double_p, c_size_t, c_double, c_int)
dll.interaction_mean_v.restype = ResultHandler

dll.interaction_mean_v_y0_vectord.argtypes = (InteractionHandler, c_double_p, c_size_t, c_double, c_double_p)
dll.interaction_mean_v_y0_vectord.restype = SpectrumHandler

dll.interaction_mean_v_y0_vectorcd.argtypes = (InteractionHandler, c_double_p, c_size_t, c_double, c_complex_p)
dll.interaction_mean_v_y0_vectorcd.restype = SpectrumHandler

dll.interaction_mean_v_y0_matrixcd.argtypes = (InteractionHandler, c_double_p, c_size_t, c_double, c_complex_p)
dll.interaction_mean_v_y0_matrixcd.restype = SpectrumHandler

dll.interaction_spectrum.argtypes = (InteractionHandler, c_double_p, c_size_t, c_double, c_int)
dll.interaction_spectrum.restype = SpectrumHandler

dll.interaction_spectrum_y0_vectord.argtypes = (InteractionHandler, c_double_p, c_size_t, c_double, c_double_p)
dll.interaction_spectrum_y0_vectord.restype = SpectrumHandler

dll.interaction_spectrum_y0_vectorcd.argtypes = (InteractionHandler, c_double_p, c_size_t, c_double, c_complex_p)
dll.interaction_spectrum_y0_vectorcd.restype = SpectrumHandler

dll.interaction_spectrum_y0_matrixcd.argtypes = (InteractionHandler, c_double_p, c_size_t, c_double, c_complex_p)
dll.interaction_spectrum_y0_matrixcd.restype = SpectrumHandler

dll.interaction_spectrum_mean_v.argtypes = \
    (InteractionHandler, c_double_p, c_size_t, c_double_p, c_size_t, c_double, c_int)
dll.interaction_spectrum_mean_v.restype = SpectrumHandler

dll.interaction_spectrum_mean_v_y0_vectord.argtypes = \
    (InteractionHandler, c_double_p, c_size_t, c_double_p, c_size_t, c_double, c_double_p)
dll.interaction_spectrum_mean_v_y0_vectord.restype = SpectrumHandler

dll.interaction_spectrum_mean_v_y0_vectorcd.argtypes = \
    (InteractionHandler, c_double_p, c_size_t, c_double_p, c_size_t, c_double, c_complex_p)
dll.interaction_spectrum_mean_v_y0_vectorcd.restype = SpectrumHandler

dll.interaction_spectrum_mean_v_y0_matrixcd.argtypes = \
    (InteractionHandler, c_double_p, c_size_t, c_double_p, c_size_t, c_double, c_complex_p)
dll.interaction_spectrum_mean_v_y0_matrixcd.restype = SpectrumHandler

dll.interaction_spectrum_mean_v_y0_vector_vectorcd.argtypes = \
    (InteractionHandler, c_double_p, c_size_t, c_double_p, c_size_t, c_double, c_complex_p, c_size_t, c_bool)
dll.interaction_spectrum_mean_v_y0_vector_vectorcd.restype = SpectrumHandler


# @ScatteringRate
dll.sr_generate_y.argtypes = (c_complex_p, c_complex_p, c_complex_p, c_int_p, c_int_p, c_double_p)
dll.sr_generate_y.restype = ctypes.c_void_p
