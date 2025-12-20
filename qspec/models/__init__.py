"""
qspec.models
============

Module for constructing lineshape models.
"""

from qspec.models import _base, _convolved, _fit, _helper, _spectrum, _splitter
from qspec.models._base import Amplifier, Custom, Empty, Linked, Listed, Model, NPeak, Offset, Summed, YPars
from qspec.models._convolved import CONVOLVE, Convolved, GaussChi2Convolved, GaussConvolved, LorentzConvolved
from qspec.models._fit import ROUTINES, fit, reduced_chi2, residuals, sigma_poisson
from qspec.models._helper import find_model, find_models, gen_model
from qspec.models._spectrum import (
    SPECTRA,
    Gauss,
    GaussChi2,
    Lorentz,
    LorentzQI,
    Spectrum,
    Voigt,
    VoigtAsy,
    VoigtCEC,
    VoigtDerivative,
    fwhm_voigt,
    fwhm_voigt_d,
)
from qspec.models._splitter import (
    Hyperfine,
    HyperfineMixed,
    HyperfineQI,
    HyperfineZeeman,
    Splitter,
    SplitterSummed,
    gen_splitter_model,
    get_all_f,
    hf_coeff,
    hf_int,
    hf_shift,
    hf_trans,
)

__all__ = []
__all__.extend(_base.__all__)
__all__.extend(_convolved.__all__)
__all__.extend(_spectrum.__all__)
__all__.extend(_splitter.__all__)
__all__.extend(_helper.__all__)
__all__.extend(_fit.__all__)
