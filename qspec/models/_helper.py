# -*- coding: utf-8 -*-
"""
qspec.models._helper
====================

Helper functions for the models.
"""

import numpy as np

from qspec.qtypes import *
from qspec.models import _base, _convolved, _splitter, _spectrum


__all__ = ['gen_model', 'find_model', 'find_models']


def gen_model(ijj, shape: Union[str, type, _spectrum.Spectrum] = None,
              qi: bool = False, hf_mixing: bool = False, hf_config: dict = None,
              n_peaks: int = None, offsets: Union[int, list] = None, x_cuts: Union[int, float, list] = None,
              convolve: Union[str, type, _spectrum.Spectrum] = None) -> _base.Model:
    """
    Create a lineshape model to fit arbitrary atomic fluorescence spectra.

    :param ijj: The three or an Iterable of three quantum numbers $I$, $J_l$ and $J_u$.
     Must be a list with the format `[I, J_l, J_u]` or `[[I0, J0_l, J0_u], [I1, J1_l, J1_u], ...]`.
    :param shape: A str representation of or a `Spectrum` type.
    :param qi: Whether to use a quantum interference model.
    :param hf_mixing: Whether to use a hyperfine-induced mixing model. Not implemented for `qi=True`.
    :param hf_config: A `dict` containing the configuration for the hf mixing. Only used if `hf_mixing=True`.
    :param n_peaks: The number of "peaks per resonance" or "lineshape duplicates".
    :param offsets: The orders of the offset polynomials of the separate x-axis intervals.
     Must be a list or a single value. In the former case `len(offsets) == len(x_cuts) + 1` must hold.
     If `offsets` is `None`, a single constant offset is assumed.
    :param x_cuts: The x values where to cut the x-axis. Must be a list or a single value.
     In the former case `len(offsets) == len(x_cuts) + 1` must hold.
     If `x_cuts` is `None`, the x-axis will not be cut.
    :param convolve: A str representation of or a `Convolved` type.
    :returns: The constructed lineshape model.
    """
    if qi and hf_mixing:
        raise NotImplementedError('QI with HF-mixing is not implemented yet.')
    ijj = np.asarray(ijj, float)
    if len(ijj.shape) == 1:
        ijj = np.expand_dims(ijj, axis=0)
    elif len(ijj.shape) != 2:
        raise ValueError('\'ijj\' must have shape (3, ) or (., 3) but has shape {}.'.format(ijj.shape))
    ijj = ijj.tolist()

    if shape is None or not shape:
        shape = _base.Empty
    elif isinstance(shape, str):
        if shape[0].islower():
            shape = shape[0].upper() + shape[1:]
            if shape not in _spectrum.SPECTRA:
                raise ValueError('The shape {} is not available. Choose one of {}'.format(shape, _spectrum.SPECTRA))
            shape = eval('_spectrum.{}'.format(shape), {'_spectrum': _spectrum})
    elif isinstance(shape, type) and issubclass(shape, _spectrum.Spectrum):
        pass
    else:
        raise ValueError('shape must be None, a str representation of or a Spectrum type.')

    if convolve is not None:
        if isinstance(convolve, str):
            if convolve[0].islower():
                convolve = convolve[0].upper() + convolve[1:]
                if convolve not in _convolved.CONVOLVE:
                    raise ValueError('The convolution {} is not available. Choose one of {}'
                                     .format(convolve, _convolved.CONVOLVE))
                convolve = eval('_convolved.{}Convolved'.format(convolve), {'_convolved': _convolved})
        elif isinstance(convolve, type) and issubclass(convolve, _convolved.Convolved):
            pass
        else:
            raise ValueError('convolve must be a str representation of or a Convolved type.')

    spl = _splitter.gen_splitter_model(qi=qi, hf_mixing=hf_mixing)
    if hf_mixing:
        npeaks_model = _splitter.SplitterSummed([spl(shape(), i, j_l, j_u, 'HF{}'.format(n), hf_config)
                                                 for n, (i, j_l, j_u) in enumerate(ijj)])
    else:
        npeaks_model = _splitter.SplitterSummed([spl(shape(), i, j_l, j_u, 'HF{}'.format(n))
                                                 for n, (i, j_l, j_u) in enumerate(ijj)])

    if n_peaks is not None:
        npeaks_model = _base.NPeak(model=npeaks_model, n_peaks=n_peaks)
    if convolve is not None:
        npeaks_model = convolve(model=npeaks_model)

    offset_model = _base.Offset(model=npeaks_model, offsets=offsets, x_cuts=x_cuts)
    return offset_model
    

def find_model(model: _base.Model, sub_model: Union[_base.Model, type]) -> Optional[_base.Model]:
    """
    :param model: The model to search in.
    :param sub_model: The submodel to find.
    :returns: The first submodel of type or with the same type as `sub_model`. If `model` already has the same type as
     `sub_model`, `model` will be returned. Returns `None` if `model` has no submodel `sub_model`.
    """
    model_type = sub_model
    if isinstance(sub_model, _base.Model):
        model_type = type(sub_model)
    if model is None:
        return None
    if isinstance(model, model_type):
        return model
    _model = model.model
    if _model is None and hasattr(model, 'models'):
        _model = model.models[0]
    return find_model(_model, sub_model)


def find_models(model: _base.Model, sub_model: Union[_base.Model, type], model_list: Iterable = None) \
        -> list[_base.Model]:
    """
    :param model: The model to search in.
    :param sub_model: The submodel to find.
    :param model_list: A list of models to append the found submodel to.
    :returns: This function returns a list of the first models of type or with the same type as `sub_model`
     for every branch in model.
     
    """
    model_type = sub_model
    if isinstance(sub_model, _base.Model):
        model_type = type(sub_model)
    if model_list is None:
        model_list = []
    if model is None:
        return model_list
    if isinstance(model, model_type):
        return model_list + [model, ]
    _model = [model.model]
    if _model is None and hasattr(model, 'models'):
        _model = [m for m in model.models]
    for m in _model:
        find_models(m, sub_model, model_list=model_list)
    return model_list
