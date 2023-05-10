# -*- coding: utf-8 -*-
"""
pycol.models.helper

Created on 08.05.2023

@author: Patrick Mueller

Helper functions for the models.
"""

import numpy as np

from pycol.types import *
from pycol.models import base, convolved, splitter, spectrum


def gen_model(ijj, shape: Union[str, type, spectrum.Spectrum], qi: bool = False, hf_mixing: bool = False,
              n_peaks: int = 1, offsets: Union[int, list] = None, x_cuts: Union[int, float, list] = None,
              convolve: Union[str, type, spectrum.Spectrum] = None):
    """
    Create a lineshape model to fit arbitrary atomic fluorescence spectra.

    :param ijj: The three or an Iterable of three quantum numbers I, J_l and J_u.
     Must have the format [I, J_l, J_u] or [[I0, J0_l, J0_u], [I1, J1_l, J1_u], ...].
    :param shape: A str representation of or a Spectrum type.
    :param qi: Whether to use a quantum interference model. NOT IMPLEMENTED.
    :param hf_mixing: Whether to use a hyperfine mixing model. NOT IMPLEMENTED.
    :param n_peaks: The number of "peaks per resonance".
    :param offsets: The orders of the offset polynomials of the separate x-axis intervals.
     Must be a list or a single value. In the former case len(offsets) == len(x_cuts) + 1 must hold.
     If offsets is None, a single constant offset is assumed.
    :param x_cuts: The x values where to cut the x-axis. Must be a list or a single value.
     In the former case len(offsets) == len(x_cuts) + 1 must hold.
     If x_cuts is None, the x-axis will not be cut.
    :param convolve: A str representation of or a Convolved type.
    :returns: The defined lineshape model.
    """
    ijj = np.asarray(ijj, float)
    if len(ijj.shape) == 1:
        ijj = np.expand_dims(ijj, axis=0)
    elif len(ijj.shape) != 2:
        raise ValueError('\'ijj\' must have shape (3, ) or (., 3) but has shape {}.'.format(ijj.shape))

    if isinstance(shape, str):
        if shape[0].islower():
            shape = shape[0].upper() + shape[1:]
            if shape not in spectrum.SPECTRA:
                raise ValueError('The shape {} is not available. Choose one of {}'.format(shape, spectrum.SPECTRA))
            shape = eval('spectrum.{}'.format(shape), {'spectrum': spectrum})
    elif isinstance(shape, type) and issubclass(shape, spectrum.Spectrum):
        pass
    else:
        raise ValueError('shape must be a str representation of or a Spectrum type.')

    if convolve is not None:
        if isinstance(convolve, str):
            if convolve[0].islower():
                convolve = convolve[0].upper() + convolve[1:]
                if convolve not in convolved.CONVOLVE:
                    raise ValueError('The convolution {} is not available. Choose one of {}'
                                     .format(convolve, convolved.CONVOLVE))
                convolve = eval('convolved.{}Convolved'.format(convolve), {'convolved': convolved})
        elif isinstance(convolve, type) and issubclass(convolve, convolved.Convolved):
            pass
        else:
            raise ValueError('convolve must be a str representation of or a Convolved type.')

    spl = splitter.gen_splitter_model(qi=qi, hf_mixing=hf_mixing)
    spl_model = splitter.SplitterSummed([spl(shape(), i, j_l, j_u, 'HF{}'.format(n))
                                         for n, (i, j_l, j_u) in enumerate(ijj)])

    npeaks_model = base.NPeak(model=spl_model, n_peaks=n_peaks)
    if convolve is not None:
        npeaks_model = convolve(model=npeaks_model)

    offset_model = base.Offset(model=npeaks_model, offsets=offsets, x_cuts=x_cuts)
    return offset_model
    

def find_model(model: base.Model, sub_model: Union[base.Model, type]):
    """
    :param model: The model to search.
    :param sub_model: The sub model to find.
    :returns: The first sub model of type or with the same type as 'sub_model'. If 'model' already hast the same type as
     'sub_model', 'model' will be returned. Returns None if 'model' has no sub model 'sub_model'.
    """
    model_type = sub_model
    if isinstance(sub_model, base.Model):
        model_type = type(sub_model)
    if model is None:
        return None
    if isinstance(model, model_type):
        return model
    _model = model.model
    if _model is None and hasattr(model, 'models'):
        _model = model.models[0]
    return find_model(_model, sub_model)


def find_models(model: base.Model, sub_model: Union[base.Model, type], model_list: Iterable = None):
    """
    :param model: The model to search.
    :param sub_model: The sub model to find.
    :param model_list: The initial list of models to return.
    :returns: The first sub models of type or with the same type as 'sub_model'. If 'model' already hast the same type as
     'sub_model', 'model' will be returned. Returns None if 'model' has no sub model 'sub_model'.
     This function returns a list of the first models of every branch in model.
    """
    model_type = sub_model
    if isinstance(sub_model, base.Model):
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
