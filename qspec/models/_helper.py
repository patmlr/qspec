"""
qspec.models._helper
====================

Helper functions to create and search lineshape models.
"""

import numpy as np

from qspec.models import _base, _convolved, _spectrum, _splitter
from qspec.qtypes import Iterable, array_like, scalar_nd

__all__ = ["find_model", "find_models", "gen_model"]


def gen_model(
    ijj: scalar_nd,
    shape: str | type | _spectrum.Spectrum | None = None,
    qi: bool = False,
    hf_mixing: bool = False,
    hf_config: dict | None = None,
    n_peaks: int | None = None,
    offsets: array_like | None = None,
    x_cuts: array_like | None = None,
    convolve: str | type | _convolved.Convolved | None = None,
) -> _base.Model:
    """
    Create a lineshape model to fit arbitrary atomic fluorescence spectra.
    The model is composed of the `Spectrum`, a `Splitter` for each set of quantum numbers in `ijj`,
    optionally an [`NPeak`] and a [`Convolved`] model, and an [`Offset`] model.

    :param ijj: The three or an iterable of three quantum numbers $I$, $J_l$ and $J_u$.
     Must be a list with the format `[I, J_l, J_u]` or `[[I0, J0_l, J0_u], [I1, J1_l, J1_u], ...]`.
    :param shape: A `str` representation of a `Spectrum` type, see `SPECTRA`, or a `Spectrum` type.
     If `None`, the [`Empty`] model is used.
    :param qi: Whether to use a quantum interference model, see [`HyperfineQI`].
    :param hf_mixing: Whether to use a hyperfine-induced mixing model, see [`HyperfineMixed`].
     Not implemented for `qi=True`.
    :param hf_config: A `dict` containing the configuration for the hf mixing. Only used if `hf_mixing=True`.
    :param n_peaks: The number of "peaks per resonance" or "lineshape duplicates", see [`NPeak`].
    :param offsets: The orders of the offset polynomials of the separate x-axis intervals.
     Must be a list or a single value. In the former case `len(offsets) == len(x_cuts) + 1` must hold.
     If `offsets` is `None`, a single constant offset is assumed.
    :param x_cuts: The x values where to cut the x-axis. Must be a list or a single value.
     In the former case `len(offsets) == len(x_cuts) + 1` must hold.
     If `x_cuts` is `None`, the x-axis will not be cut.
    :param convolve: A `str` representation of a `Convolved` type, see `CONVOLVE`, or a `Convolved` type.
    :returns: (model) The constructed lineshape model.
    :raises NotImplementedError: If both `qi` and `hf_mixing` are `True`.
    :raises ValueError: If `ijj`, `shape` or `convolve` have an invalid format or value.
    """
    if qi and hf_mixing:
        raise NotImplementedError("QI with HF-mixing is not implemented yet.")

    ijj = np.asarray(ijj, float)

    if len(ijj.shape) == 1:
        ijj = np.expand_dims(ijj, axis=0)

    elif len(ijj.shape) != 2:
        raise ValueError(f"`ijj` must have shape (3, ) or (., 3) but has shape {ijj.shape}.")

    _ijj: list[list[int]] = ijj.tolist()

    if shape is None or not shape:
        model: type[_base.Model] = _base.Empty

    elif isinstance(shape, type) and issubclass(shape, _spectrum.Spectrum):
        model = shape

    elif isinstance(shape, str):
        if shape[0].islower():
            shape = shape[0].upper() + shape[1:]

        if shape not in _spectrum.SPECTRA:
            raise ValueError(f"The shape {shape} is not available. Choose one of {_spectrum.SPECTRA}")
        model = eval(f"_spectrum.{shape}", {"_spectrum": _spectrum})

    else:
        raise ValueError("`shape` must be `None`, a `str` representation of, or a `Spectrum` type.")

    if convolve is None:
        model_c: type[_convolved.Convolved] | None = None

    else:
        if isinstance(convolve, type) and issubclass(convolve, _convolved.Convolved):
            model_c = convolve

        elif isinstance(convolve, str):
            if convolve[0].islower():
                convolve = convolve[0].upper() + convolve[1:]

            if convolve not in _convolved.CONVOLVE:
                raise ValueError(f"The convolution {convolve} is not available. Choose one of {_convolved.CONVOLVE}")

            model_c = eval(f"_convolved.{convolve}Convolved", {"_convolved": _convolved})

        else:
            raise ValueError("convolve must be a str representation of or a Convolved type.")

    spl = _splitter.gen_splitter_model(qi=qi, hf_mixing=hf_mixing)
    npeaks_model: _base.Model
    if hf_mixing:
        npeaks_model = _splitter.SplitterSummed(
            [spl(model(), i, j_l, j_u, f"HF{n}", config=hf_config) for n, (i, j_l, j_u) in enumerate(_ijj)]
        )
    else:
        npeaks_model = _splitter.SplitterSummed(
            [spl(model(), i, j_l, j_u, f"HF{n}") for n, (i, j_l, j_u) in enumerate(_ijj)]
        )

    if n_peaks is not None:
        npeaks_model = _base.NPeak(model=npeaks_model, n_peaks=n_peaks)

    if model_c is not None:
        npeaks_model = model_c(model=npeaks_model)

    return _base.Offset(model=npeaks_model, offsets=offsets, x_cuts=x_cuts)


def find_model[T: _base.Model](model: _base.Model | None, sub_model: type[T]) -> T | None:
    """
    Find the first submodel of a given type. For [`Listed`] models, only the first branch is searched.

    :param model: The model to search in.
    :param sub_model: The type of the submodel to find.
    :returns: (sub) The first submodel of type `sub_model`. If `model` already has the type `sub_model`,
     `model` is returned. Returns `None` if `model` has no submodel of type `sub_model`.
    """
    if model is None:
        return None

    if isinstance(model, sub_model):
        return model

    _model = model.model
    if _model is None and isinstance(model, _base.Listed):
        _model = model.models[0]

    return find_model(_model, sub_model)


def find_models[T: _base.Model](
    model: _base.Model | None, sub_model: type[T], model_list: Iterable[T] | None = None
) -> list[T]:
    """
    Find the first submodel of a given type in every branch of a model.
    The branches are the submodels of [`Listed`] models.

    :param model: The model to search in.
    :param sub_model: The type of the submodels to find.
    :param model_list: A list of models to append the found submodels to.
    :returns: (subs) The first submodels of type `sub_model` of every branch in `model`,
     appended to `model_list`. Branches without a submodel of type `sub_model` are skipped.
    """
    model_list = [] if model_list is None else list(model_list)

    if model is None:
        return model_list

    if isinstance(model, sub_model):
        return [*model_list, model]

    sub_models = model.models if isinstance(model, _base.Listed) else [model.model]
    for m in sub_models:
        model_list = find_models(m, sub_model, model_list=model_list)

    return model_list
