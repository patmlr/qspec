# -*- coding: utf-8 -*-
"""
qspec.models._convolved
=======================

Convolution classes for lineshape models.
"""

import numpy as np

from qspec.qtypes import *
from qspec.tools import merge_intervals
from qspec.physics import source_energy_pdf
from qspec.models._base import Model
from qspec.models._spectrum import Gauss, Lorentz, GaussChi2

__all__ = ['CONVOLVE', 'Convolved', 'GaussConvolved', 'LorentzConvolved', 'GaussChi2Convolved']


CONVOLVE = ['None', 'Gauss', 'Lorentz', 'GaussChi2']


class Convolved(Model):
    def __init__(self, model_0: Model, model_1: Model):
        """
        A generic numerical convolution model.

        :param model_0: The first model to convolve.
        :param model_0: The second model to convolve.
        """
        super().__init__(model=model_0)
        self.type = 'Convolved'

        self.model_1 = model_1
        self.i_1 = self._index

        for (name, val, fix, link) in self.model_1.get_pars():
            self._add_arg('{}(conv)'.format(name), val, fix, link)

        self.j_1 = self._index

        self.precision = 8
        self.n = 2 ** self.precision + 1
        self.x_int = None

    def evaluate(self, x: array_like, *args: array_iter, **kwargs: dict) -> ndarray:
        self.gen_x_int(*args)
        y = self.model.evaluate(np.expand_dims(x, axis=-1) - self.x_int, *args[:self.model.size], **kwargs) \
            * self.model_1.evaluate(self.x_int, *args[self.i_1:self.j_1])
        return np.trapezoid(y, dx=self.x_int[0, 1] - self.x_int[0, 0])

    def set_val(self, i: Union[int_like, str], val: array_like, force: bool = False):
        """
        Set a specific parameter value.

        :param i: The index (`int`) or the name (`str`) of the parameter.
        :param val: The new parameter value.
        :param force: Force the parameter to take exactly the new value.
         If `False`, the parameter is converted to the correct format.
        :returns:
        """
        if force or isinstance(val, int) or isinstance(val, float):
            if self.model is None:
                self.vals[i] = val
            else:
                if self.i_1 <= i < self.j_1:
                    self.model_1.set_val(i - self.i_1, val, force=True)
                self.model.set_val(i, val, force=True)

    def min(self) -> float:
        """
        :returns: A hint for an x-axis minimum for a complete display of the model.
        """
        return self.model.min() + self.model_1.min()

    def max(self) -> float:
        """
        :returns: A hint for an x-axis maximum for a complete display of the model.
        """
        return self.model.max() + self.model_1.max()

    def intervals(self) -> list[list[float]]:
        """
        :returns: A list of x-axis intervals for a complete display of the model.
        """
        return merge_intervals([[i[0] + self.model_1.min(), i[1] + self.model_1.max()]
                                for i in self.model.intervals()]).tolist()

    @property
    def dx(self):
        """
        A hint for an x-axis step size for a smooth display of the model.
        """
        return max([self.model.dx, self.model_1.dx])

    """ Preprocessing """

    def gen_x_int(self, *args: array_iter):
        """
        Generates x-axis arrays for numerical integration.

        :param args: The function parameters.
        :returns:
        """
        temp_vals = [v for v in self.vals[:self.j_1]]
        self.set_vals(args[:self.j_1], force=True)
        dx = min([self.model.dx, self.model_1.dx])
        self.x_int = np.expand_dims(np.arange(self.model_1.min(), self.model_1.max() + 0.5 * dx, dx), axis=0)
        self.set_vals(temp_vals, force=True)


class GaussConvolved(Convolved):
    def __init__(self, model: Model):
        """
        A convolution with a `Gauss` kernel.

        :param model: The model to convolve with a `Gauss` kernel.
        """
        super().__init__(model_0=model, model_1=Gauss())
        self.type = 'GaussConvolved'

    def evaluate(self, x: array_like, *args: array_iter, **kwargs: dict) -> ndarray:  # Normalize the kernel function of the convolution to its integral.
        return super().evaluate(x, *args, **kwargs) / (np.sqrt(2 * np.pi) * args[self.i_1])


class LorentzConvolved(Convolved):
    def __init__(self, model: Model):
        """
        A convolution with a `Lorentz` kernel.

        :param model: The model to convolve with a `Lorentz` kernel.
        """
        super().__init__(model_0=model, model_1=Lorentz())
        self.type = 'LorentzConvolved'

    def evaluate(self, x: array_like, *args: array_iter, **kwargs: dict) -> ndarray:  # Normalize the kernel function of the convolution to its integral.
        return super().evaluate(x, *args, **kwargs) / (0.5 * np.pi * args[self.i_1])


class GaussChi2Convolved(Convolved):
    def __init__(self, model: Model):
        """
        A convolution with a `GaussChi2` kernel.

        :param model: The model to convolve with a `GaussChi2` kernel.
        """
        super().__init__(model_0=model, model_1=GaussChi2())
        self.type = 'GaussChi2Convolved'

    def evaluate(self, x: array_like, *args: array_iter, **kwargs: dict) -> ndarray:  # Normalize the kernel function of the convolution to its integral.
        return super().evaluate(x, *args, **kwargs) \
            * source_energy_pdf(0, 0, args[self.i_1], args[self.i_1 + 1], collinear=True)
