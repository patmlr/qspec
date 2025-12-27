"""
qspec.models._convolved
=======================

Convolution classes for lineshape models.
"""

import numpy as np

from qspec.models._base import CONVOLVE_IS_NONE_ERROR, Model
from qspec.models._spectrum import Gauss, GaussChi2, Lorentz
from qspec.physics import source_energy_pdf
from qspec.qtypes import int_like, is_scalar, ndarray, scalar
from qspec.tools import merge_intervals

__all__ = ["CONVOLVE", "Convolved", "GaussChi2Convolved", "GaussConvolved", "LorentzConvolved"]


CONVOLVE = ["None", "Gauss", "Lorentz", "GaussChi2"]


class Convolved(Model):
    def __init__(self, model: Model | None = None, model_1: Model | None = None) -> None:
        """
        A generic numerical convolution model.

        :param model_0: The first model to convolve.
        :param model_0: The second model to convolve.
        """
        super().__init__(model=model)
        self.type = "Convolved"
        if model is None and model_1 is None:
            raise ValueError(CONVOLVE_IS_NONE_ERROR)

        self.model_1 = model_1
        self.i_1 = self._index

        if self.model_1 is not None:
            for name, val, fix, link in self.model_1.get_pars():
                self._add_arg(f"{name}(conv)", val, fix, link)

        self.j_1 = self._index

        self.precision = 8
        self.n = 2**self.precision + 1
        self.x_int = np.array([])

    def evaluate(self, x: ndarray, *args: scalar, **kwargs) -> ndarray:
        if self.model is None:
            model = self.model_1
            if model is None:
                raise ValueError(CONVOLVE_IS_NONE_ERROR)
            return model.evaluate(x, *args)

        if self.model_1 is None:
            return self.model.evaluate(x, *args)

        self.gen_x_int(*args)

        y = self.model.evaluate(
            np.expand_dims(x, axis=-1) - self.x_int, *args[: self.model.size], **kwargs
        ) * self.model_1.evaluate(self.x_int, *args[self.i_1 : self.j_1])

        return np.asarray(np.trapezoid(y, dx=self.x_int[0, 1] - self.x_int[0, 0]), dtype=float)

    def set_val(self, i: int_like | str, val: scalar, force: bool = False) -> None:
        """
        Set a specific parameter value.

        :param i: The index (`int`) or the name (`str`) of the parameter.
        :param val: The new parameter value.
        :param force: Force the parameter to take exactly the new value.
         If `False`, the parameter is converted to the correct format.
        :returns:
        """
        if isinstance(i, str):
            i = self.p[i]

        if force or is_scalar(val):
            if self.model is None:
                self.vals[i] = float(val)
            else:
                if self.model_1 is not None and self.i_1 <= i < self.j_1:
                    self.model_1.set_val(i - self.i_1, val, force=True)
                self.model.set_val(i, val, force=True)

    def min(self) -> float:
        """
        :returns: A hint for an x-axis minimum for a complete display of the model.
        """
        if self.model is None:
            if self.model_1 is None:
                raise ValueError(CONVOLVE_IS_NONE_ERROR)
            return self.model_1.min()

        if self.model_1 is None:
            return self.model.min()

        return min([self.model.min(), self.model_1.min()])

    def max(self) -> float:
        """
        :returns: A hint for an x-axis maximum for a complete display of the model.
        """
        if self.model is None:
            if self.model_1 is None:
                raise ValueError(CONVOLVE_IS_NONE_ERROR)
            return self.model_1.max()

        if self.model_1 is None:
            return self.model.max()

        return max([self.model.max(), self.model_1.max()])

    def intervals(self) -> list[list[float]]:
        """
        :returns: A list of x-axis intervals for a complete display of the model.
        """
        if self.model is None:
            if self.model_1 is None:
                raise ValueError(CONVOLVE_IS_NONE_ERROR)
            return self.model_1.intervals()

        if self.model_1 is None:
            return self.model.intervals()

        return merge_intervals(
            [[min([i[0], self.model_1.min()]), max([i[1], self.model_1.max()])] for i in self.model.intervals()]
        ).tolist()

    @property
    def dx(self) -> float:
        """
        A hint for an x-axis step size for a smooth display of the model.
        """
        if self.model is None:
            if self.model_1 is None:
                raise ValueError(CONVOLVE_IS_NONE_ERROR)
            return self.model_1.dx

        if self.model_1 is None:
            return self.model.dx

        return max([self.model.dx, self.model_1.dx])

    """ Preprocessing """

    def gen_x_int(self, *args: scalar) -> None:
        """
        Generates x-axis arrays for numerical integration.

        :param args: The function parameters.
        :returns:
        """
        if self.model is None or self.model_1 is None:
            self.x_int = np.array([])
            return

        temp_vals = [v for v in self.vals[: self.j_1]]
        self.set_vals(args[: self.j_1], force=True)

        dx = min([self.model.dx, self.model_1.dx])
        self.x_int = np.expand_dims(np.arange(self.model_1.min(), self.model_1.max() + 0.5 * dx, dx), axis=0)

        self.set_vals(temp_vals, force=True)


class GaussConvolved(Convolved):
    def __init__(self, model: Model) -> None:
        """
        A convolution with a `Gauss` kernel.

        :param model: The model to convolve with a `Gauss` kernel.
        """
        super().__init__(model=model, model_1=Gauss())
        self.type = "GaussConvolved"

    def evaluate(
        self, x: ndarray, *args: scalar, **kwargs: dict
    ) -> ndarray:  # Normalize the kernel function of the convolution to its integral.
        return super().evaluate(x, *args, **kwargs) / (np.sqrt(2 * np.pi) * args[self.i_1])


class LorentzConvolved(Convolved):
    def __init__(self, model: Model) -> None:
        """
        A convolution with a `Lorentz` kernel.

        :param model: The model to convolve with a `Lorentz` kernel.
        """
        super().__init__(model=model, model_1=Lorentz())
        self.type = "LorentzConvolved"

    def evaluate(
        self, x: ndarray, *args: scalar, **kwargs: dict
    ) -> ndarray:  # Normalize the kernel function of the convolution to its integral.
        return super().evaluate(x, *args, **kwargs) / (0.5 * np.pi * args[self.i_1])


class GaussChi2Convolved(Convolved):
    def __init__(self, model: Model) -> None:
        """
        A convolution with a `GaussChi2` kernel.

        :param model: The model to convolve with a `GaussChi2` kernel.
        """
        super().__init__(model=model, model_1=GaussChi2())
        self.type = "GaussChi2Convolved"

    def evaluate(
        self, x: ndarray, *args: scalar, **kwargs: dict
    ) -> ndarray:  # Normalize the kernel function of the convolution to its integral.
        return super().evaluate(x, *args, **kwargs) * source_energy_pdf(
            0, 0, args[self.i_1], args[self.i_1 + 1], collinear=True
        )
