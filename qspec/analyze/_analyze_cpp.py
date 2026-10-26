"""
qspec.analyze._analyze_cpp
==========================

Classes and methods for the `analyze` module using the Python/C++ interface.
"""

import numpy as np

from qspec._cpp import CppHandler, c_bool, c_double_p, c_size_t, c_size_t_p, dll
from qspec.qtypes import array_like, floating, int_like, ndarray

__all__ = ["generate_collinear_points_cpp"]


class MultivariateNormal:
    def __init__(
        self,
        mean: array_like,
        cov: array_like,
        instance: "MultivariateNormal | CppHandler | None" = None,
    ) -> None:
        self.instance = instance
        if self.instance is None:
            mean, cov = np.ascontiguousarray(mean, dtype=float), np.ascontiguousarray(cov, dtype=float)
            self.instance = dll.multivariatenormal_construct(
                mean.ctypes.data_as(c_double_p), cov.ctypes.data_as(c_double_p), c_size_t(mean.size)
            )

    def __del__(self) -> None:
        dll.multivariatenormal_destruct(self.instance)

    @property
    def size(self) -> int:
        return dll.multivariatenormal_size(self.instance)

    def rvs(self) -> ndarray[floating]:
        ret = np.zeros(self.size, dtype=float)
        dll.multivariatenormal_rvs(self.instance, ret.ctypes.data_as(c_double_p))
        return ret


def generate_collinear_points_cpp(
    mean: ndarray,
    cov: ndarray,
    n_samples: int_like | None = None,
    n_accepted: int_like | None = None,
    seed: int | None = None,
    report: bool | None = None,
    **kwargs,
) -> tuple[ndarray, int, int]:
    r"""
    Randomly generate points $\vec{p}_i$ according to the given data vectors $\vec{\mu}_i\in\mathbb{R}^n$
    and covariance matrices $\mathbf{\Sigma}_i\in\mathbb{R}^{n\times n}$,
    under the condition that they are aligned on a straight line. This function uses C++.

    :param mean: The data vectors $\vec{\mu}_i$. Must have shape `(k, n)`, where `k` is the number of data points
     and `n` is the number of dimensions of each point.
    :param cov: The covariance matrices $\mathbf{\Sigma}_i$ of the data vectors. Must have shape `(k, n, n)`.
     Use [`covariance_matrix`] to construct covariance matrices.
    :param n_samples: The maximum number of samples generated for each data point. If `None`, it is 100,000.
    :param n_accepted: The number of samples to be accepted for each data point.
     If `None`, it is the maximum of 100,000 and `n_samples`.
    :param seed: A seed for the random number generator. If `None`, a random seed is used.
    :param report: Whether to report the number of samples.
    :param kwargs: Additional keyword arguments are accepted for compatibility with
     [`generate_collinear_points_py`] and ignored.
    :returns: (p, n_accepted, n_samples) The generated data vectors $\vec{p}_i$ with shape `(n_accepted, k, n)`
     and the number of accepted and generated samples.
    """
    mean, cov = np.ascontiguousarray(mean, dtype=float), np.ascontiguousarray(cov, dtype=float)
    size = mean.shape[0]
    dim = mean.shape[1]

    n_samples = n_samples or 100000
    n_accepted = n_accepted or max(100000, n_samples)
    _n_samples, _n_accepted = int(n_samples), int(n_accepted)

    if seed is None:
        user_seed, seed = False, 0
    else:
        user_seed, seed = True, int(seed)

    n_target = _n_accepted
    x = np.zeros((_n_accepted, size, dim), dtype=float)
    n_accepted_c = c_size_t(_n_accepted)
    n_samples_c = c_size_t(_n_samples)
    dll.gen_collinear(
        x.ctypes.data_as(c_double_p),
        mean.ctypes.data_as(c_double_p),
        cov.ctypes.data_as(c_double_p),
        c_size_t_p(n_accepted_c),
        c_size_t(size),
        c_size_t(dim),
        c_size_t_p(n_samples_c),
        c_bool(user_seed),
        c_size_t(seed),
        c_bool(report),
    )
    if n_accepted_c.value < n_target:
        x = x[: n_accepted_c.value]
    return x, n_accepted_c.value, n_samples_c.value
