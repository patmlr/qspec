# -*- coding: utf-8 -*-
"""
qspec._analyze_cpp
==================

Classes and methods for the 'analyze' module using the Python/C++ interface.
"""

import numpy as np

from qspec._types import *
from qspec._cpp import *

__all__ = ['generate_collinear_points_cpp']


class MultivariateNormal:
    def __init__(self, mean: array_like, cov: array_like, instance=None):
        self.instance = instance
        if self.instance is None:
            mean, cov = np.ascontiguousarray(mean, dtype=float), np.ascontiguousarray(cov, dtype=float)
            self.instance = dll.multivariatenormal_construct(
                mean.ctypes.data_as(c_double_p), cov.ctypes.data_as(c_double_p), c_size_t(mean.size))

    def __del__(self):
        dll.multivariatenormal_destruct(self.instance)

    @property
    def size(self):
        return dll.multivariatenormal_size(self.instance)

    def rvs(self):
        ret = np.zeros(self.size, dtype=float)
        dll.multivariatenormal_rvs(self.instance, ret.ctypes.data_as(c_double_p))
        return ret


def generate_collinear_points_cpp(mean: ndarray, cov: ndarray, n_samples: int = None, n_accepted: int = None,
                                  seed: int = None, report: bool = None, **kwargs):
    """
    :param mean: The data vectors. Must have shape (k, n), where k is the number of data points
     and n is the number of dimensions of each point.
    :param cov: The covariance matrices of the data vectors. Must have shape (k, n, n).
     Use 'covariance_matrix' to construct covariance matrices.
    :param n_samples: The number of samples generated for each data point.
    :param n_accepted: The number of samples to be accepted for each data point.
    :param seed: A seed for the random number generator.
    :param report: Whether to report the number of samples.
    :param kwargs: Additional keyword arguments.
    :returns: The randomly generated data vectors p with shape (n_accepted, k ,n) aligned along a straight line
     and the number of accepted and generated samples.
    """
    mean, cov = np.ascontiguousarray(mean, dtype=float), np.ascontiguousarray(cov, dtype=float)
    size = mean.shape[0]
    dim = mean.shape[1]

    if n_samples is None and n_accepted is None:
        n_samples, n_accepted = 100000, 100000
    elif n_samples is None:
        n_samples = 0
    elif n_accepted is None:
        n_accepted = max(100000, n_samples)

    if seed is None:
        user_seed, seed = False, 0
    else:
        user_seed, seed = True, int(seed)

    n_target = n_accepted
    x = np.zeros((n_accepted, size, dim), dtype=float)
    n_accepted = c_size_t(n_accepted)
    n_samples = c_size_t(n_samples)
    dll.gen_collinear(x.ctypes.data_as(c_double_p), mean.ctypes.data_as(c_double_p), cov.ctypes.data_as(c_double_p),
                      c_size_t_p(n_accepted), c_size_t(size), c_size_t(dim),
                      c_size_t_p(n_samples), c_bool(user_seed), c_size_t(seed), c_bool(report))
    if n_accepted.value < n_target:
        x = x[:n_accepted.value]
    return x, n_accepted.value, n_samples.value
