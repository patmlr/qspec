import matplotlib.pyplot as plt
import numpy as np

from qspec._types import *
from qspec._cpp import *

__all__ = ['gen_collinear']


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


def gen_collinear(mean, cov, n):
    mean, cov = np.ascontiguousarray(mean, dtype=float), np.ascontiguousarray(cov, dtype=float)
    size = mean.shape[0]
    dim = mean.shape[1]
    x = np.zeros((n, size, dim), dtype=float)
    dll.gen_collinear(x.ctypes.data_as(c_double_p), mean.ctypes.data_as(c_double_p), cov.ctypes.data_as(c_double_p),
                      c_size_t(n), c_size_t(size), c_size_t(dim))
    return x


if __name__ == '__main__':
    # size = 10000
    # m, c = np.array([0, 1]), np.array([[1, 1], [1, 4]])
    # mvn = MultivariateNormal(m, c)
    # x = np.array([mvn.rvs() for _ in range(size)])
    # y = np.random.multivariate_normal(m, c, size=size)
    # plt.plot(*y.T, '.C0')
    # plt.plot(*x.T, '.C1')
    # plt.show()
    pass
