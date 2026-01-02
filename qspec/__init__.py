"""
qspec
=====

A physics toolbox for laser spectroscopy.
"""

from qspec import models, simulate
from qspec.algebra import *
from qspec.analyze import *
from qspec.physics import *
from qspec.qtypes import (
    HasGetItem,
    HasShape,
    asarray,
    asarray_none,
    cast,
    cast_sympy,
    has_getitem,
    has_shape,
    is_scalar,
    is_sympy_expr,
    quant,
)
from qspec.stats import *
from qspec.tools import *
