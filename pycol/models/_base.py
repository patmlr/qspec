# -*- coding: utf-8 -*-
"""
pycol.models._base

Created on 14.03.2022

@author: Patrick Mueller

Base classes for lineshape models.
"""

import numpy as np

from pycol.tools import merge_intervals

__all__ = ['Model', 'Empty', 'NPeak', 'Offset', 'Amplifier', 'Custom', 'YPars', 'Listed', 'Summed', 'Linked']


np_version = np.version.version.split('.')


def _is_unc(fix):
    if isinstance(fix, float):
        return True
    if not isinstance(fix, str):
        return False

    j, k = fix.find('('), fix.find(')')
    try:
        if j == -1 or k == -1:
            raise ValueError('Not a value with uncertainty.')
        _ = '{}({})'.format(float(fix[:j]), float(fix[j+1:k]))
    except ValueError:
        return False
    return True


def _val_fix_to_val(val, fix):
    if isinstance(fix, float):
        return val
    j = fix.find('(')
    return float(fix[:j])


def _fix_to_unc(fix):
    if isinstance(fix, float):
        return fix
    j, k = fix.find('('), fix.find(')')
    return float(fix[j+1:k])


def _args_ordered(args, order):
    return [args[i] for i in order]


def _poly(x, *args):
    return np.sum([args[n] * x ** n for n in range(len(args))], axis=0)


class Model:
    def __init__(self, model=None):
        self.model = model
        self.type = 'Model'

    def __call__(self, x, *args, **kwargs):
        return self.evaluate(x, *self.update_args(args), **kwargs)

    def evaluate(self, x, *args, **kwargs):  # Reimplement this function in subclasses.
        pass

    def _add_arg(self, name, val, fix, link):
        if name in self.names:
            raise ValueError('Parameter {} already exists.'.format(name))
        self.names.append(name)
        self.vals.append(val)
        self.fixes.append(fix)
        self.links.append(link)
        self.expressions.append('args[{}]'.format(self._index))

        self.p[name] = self._index

        self._index += 1
        self._size += 1

    @property
    def description(self):
        label = ''
        super_model = self
        while super_model is not None:
            if isinstance(super_model, Listed):
                label += super_model.type + '[0].'
                super_model = super_model.models[0]
            else:
                label += super_model.type + '.'
                super_model = super_model.model
        return label[:-1]

    @property
    def size(self):
        """
        :returns: The number of parameters required by the model.
        """
        return self._size

    @property
    def dx(self):
        return 0.1 if self.model is None else self.model.dx

    @property
    def error(self):
        return self._error

    @error.setter
    def error(self, value):
        self._error = value

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        self._model = value
        if self._model is None:
            self.names, self.vals, self.fixes, self.links = [], [], [], []
            self.expressions = []
            self.p = {}
            self._index = 0
            self._size = 0
            self._error = ''
        else:
            self.names, self.vals, self.fixes, self.links = \
                self._model.names, self._model.vals, self._model.fixes, self._model.links
            self.expressions = self._model.expressions
            self.p = self._model.p
            self._index = len(self._model.names)
            self._size = len(self._model.names)
            self._error = self._model.error

    def get_pars(self):
        return zip(self.names, self.vals, self.fixes, self.links)

    def set_pars(self, pars, force=False):
        for i, p in enumerate(pars):
            self.set_val(i, p[0], force=force)
            self.set_fix(i, p[1], force=force)
            self.set_link(i, p[2], force=force)

    def set_vals(self, vals, force=False):
        for i, val in enumerate(vals):
            self.set_val(i, val, force=force)

    def set_fixes(self, fixes, force=False):
        for i, fix in enumerate(fixes):
            self.set_fix(i, fix, force=force)

    def set_links(self, links, force=False):
        for i, link in enumerate(links):
            self.set_link(i, link, force=force)

    def set_val(self, i, val, force=False):
        if force or isinstance(val, int) or isinstance(val, float):
            if self.model is None:
                self.vals[i] = val
            else:
                self.model.set_val(i, val, force=True)  # Set val for all sub-models
                # to ensure set_val is called for a ListedModel if one is part of the sub-models.
                # This is only needed for the vals since these may be predicted by the specific model.
                # Everything else is handled by the top-model.

    def set_fix(self, i, fix, force=False):
        if force:
            self.fixes[i] = fix
            return
        if isinstance(fix, int) or isinstance(fix, float):
            if isinstance(fix, bool):
                fix = bool(fix)
            elif fix <= 0 or np.isinf(fix):
                fix = fix <= 0
            else:
                fix = float(fix)
            expr = 'args[{}]'.format(i)
        elif isinstance(fix, str):
            j, k = fix.find('('), fix.find(')')
            try:
                if j == -1 or k == -1:
                    raise ValueError('Not a value with uncertainty.')
                fix = '{}({})'.format(float(fix[:j]), float(fix[j+1:k]))
                expr = 'args[{}]'.format(i)
            except ValueError:
                _fix = fix
                for j, name in enumerate(self.names):
                    _fix = _fix.replace(name, 'eval(self.expressions[{}])'.format(j))
                expr = _fix
                try:
                    self._eval_zero_division(self.vals, expr)
                except (ValueError, TypeError, SyntaxError, NameError) as e:
                    print('Invalid expression for parameter \'{}\': {}. Got a {}.'.format(self.names[i], fix, repr(e)))
                    return
        elif isinstance(fix, list):
            if len(fix) == 0:
                fix = [0, 1]
            elif len(fix) == 1:
                fix = [0, fix[0]]
            else:
                fix = fix[:2]
            expr = 'args[{}]'.format(i)
        else:
            return
        temp_expr = self.expressions[i]
        temp_fix = self.fixes[i]
        self.expressions[i] = compile(expr, '<string>', 'eval', optimize=2)  # Compile beforehand to save time.
        self.fixes[i] = fix
        try:
            self.update_args(self.vals)
        except RecursionError as e:
            print('Expressions form a loop. Got a {}.'.format(repr(e)))
            self.expressions[i] = temp_expr
            self.fixes[i] = temp_fix

    def set_link(self, i, link, force=False):
        if force:
            self.links[i] = link
            return
        if isinstance(link, int) or isinstance(link, float):
            self.links[i] = bool(link)

    def _eval_zero_division(self, args, expr):
        try:
            return eval(expr, {}, {'self': self, 'args': args})
        except ZeroDivisionError:
            return 0.

    def update_args(self, args):
        return tuple(self._eval_zero_division(args, expr) for expr in self.expressions)

    def update(self):
        self.set_vals(self.update_args(self.vals), force=True)

    def min(self):
        return -1. if self.model is None else self.model.min()

    def max(self):
        return 1. if self.model is None else self.model.max()

    def intervals(self):
        return [[self.min(), self.max()]] if self.model is None else self.model.intervals()

    def x(self):
        return np.concatenate([np.arange(i[0], i[1], self.dx, dtype=float) for i in self.intervals()], axis=0)

    def fit_prepare(self):
        bounds = (-np.inf, np.inf)
        fixed = [fix for fix in self.fixes]
        b_lower = []
        b_upper = []
        _bounds = False
        for i, fix in enumerate(self.fixes):
            if isinstance(fix, bool):
                b_lower.append(-np.inf)
                b_upper.append(np.inf)
            elif _is_unc(fix):
                b_lower.append(-np.inf)
                b_upper.append(np.inf)
                fixed[i] = False
            elif isinstance(fix, list):
                _bounds = True
                b_lower.append(fix[0])
                b_upper.append(fix[1])
                fixed[i] = False
            elif isinstance(fix, str):
                b_lower.append(-np.inf)
                b_upper.append(np.inf)
                fixed[i] = True
                _fix = fix
            else:
                raise TypeError('The type {} of the element {} in self.fixes is not supported.'
                                .format(type(fix), fix))
        if _bounds:
            bounds = (b_lower, b_upper)
        return fixed, bounds


class Empty(Model):
    def __init__(self):
        super().__init__(model=None)
        self.type = 'Empty'

    def evaluate(self, x, *args, **kwargs):
        return np.zeros_like(x)


class NPeak(Model):
    def __init__(self, model, n_peaks=1):
        super().__init__(model=model)
        self.type = 'NPeak'
        self.n_peaks = int(n_peaks)
        for n in range(self.n_peaks):
            self._add_arg('x{}'.format(n), 0., n == 0, False)
            self._add_arg('p{}'.format(n), 1., n == 0, False)

    def evaluate(self, x, *args, **kwargs):
        return np.sum([args[self.model.size + 2 * n + 1]
                       * self.model.evaluate(x - args[self.model.size + 2 * n], *args[:self.model.size])
                       for n in range(self.n_peaks)], axis=0)

    def min(self):
        min_center = min(self.vals[self.p['x{}'.format(n)]] for n in range(self.n_peaks))
        return min_center + self.model.min()

    def max(self):
        max_center = max(self.vals[self.p['x{}'.format(n)]] for n in range(self.n_peaks))
        return max_center + self.model.max()

    def intervals(self):
        return merge_intervals([[i[0] + self.vals[self.model.size + 2 * n],
                                 i[1] + self.vals[self.model.size + 2 * n]]
                                for i in self.model.intervals() for n in range(self.n_peaks)])


class Offset(Model):
    def __init__(self, model=None, x_cuts=None, offsets=None):
        """
        :param model: The model the offset will be added to. If None, the offset will be added to zero.
        :param x_cuts: x values where to cut the x-axis.
        :param offsets: A list of maximally considered polynomial orders for each slice.
         The list must have length len(x_cuts) + 1.
        """
        super().__init__(model=model)
        self.type = 'Offset'
        if x_cuts is None:
            x_cuts = []
        self.x_cuts = sorted(list(x_cuts))
        self.offsets = offsets
        if self.offsets is None:
            self.offsets = [0]
        if len(self.offsets) != len(self.x_cuts) + 1:
            raise ValueError('The parameter offset must be a list of size \'len(x_cuts) + 1\''
                             ' and contain the maximally considered polynomial order for each slice.')

        self.offset_map = []
        self.offset_masks = []
        self.update_on_call = True

        self.gen_offset_map()

    def evaluate(self, x, *args, **kwargs):
        if self._model is None:
            return self._offset(x, *args)
        return self.model.evaluate(x, *args[:self.model.size]) + self._offset(x, *args)

    def set_x_cuts(self, x_cuts):
        x_cuts = list(x_cuts)
        if len(x_cuts) != len(self.x_cuts):
            raise ValueError('\'x_cuts\' must not change its size.')
        self.x_cuts = sorted(list(x_cuts))

    def _offset(self, x, *args):
        if self.update_on_call:
            self.gen_offset_masks(x)
        ret = np.zeros_like(x)
        for i, mask in enumerate(self.offset_masks):
            ret[mask] = _poly(x[mask], *_args_ordered(args, self.offset_map[i]))
        return ret

    def gen_offset_map(self):
        self.offset_map = []
        for i, n in enumerate(self.offsets):
            self.offset_map.append([])
            for k in range(n + 1):
                self.offset_map[-1].append(self._index)
                self._add_arg('off{}e{}'.format(i, k), 0., False, False)

    """ Preprocessing """

    def gen_offset_masks(self, x):
        self.offset_masks = []
        for x0, x1 in zip([np.min(x) - 1., ] + self.x_cuts, self.x_cuts + [np.max(x) + 1., ]):
            x_mean = 0.5 * (x0 + x1)
            self.offset_masks.append(np.abs(x - x_mean) < x1 - x_mean)

    def guess_offset(self, x, y):
        for i, mask in enumerate(self.offset_masks):
            self.vals[self.p['off{}e0'.format(i)]] = 0.5 * (y[mask][0] + y[mask][-1])
            try:
                self.vals[self.p['off{}e1'.format(i)]] = (y[mask][-1] - y[mask][0]) / (x[mask][-1] - x[mask][0])
            except KeyError:
                return


class Amplifier(Model):
    def __init__(self, order=None):
        super().__init__(model=None)
        self.type = 'Amplifier'
        if order is None:
            order = 1
        self.order = order
        for n in range(order + 1):
            self._add_arg('a{}'.format(n), 1. if n == 1 else 0., False, False)
        self._min = -10
        self._max = 10

    def evaluate(self, x, *args, **kwargs):
        self._min = np.min(x)
        self._max = np.max(x)
        return _poly(x, *args)

    @property
    def dx(self):
        return 1e-2

    def min(self):
        return self._min

    def max(self):
        return self._max


class Custom(Model):
    def __init__(self, model=None, parameters=None):
        super().__init__(model=model)
        self.type = 'Custom'
        if parameters is None:
            parameters = []
        self.parameters = parameters

        for p in self.parameters:
            self._add_arg(p, 0., False, False)

    def evaluate(self, x, *args, **kwargs):
        if self.model is None:
            return np.array(args, dtype=float)
        return self.model.evaluate(x, *args[:self.model.size], **kwargs)


class YPars(Model):
    def __init__(self, model):
        super().__init__(model=model)
        self.type = 'YPars'

        self.p_y = [i for i, fix in enumerate(self.model.fixes) if _is_unc(fix)]

    def evaluate(self, x, *args, **kwargs):
        return np.concatenate([self.model.evaluate(x, *args, **kwargs),
                               np.array([args[p_y] for p_y in self.p_y], dtype=float)], axis=0)


class Listed(Model):
    def __init__(self, models, labels=None):
        super().__init__(model=None)
        self.type = 'Listed'

        self.models = models  # models is just a reference to a list defined somewhere else.
        self.labels = labels
        if self.labels is None:
            if len(self.models) == 1:
                self.labels = ['', ]
            else:
                self.labels = ['__{}'.format(i) for i in range(len(self.models))]

        self.slices = []
        self.model_map = []
        self.index_map = []
        for i, (model, label) in enumerate(zip(self.models, self.labels)):
            self.slices.append(slice(self._index, self._index + model.size, 1))
            for j, (name, val, fix, link) in enumerate(model.get_pars()):
                self.model_map.append(i)
                self.index_map.append(j)
                if isinstance(fix, str) and not _is_unc(fix):
                    for _name in model.names:
                        fix = fix.replace(_name, '{}{}'.format(_name, label))
                self._add_arg('{}{}'.format(name, label), val, fix, link)

    def set_val(self, i, val, force=False):
        super().set_val(i, val, force=force)
        if i < len(self.model_map):
            self.models[self.model_map[i]].set_val(self.index_map[i], self.vals[i], force=True)

    def set_fix(self, i, fix, force=False):
        super().set_fix(i, fix, force=force)
        if i < len(self.model_map):
            self.models[self.model_map[i]].set_fix(self.index_map[i], self.fixes[i], force=True)

    def set_link(self, i, link, force=False):
        super().set_link(i, link, force=force)
        if i < len(self.model_map):
            self.models[self.model_map[i]].set_link(self.index_map[i], self.links[i], force=True)

    def inherit_vals(self):
        self.set_vals([val for model in self.models for val in model.vals], force=True)

    def inherit_fixes(self):
        self.set_fixes([fix for model in self.models for fix in model.fixes], force=True)

    def inherit_links(self):
        self.set_links([link for model in self.models for link in model.links], force=True)


class Summed(Listed):
    def __init__(self, models, labels=None):
        super().__init__(models, labels=labels)
        self.type = 'Summed'

        self.indices_add = []
        for n, (model, label) in enumerate(zip(self.models, self.labels)):
            self.indices_add.append([self._index, self._index + 1])
            self._add_arg('center{}'.format(label), 0., False, False)
            self._add_arg('int{}'.format(label), 1., False, False)

    def evaluate(self, x, *args, **kwargs):
        return np.sum([args[i[1]] * model.evaluate(x - args[i[0]], *args[_slice], **kwargs)
                       for model, _slice, i in zip(self.models, self.slices, self.indices_add)], axis=0)

    @property
    def dx(self):
        return min(model.dx for model in self.models)

    def min(self):
        return min(model.min() for model in self.models)

    def max(self):
        return max(model.max() for model in self.models)

    def intervals(self):
        return merge_intervals([[i[0] + self.vals[j[0]], i[1] + self.vals[j[0]]]
                                for model, j in zip(self.models, self.indices_add) for i in model.intervals()])


class Linked(Listed):
    def __init__(self, models):
        super().__init__(models, labels=None)
        self.type = 'Linked'

        for i, (name, val, fix, link) in enumerate(self.get_pars()):
            if link and (not fix or isinstance(fix, list) or _is_unc(fix)):
                _name = name[:name.rfind('__')]
                for j, model in enumerate(self.models):
                    if j < self.model_map[i] and _name in model.names:
                        self.set_fix(i, '{}__{}'.format(_name, j))
                        break

    def evaluate(self, x, *args, **kwargs):
        return np.concatenate(tuple(model.evaluate(_x, *args[_slice], **kwargs)
                                    for model, _slice, _x in zip(self.models, self.slices, x)), axis=0)

    def set_fix(self, i, fix, force=False):
        super(Listed, self).set_fix(i, fix, force=force)

    def set_link(self, i, link, force=False):
        super(Listed, self).set_link(i, link, force=force)
