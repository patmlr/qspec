"""
Created on 20.02.2022

@author: Patrick Mueller
"""

import os
import numpy as np
import sqlite3
from scipy.stats import norm
import itertools as it
from PyQt5.QtCore import QObject, pyqtSignal

from qspec.tools import print_colored
from qspec.analyze import curve_fit, const
from qspec.models._base import Linked
from qspec.models._fit import fit


COL_ACOL_CONFIG = {'enabled': False, 'rule': 'acca / caac', 'parameter': 'center',
                   'iterate': 3, 'volt': 1., 'mhz': 1., 'save_voltage': True, 'mc': False, 'mc_size': 100000,
                   'show_results': False, 'save_results': True, 'file': 'ColAcol_{db}_{run}.txt'}


class Fitter(QObject):

    finished = pyqtSignal()

    def __init__(self, models, meas, st, iso, config=None, run=None):
        """
        :param models: A list of models.
        :param meas: A list of SpecData objects.
        :param st: A list of scaler and track info.
        :param iso: A list of isotopes used for the axis conversion of each SpecData object.
        :param config: A dictionary with information for the fit.
        :param run: The run of the fitter. This is just used as an info. The default value is ''.
        """
        super().__init__()
        self.models = models
        self.meas = meas
        self.st = st
        self.iso = iso
        self.config = config
        if self.config is None:
            self.config = dict(routine='curve_fit', absolute_sigma=False, unc_from_fit=False, guess_offset=True,
                               cov_mc=False, samples_mc=100, arithmetics='', save_to_disk=False, norm_scans=False,
                               summed=False, linked=False, col_acol_config=COL_ACOL_CONFIG)
        self.run = '' if run is None else run
        self.size = len(self.meas)
        self.n_scaler = min(min(meas.nrScalers if isinstance(meas.nrScalers, list) else [meas.nrScalers])
                            for meas in self.meas)  # The minimum number of scalers for all files and tracks.

        self.routines = {'curve_fit', }

        self.sizes = []
        self.x_raw, self.x, self.y, self.yerr = [], [], [], []
        self.y_norm = []
        self.popt, self.pcov, self.info = [], [], {}
        self.gen_data()

    def get_pars(self, i):
        return self.models[i].get_pars()

    def set_val(self, i, j, val, force=False):
        self.models[i].set_val(j, val, force=force)

    def set_fix(self, i, j, fix, force=False):
        self.models[i].set_fix(j, fix, force=force)

    def set_link(self, i, j, link, force=False):
        self.models[i].set_link(j, link, force=force)

    def get_meas_x_in_freq(self, i):
        meas, iso = self.meas[i], self.iso[i]
        return [[Ph.volt_to_rel_freq(x, iso.q if iso.q else 1, iso.mass, meas.laserFreq, iso.freq, meas.col)
                 for x in x_track] for x_track in meas.x]

    def gen_y_sigma(self, linked, i):
        if self.config['unc_from_fit'] and self.y_norm:
            y_norm = np.concatenate(self.y_norm, axis=0) if linked else self.y_norm[i]

            def y_sigma(x, y, y_fit, *params):
                return np.sqrt(y_fit * y_norm)

            return y_sigma
        if linked:
            return self.yerr
        return self.yerr[i]

    @staticmethod
    def _yerr_from_array(a):
        yerr = np.ones_like(a)
        mask = a != 0
        yerr[mask] = np.sqrt(a[mask])
        return yerr

    def _gen_yerr(self, meas, st, n=10000):
        """
        This should only be called for real spectra once per meas in 'gen_data'.

        :param meas: A spec_data object.
        :param st: The scaler and track info.
        :param n: The number of samples to estimate the uncertainties in the 'function' mode.
        :returns: None.
        """
        cts = [np.array(t, dtype=float) for t in meas.cts]
        track_slice = slice(None, None, 1)
        if isinstance(st[1], int) and st[1] > -1:
            track_slice = slice(st[1], st[1] + 1, 1)

        if not self.config['arithmetics']:
            self.config['arithmetics'] = '[{}]'.format(', '.join(st[0]))
        if self.config['arithmetics'][0] == '[':
            indexes = np.array(eval(self.config['arithmetics']), dtype=int)
            cts_sum = [np.sum(t[indexes, :], axis=0) for t in cts]
            cts_d = [self._yerr_from_array(t) for t in cts_sum]
            cts_norm = [np.ones_like(t, dtype=float) for t in cts_sum]
            if self.config['norm_scans']:
                cts_norm = [np.full_like(t, 1 / n, dtype=float) for t, n in zip(cts_sum, meas.nrScans)]
                cts_sum = [t / n for t, n in zip(cts_sum, meas.nrScans)]
                cts_d = [t / n for t, n in zip(cts_d, meas.nrScans)]
            self.y.append(np.concatenate(cts_sum[track_slice], axis=0))
            self.yerr.append(np.concatenate(cts_d[track_slice], axis=0))
            self.y_norm.append(np.concatenate(cts_norm[track_slice], axis=0))
        else:
            cts_d = [self._yerr_from_array(t) for t in cts]
            if self.config['norm_scans']:
                cts = [t / n for t, n in zip(cts, meas.nrScans)]
                cts_d = [t / n for t, n in zip(cts_d, meas.nrScans)]
            cts = np.concatenate(cts[track_slice], axis=1)
            cts_d = np.concatenate(cts_d[track_slice], axis=1)
            y_mean = {'s{}'.format(i): cts[i] for i in range(self.n_scaler)
                      if 's{}'.format(i) in self.config['arithmetics']}
            y_samples = {'s{}'.format(i): norm.rvs(loc=cts[i], scale=cts_d[i], size=(n, cts[i].size))
                         for i in range(self.n_scaler) if 's{}'.format(i) in self.config['arithmetics']}
            # noinspection PyTypeChecker
            self.y.append(eval(self.config['arithmetics'], y_mean))
            self.yerr.append(np.std(eval(self.config['arithmetics'], y_samples), axis=0, ddof=1))

    def _gen_yerr_legacy_2(self, meas, st, n=10000):
        """
        This should only be called for real spectra once per meas in 'gen_data'.

        :param meas: A spec_data object.
        :param st: The scaler and track info.
        :param n: The number of samples to estimate the uncertainties in the 'function' mode.
        :returns: None.
        """
        cts = [np.array(t, dtype=float) / n_scans if self.config['norm_scans']
               else np.array(t, dtype=float) for t, n_scans in zip(meas.cts, meas.nrScans)]
        cts = np.concatenate(cts, axis=1)

        if not self.config['arithmetics']:
            self.config['arithmetics'] = '[{}]'.format(', '.join(st[0]))
        if self.config['arithmetics'][0] == '[':
            indexes = np.array(eval(self.config['arithmetics']), dtype=int)
            self.y.append(np.sum(cts[indexes], axis=0))
            self.yerr.append(np.sqrt(np.sum(cts[indexes, :] ** 2, axis=0)))
        else:
            cts_d = [np.sqrt(np.array(t, dtype=float)) / n_scans if self.config['norm_scans']
                     else self._yerr_from_array(np.array(t, dtype=float)) for t, n_scans in zip(meas.cts, meas.nrScans)]
            cts_d = np.concatenate(cts_d, axis=1)
            y_mean = {'s{}'.format(i): cts[i] for i in range(self.n_scaler)
                      if 's{}'.format(i) in self.config['arithmetics']}
            y_samples = {'s{}'.format(i): norm.rvs(loc=cts[i], scale=cts_d[i], size=(n, cts[i].size))
                         for i in range(self.n_scaler) if 's{}'.format(i) in self.config['arithmetics']}
            # noinspection PyTypeChecker
            self.y.append(eval(self.config['arithmetics'], y_mean))
            self.yerr.append(np.std(eval(self.config['arithmetics'], y_samples), axis=0, ddof=1))

    def _gen_yerr_legacy_1(self, meas, st, data, n=10000):
        """
        This should only be called for real spectra once per meas in 'gen_data'.

        :param meas: A spec_data object.
        :param st: The scaler and track info.
        :param data: The data as returned by spec_data.getArithSpec.
        :param n: The number of samples to estimate the uncertainties in the 'function' mode.
        :returns: None.
        """
        if self.config['arithmetics'] is None or 's' not in self.config['arithmetics']:
            self.yerr.append(self._yerr_from_array(data[1]))
        else:
            if st[1] == -1:
                cts = [np.array([i for i in it.chain(*(t[scaler] for t in meas.cts))]) for scaler in st[0]]
            else:
                cts = [np.array(meas.cts[st[1]][scaler]) for scaler in st[0]]
            y_samples = {'s{}'.format(i): norm.rvs(
                loc=cts[i], scale=self._yerr_from_array(cts[i]), size=(n, cts[i].size))
                         for i in range(self.n_scaler) if 's{}'.format(i) in self.config['arithmetics']}
            # noinspection PyTypeChecker
            self.yerr.append(np.std(eval(self.config['arithmetics'], y_samples), axis=0, ddof=1))

    def gen_data(self):
        """
        :returns: x_volt, x, y, yerr. The combined sorted data of the given measurements and fitting options.
        """
        self.sizes = []
        self.x_raw, self.x, self.y, self.yerr, self.y_norm = [], [], [], [], []
        for meas, st, iso in zip(self.meas, self.st, self.iso):
            data = meas.getArithSpec(*st, function=self.config['arithmetics'], eval_on=True)
            self.sizes.append(data[0].size)
            self.x_raw.append(data[0])
            if meas.seq_type == 'kepco':
                self.x.append(data[0])
                self.y.append(data[1])
                self.yerr.append(data[2])
            else:
                if 'CounterDrift' in meas.scan_dev_dict_tr_wise[0]['name']:
                    volt = meas.accVolt
                    laser_freq = data[0]
                else:
                    volt = data[0]
                    laser_freq = meas.laserFreq

                if self.config['x_axis'] in ['ion frequencies', 'lab frequencies']:
                    self.x.append(Ph.volt_to_rel_freq(volt, iso.q if iso.q else 1, iso.mass, laser_freq, iso.freq,
                                                      meas.col, lab_frame=self.config['x_axis'] == 'lab frequencies'))
                elif self.config['x_axis'] == 'DAC voltages':
                    self.x.append(np.array([i for i in it.chain(*meas.x_dac_volt)]))
                else:
                    self.x.append(data[0])
                self._gen_yerr(meas, st)
                # self._gen_yerr(meas, st, data)
        if all(model.type == 'Offset' for model in self.models):
            self.gen_x_cuts()

    def gen_x_cuts(self):
        for model, meas, iso, x in zip(self.models, self.meas, self.iso, self.x):
            if not model.x_cuts:
                model.gen_offset_masks(x)
                continue
            x_min = np.array([_x[0] if _x[0] <= _x[-1] else _x[-1] for _x in meas.x])
            # Array of the tracks lowest voltages
            x_max = np.array([_x[-1] if _x[0] <= _x[-1] else _x[0] for _x in meas.x])
            # Array of the tracks highest voltages
            order = np.argsort(x_min)  # Find ascending order of the lowest voltages
            x_min = x_min[order]  # apply order to the lowest voltages
            x_max = x_max[order]  # apply order to the highest voltages
            # cut at the mean between the highest voltage and the corresponding lowest voltage of the next track.
            # Iteration goes over the sorted tracks and only non-overlapping tracks get a unique offset parameter.
            x_cuts = [0.5 * float(x_max[i] + x_min[i + 1]) for i in range(len(model.x_cuts))]
            if any(x0 > x1 for x0, x1 in zip(x_cuts[:-1], x_cuts[1:])):
                print_colored('WARNING', 'Tracks are overlapping in file {}.'
                                         ' Cannot use \'offset per track\' option'.format(meas.file))
                continue
            x_cuts = [Ph.volt_to_rel_freq(_x, iso.q if iso.q else 1, iso.mass, meas.laserFreq, iso.freq, meas.col)
                      for _x in x_cuts]
            model.set_x_cuts(x_cuts)

    def get_routine(self):
        if self.config['routine'] not in self.routines:
            raise ValueError('The fit routine {} is not one of the available routines {}.'
                             .format(self.config['routine'], self.routines))
        return eval(self.config['routine'])

    def reduced_chi2(self, i=None):
        """ Calculate the reduced chi square """
        if i is None:
            return [self.reduced_chi2(i) for i in range(self.size)]
        else:
            y_err = self.yerr[i]
            sigma_y = self.gen_y_sigma(False, i)
            if callable(sigma_y):
                y_err = sigma_y(self.x[i], self.y[i], self.models[i](self.x[i], *self.popt))
            return np.sum(self.residuals(i) ** 2 / y_err ** 2) / self.n_dof(i)

    def n_dof(self, i=None):
        """ Calculate number of degrees of freedom """
        if i is None:
            return [self.n_dof(i) for i in range(self.size)]
        else:
            # if bounds are given instead of boolean, write False to fixed bool list.
            fixed_sum = sum(f if isinstance(f, bool) else False for f in self.models[i].fixes)
            return self.x_raw[i].size - (self.models[i].size - fixed_sum)

    def residuals(self, i=None):
        """ Calculate the residuals of the current parameter set """
        if i is None:
            return [self.residuals(i) for i in range(self.size)]
        else:
            model = self.models[i]
            y_model = model(self.x[i], *model.vals)
            return self.y[i] - y_model

    def fit_batch(self):
        info = dict(warn=[], err=[], chi2=[])
        popt, pcov = [], []
        for i, (meas, model, x, y, yerr) in enumerate(zip(self.meas, self.models, self.x, self.y, self.yerr)):
            sigma_y = self.gen_y_sigma(False, i)
            _popt, _pcov, _info = fit(
                model, x, y, sigma_y=sigma_y, report=True, routine=self.config['routine'],
                absolute_sigma=self.config['absolute_sigma'], guess_offset=self.config['guess_offset'],
                mc_sigma=self.config['samples_mc'] if self.config['cov_mc'] else 0)
            popt.append(_popt)
            pcov.append(_pcov)
            for k in ['warn', 'err']:
                if _info[k]:
                    info[k].append(i)
            info['chi2'].append(_info['chi2'])
        color = 'OKGREEN'
        if len(info['warn']) > 0:
            color = 'WARNING'
        if len(info['err']) > 0:
            color = 'FAIL'
        print_colored(color, '\nFits completed, success in {} / {}.'.format(self.size - len(info['warn']), self.size))
        return popt, pcov, info

    def fit_summed(self):
        return None, None, {}

    def fit_linked(self):
        info = dict(warn=[], err=[], chi2=[])
        model = Linked(self.models)
        sigma_y = self.gen_y_sigma(True, None)
        _popt, _pcov, _info = fit(
            model, self.x, self.y, sigma_y=sigma_y, report=True, routine=self.config['routine'],
            absolute_sigma=self.config['absolute_sigma'], guess_offset=self.config['guess_offset'],
            mc_sigma=self.config['samples_mc'] if self.config['cov_mc'] else 0)

        popt = [_popt[_slice] for _slice in model.slices]
        pcov = [_pcov[_slice, _slice] for _slice in model.slices]
        for k in ['warn', 'err']:
            if _info[k]:
                info[k] = list(range(self.size))
        info['chi2'] = [self.reduced_chi2(i) for i in range(self.size)]  # _info['chi2']

        color = 'OKGREEN'
        if len(info['warn']) > 0:
            color = 'WARNING'
        if len(info['err']) > 0:
            color = 'FAIL'
        print_colored(color, '\nLinked fit completed, success in {} / {}.'.format(self.size - len(info['warn']), self.size))
        return popt, pcov, info

    def save_linked_fit(self):
        pass

    def _check_col_acol(self):
        rule = self.config['col_acol_config']['rule']
        false_flag = False
        if rule == 'acca / caac':
            col = -1
            index = 0
            for meas in self.meas:
                if index % 2 == 1 and meas.col == col:
                    false_flag = True
                    break
                elif index == 2 and meas.col != col:
                    false_flag = True
                    break
                col = meas.col
                index = (index + 1) % 4
            if index != 0:
                false_flag = True
        elif rule == 'free':
            size_col = len([0 for meas in self.meas if meas.col])
            size_acol = len([0 for meas in self.meas if not meas.col])
            if size_col != size_acol or size_col + size_acol != len(self.meas):
                false_flag = True
        else:
            size_col = len([0 for meas in self.meas if meas.col])
            size_acol = len([0 for meas in self.meas if not meas.col])
            if size_col == 0 or size_acol == 0:
                false_flag = True
        if false_flag:
            print_colored('WARNING', 'Col/Acol rule \'{}\' not fulfilled. Stopping fit.'.format(rule))
            return False
        return True

    def _gen_col_acol(self):
        rule = self.config['col_acol_config']['rule']
        if rule == 'acca / caac' or rule == 'free':
            return [[i, ] for i, meas in enumerate(self.meas) if meas.col], \
                [[i, ] for i, meas in enumerate(self.meas) if not meas.col]
        else:
            return [[i for i, meas in enumerate(self.meas) if meas.col], ], \
                [[i for i, meas in enumerate(self.meas) if not meas.col], ]

    def _average_col_acol(self, c_a):
        par = self.config['col_acol_config']['parameter']
        p = self.models
        if len(c_a) == 1:
            i = c_a[0]
            p = self.models[i].p[par]
            return (self.popt[i][p], np.sqrt(np.diag(self.pcov[i])[p])), \
                (self.meas[i].laserFreq, self.meas[i].laserFreq_d)
        f_ion, f_laser = [], []
        w_ion, w_laser = [], []
        for i in c_a:
            f_ion.append(self.popt[i][p])
            w_ion.append(1 / np.diag(self.pcov[i])[p])
            f_laser.append(self.meas[i].laserFreq)
            w_laser.append(1 / self.meas[i].laserFreq_d ** 2)
        f_ion, w_ion = np.average(f_ion, weights=w_ion, returned=True)
        f_ion = [f_ion, np.sqrt(1 / w_ion)]
        f_laser, w_laser = np.average(f_laser, weights=w_laser, returned=True)
        f_laser = [f_laser, np.sqrt(1 / w_laser)]
        return f_ion, f_laser

    def _calc_abs_freq(self, c, a):
        if self.config['x_axis'] != 'ion frequencies':
            raise NotImplementedError('Col/Acol fit is only implemented for ion frequencies x-axis.')
        q = self.iso[c[0]].q
        mass = [self.iso[c[0]].mass, self.iso[c[0]].mass_d]
        freq = self.iso[c[0]].freq
        f_ion_c, f_laser_c = self._average_col_acol(c)
        f_ion_a, f_laser_a = self._average_col_acol(a)

        if self.config['col_acol_config']['mc']:
            size = self.config['col_acol_config']['mc_size']
            mass_sample = norm.rvs(loc=mass[0], scale=mass[1], size=size)
            f_ion_c_sample = norm.rvs(loc=f_ion_c[0], scale=f_ion_c[1], size=size)
            f_laser_c_sample = norm.rvs(loc=f_laser_c[0], scale=f_laser_c[1], size=size)
            f_ion_a_sample = norm.rvs(loc=f_ion_a[0], scale=f_ion_a[1], size=size)
            f_laser_a_sample = norm.rvs(loc=f_laser_a[0], scale=f_laser_a[1], size=size)

            # For u, the exact laser freq and mass that was previously used
            # for the transformation has to be used, not the samples.
            u_c = Ph.rel_freq_to_volt(f_ion_c_sample, q, mass[0], f_laser_c[0], freq, True)
            u_a = Ph.rel_freq_to_volt(f_ion_a_sample, q, mass[0], f_laser_a[0], freq, True)
            u = 0.5 * (u_c + u_a)
            du = 0.5 * (u_c - u_a)

            df = f_ion_c_sample - f_ion_a_sample
            f = Ph.col_acol_ec_ea(f_laser_c_sample, f_laser_a_sample, u_c * q, u_a * q, mass_sample)

            u_c = [np.mean(u_c), np.std(u_c, ddof=1)]
            u_a = [np.mean(u_a), np.std(u_a, ddof=1)]
            df = [np.mean(df), np.std(df, ddof=1)]
            f = [np.mean(f), np.std(f, ddof=1)]
            u = [np.mean(u), np.std(u, ddof=1)]
            du = [np.mean(du), np.std(du, ddof=1)]
        else:
            # For u, the exact laser freq and mass that was previously used
            # for the transformation has to be used, without uncertainties.
            u_c = [Ph.rel_freq_to_volt(f_ion_c[0], q, mass[0], f_laser_c[0], freq, True),
                   Ph.rel_freq_to_volt_d(f_ion_c[0], f_ion_c[1], q, mass[0], f_laser_c[0], freq)]
            u_a = [Ph.rel_freq_to_volt(f_ion_a[0], q, mass[0], f_laser_a[0], freq, True),
                   Ph.rel_freq_to_volt_d(f_ion_a[0], f_ion_a[1], q, mass[0], f_laser_a[0], freq)]

            u_d = 0.5 * np.sqrt(u_c[1] ** 2 + u_a[1] ** 2)
            u = [0.5 * (u_c[0] + u_a[0]), u_d]
            du = [0.5 * (u_c[0] - u_a[0]), u_d]

            df = [f_ion_c[0] - f_ion_a[0], np.sqrt(f_ion_c[1] ** 2 + f_ion_a[1] ** 2)]
            f = [Ph.col_acol_ec_ea(f_laser_c[0], f_laser_a[0], u_c[0] * q, u_a[0] * q, mass[0]),
                 Ph.col_acol_ec_ea_d(f_laser_c[0], f_laser_c[1], f_laser_a[0], f_laser_a[1],
                                     u_c[0] * q, u_c[1] * q, u_a[0] * q, u_a[1] * q, mass[0], mass[1])]

        return {'MeasNr': None, 'AbsFreq': f[0], 'AbsFreq_d': f[1], 'LaserFreqCol': f_laser_c[0],
                'LaserFreqCol_d': f_laser_c[1], 'LaserFreqAcol': f_laser_a[0], 'LaserFreqAcol_d': f_laser_a[1],
                'UCol': u_c[0], 'UCol_d': u_c[1], 'UAcol': u_a[0], 'UAcol_d': u_a[1],
                'U': u[0], 'U_d': u[1], 'dU': du[0], 'dU_d': du[1], 'df': df[0], 'df_d': df[1]}

    def _calc_col_acol(self, col, acol):
        results = []
        for i, (c, a) in enumerate(zip(col, acol)):
            result = self._calc_abs_freq(c, a)
            result['MeasNr'] = i + 1
            results.append(result)
        return results

    def _optimize_acc_volt(self, results, col, acol):
        par = self.config['col_acol_config']['parameter']
        con = sqlite3.connect(self.iso[0].db)
        for r, c, a in zip(results, col, acol):
            q = self.iso[c[0]].q
            df = r['df'] / 2
            du_c = df / Ph.doppler_e_d(r['AbsFreq'], 0., r['UCol'] * q, q,
                                       self.iso[c[0]].mass, 0., 0., rest_frame=True)
            du_a = df / Ph.doppler_e_d(r['AbsFreq'], 0., r['UAcol'] * q, q,
                                       self.iso[c[0]].mass, 0., np.pi, rest_frame=True)
            du = np.mean([du_c, du_a])

            for _c in c:
                for tr_ind, track in enumerate(self.meas[_c].x):
                    if tr_ind == 0:
                        self.models[_c].vals[self.models[_c].p[par]] -= df
                        self.meas[_c].accVolt += du / self.meas[_c].voltDivRatio.get('accVolt', 1.0)
                    self.meas[_c].x[tr_ind] += du
                    if self.config['col_acol_config']['save_voltage']:
                        with con:
                            con.execute('UPDATE Files SET accVolt = ? WHERE file = ?',
                                        (self.meas[_c].accVolt, self.meas[_c].file))
            
            for _a in a:
                for tr_ind, track in enumerate(self.meas[_a].x):
                    if tr_ind == 0:
                        self.models[_a].vals[self.models[_a].p[par]] += df
                        self.meas[_a].accVolt += du / self.meas[_a].voltDivRatio.get('accVolt', 1.0)
                    self.meas[_a].x[tr_ind] += du
                    if self.config['col_acol_config']['save_voltage']:
                        with con:
                            con.execute('UPDATE Files SET accVolt = ? WHERE file = ?',
                                        (self.meas[_a].accVolt, self.meas[_a].file))
        con.commit()
        con.close()
        self.gen_data()

    def save_col_acol(self, results):
        header_list = ['MeasNr', 'AbsFreq', 'AbsFreq_d', 'LaserFreqCol', 'LaserFreqCol_d',
                       'LaserFreqAcol', 'LaserFreqAcol_d', 'UCol', 'UCol_d', 'UAcol', 'UAcol_d',
                       'U', 'U_d', 'dU', 'dU_d', 'df', 'df_d']

        f = np.array([r['AbsFreq'] for r in results])
        df = np.array([r['AbsFreq_d'] for r in results])

        med = np.around(np.median(f), decimals=3)
        med_0 = np.around(med - np.percentile(f, 15.8655254), decimals=3)
        med_1 = np.around(np.percentile(f, 84.1344746) - med, decimals=3)

        av = np.around(np.average(f), decimals=3)
        av_d = np.around(np.std(f, ddof=1) / np.sqrt(f.size), decimals=3)

        wav, wav_d = np.average(f, weights=1 / df ** 2, returned=True)
        wav, wav_d = np.around(wav, decimals=3), np.around(np.sqrt(1 / wav_d), decimals=3)

        popt, pcov = curve_fit(const, np.linspace(-1, 1, f.size), f, p0=[wav], sigma=df, absolute_sigma=False)
        fit, fit_d = np.around(popt[0], decimals=3), np.around(np.sqrt(np.diag(pcov)[0]), decimals=3)

        f0, f0_d = fit, np.max([av_d, wav_d, fit_d])

        db = os.path.split(self.iso[0].db)[1]
        db = db[:-(db[::-1].find('.') + 1)]
        filename = self.config['col_acol_config']['file'].replace('{db}', db).replace('{run}', self.run)
        files = [meas.file for meas in self.meas]
        with open(os.path.join(os.path.dirname(self.iso[0].db), filename), 'w') as file:
            file.write('# {}\n'.format(repr(files)))
            file.write('# {}\n'.format(', '.join(header_list)))
            for r in results:
                file.write('{}\n'.format(', '.join([str(np.around(r[h], decimals=3)) for h in header_list])))
            file.write('#\n# median: {} +{} / -{} MHz\n'.format(med, med_1, med_0))
            file.write('# average: {} +/- {} MHz\n'.format(av, av_d))
            file.write('# weighted average: {} +/- {} MHz\n'.format(wav, wav_d))
            file.write('# const fit: {} +/- {} MHz\n'.format(fit, fit_d))
            file.write('# f0: {} +/- {} MHz'.format(f0, f0_d))

    def fit_col_acol(self):
        if not self._check_col_acol():
            warn = list(range(self.size))  # Issue warnings for all files.
            errs = list(range(self.size))
            chi2 = [0., ] * self.size
            popt = [np.array(model.vals) for model in self.models]
            pcov = [np.zeros((popt[-1].size, popt[-1].size)) for _ in self.models]
            self.popt, self.pcov, self.info = popt, pcov, dict(warn=warn, err=errs, chi2=chi2)
            return

        iterate = self.config['col_acol_config']['iterate']
        volt = self.config['col_acol_config']['volt']
        mhz = self.config['col_acol_config']['mhz']

        col, acol = self._gen_col_acol()

        # TODO: Set equal voltages for the c and a files beforehand?

        self._fit()
        results = self._calc_col_acol(col, acol)
        i = 0
        while (max(abs(r['df']) for r in results) > mhz) and i < iterate:
            self._optimize_acc_volt(results, col, acol)
            self._fit()
            results = self._calc_col_acol(col, acol)
            i += 1

        if self.config['col_acol_config']['save_results']:
            self.save_col_acol(results)

    def save_fit(self):
        db = os.path.split(self.iso[0].db)[1]
        db = db[:-(db[::-1].find('.') + 1)]
        path = os.path.join(os.path.dirname(self.iso[0].db), 'fit_results')
        if not os.path.exists(path):
            try:
                os.makedirs(path)
            except Exception as e:
                print('Saving directory has not been created. Writing permission in DB directory?\n'
                      'Error msg: {}'.format(e))
                return
        files = [os.path.splitext(meas.file)[0] for meas in self.meas]
        for j, (file, model, _popt, _pcov, chi2) in enumerate(zip(files, self.models, self.popt, self.pcov, self.info['chi2'])):
            with open(os.path.join(path, '{}_{}_{}_fit.txt'.format(db, file, self.run)), 'w') as f:
                f.write('# File: \'{}\'\n'.format(file))
                f.write('# Run: \'{}\'\n'.format(self.run))
                model_description = model.description
                if self.config['linked']:
                    model_description = 'Linked[{}].{}'.format(j, model_description)
                f.write('# Model: {}\n'.format(model_description))
                f.write('# Red. chi2: {}\n# \n'.format(np.around(chi2, decimals=2)))
                f.write('# Index, Parameter, Value, Uncertainty, Fixed, Linked\n')
                for i, (name, val, unc, fixed, linked) in enumerate(
                        zip(model.names, _popt, np.sqrt(np.diag(_pcov)), model.fixes, model.links)):
                    f.write('{}, \'{}\', {}, {}, {}, {}\n'.format(
                        i, name, val, unc, fixed, linked if self.config['linked'] else False))
            np.save(os.path.join(path, '{}_{}_{}_cov'.format(db, file, self.run)), _pcov)

    def _fit(self):
        """
        :returns: popt, pcov. The optimal parameters and their covariance matrix.
        :raises ValueError: If 'routine' is not in {'curve_fit', }.
        """
        if self.config['summed']:
            self.popt, self.pcov, self.info = self.fit_summed()
        elif self.config['linked']:
            self.popt, self.pcov, self.info = self.fit_linked()
        else:
            self.popt, self.pcov, self.info = self.fit_batch()

    def fit(self):
        """
        :raises ValueError: If 'routine' is not in {'curve_fit', }.
        """
        if self.config['col_acol_config']['enabled']:
            self.fit_col_acol()
        else:
            self._fit()
        if self.config['save_to_disk']:
            self.save_fit()
        self.finished.emit()
