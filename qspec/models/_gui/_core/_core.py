"""
Created on 18.02.2022

@author: Patrick Mueller

@description:
This is the main script for the 'PolliFit/SpectraFit' tab.
Here parameters are loaded from and saved to the DB, lineshape models are generated and DBIsotope objects are created.

The connection to the GUI and all the GUI logic is implemented in the 'SpectraFitUi' class
in the 'PolliFit.Gui.SpectraFitUi' script.

The data fitting is handled by the 'Fitter' class in the 'PolliFit.Fitter' script,
to which all 'fit options' are forwarded.

The plotting of the data is done in the 'plot_model_fit' function in the 'PolliFit.MPLPlotter' script.

CUSTOM LINESHAPE MODELS: To create custom lineshape models follow the examples in 'qspec.models'
and create them in the 'PolliFit.Models' folder similar to the existing example.
"""

import os
import ast
import sqlite3
import numpy as np

from qspec.models._spectrum import SPECTRA
from qspec.models._convolved import CONVOLVE

from qspec.models._gui._core._db import select_from_db
from qspec.models._gui._core._data import Fitter

from qspec.models._gui._core._plot import plot_model_fit
from Tilda.PolliFit.DBIsotope import DBIsotope
import Tilda.PolliFit.Measurement.MeasLoad as MeasLoad


LEGACY_PARS = {'lor': 'Gamma', 'gamma': 'Gamma', 'gau': 'sigma'}


def _legacy_par_to_par(par):
    if par in list(LEGACY_PARS.keys()):
        return LEGACY_PARS[par]
    return par


def execute(cur, command, *args):
    try:
        cur.execute(command, *args)
    except sqlite3.OperationalError as e:
        print_colored('FAIL', repr(e))


def _get_iso_par_from_dict(par_dict, par):
    try:
        return par_dict[par]
    except KeyError:
        return par_dict[par[:par.find('(')]]


def gen_splitter_model(config, iso):
    if config['qi_config']['qi'] and (config['hf_config']['enabled_l'] or config['hf_config']['enabled_u']):
        pass
    elif config['qi_config']['qi'] and iso.I > 0:
        return mod.HyperfineQI, (iso.I, iso.Jl, iso.Ju, iso.name, config['qi_config']['qi_path'])
    elif config['hf_config']['enabled_l'] or config['hf_config']['enabled_u']:
        return mod.HyperfineMixed, (iso.I, iso.Jl, iso.Ju, iso.name, config['hf_config'])
    else:
        return mod.Hyperfine, (iso.I, iso.Jl, iso.Ju, iso.name)
    raise ValueError('Specified splitter model not available.')


def gen_splitter_models(config, iso):
    _splitter, _args = gen_splitter_model(config, iso)
    splitter = [_splitter, ]
    args = [_args, ]
    _iso = iso.m
    while _iso is not None:
        _splitter, _args = gen_splitter_model(config, _iso)
        splitter.append(_splitter)
        args.append(_args)
        _iso = _iso.m
    return splitter, args


def gen_model(config, iso, spectra_fit=None):
    splitter, args = gen_splitter_models(config, iso)

    if config['lineshape'] in mod.SPECTRA:
        shape = eval('mod.{}'.format(config['lineshape']))
    elif config['lineshape'] in Spectrum.SPECTRA:
        shape = eval('Spectrum.{}'.format(config['lineshape']))
    else:
        raise ValueError('Lineshape model \'{}\' is not available.'.format(config['lineshape']))

    splitter_model = mod.SplitterSummed([
        _splitter(shape(), *_args) for _splitter, _args in zip(splitter, args)])
    if spectra_fit is not None:
        spectra_fit.splitter_models.append(splitter_model)

    npeaks_model = mod.NPeak(model=splitter_model, n_peaks=config['npeaks'])

    if config['convolve'] != 'None':
        if config['convolve'] in mod.CONVOLVE:
            npeaks_model = eval('mod.{}Convolved'.format(config['convolve']))(model=npeaks_model)
        elif config['convolve'] in Convolved.CONVOLVE:
            npeaks_model = eval('Convolved.{}Convolved'.format(config['convolve']))(model=npeaks_model)
        else:
            raise ValueError('Convolution kernel \'{}\' is not available.'.format(config['convolve']))

    offset = config['offset_order']
    x_cuts = None
    if config['offset_per_track']:
        x_cuts = [float(i) for i in range(len(offset) - 1)]  # The correct x_cuts are not known at this point.
        # The actual x_cuts are set after the fitter is created.
    else:
        offset = [offset[0], ]
    offset_model = mod.Offset(model=npeaks_model, x_cuts=x_cuts, offsets=offset)

    return offset_model


class SpectraFit:
    def __init__(self, db, files, runs, configs, index_config,
                 x_axis='ion frequencies', routine='curve_fit', absolute_sigma=False, unc_from_fit=False,
                 guess_offset=True, cov_mc=False, samples_mc=100, arithmetics=None, save_to_disk=False,
                 norm_scans=False, summed=False, linked=False, col_acol_config=None, save_to_db=False, x_as_freq=True,
                 fig_save_format='.png', zoom_data=False, fmt='.k', fontsize=10):
        self.db = db
        self.files = files
        self.runs = runs  # TODO: Use only a single run per instance.
        self.configs = configs
        self.index_config = index_config

        self.file_paths = self.load_filepaths()

        self.x_axis = x_axis
        self.routine = routine
        self.unc_from_fit = unc_from_fit
        self.absolute_sigma = absolute_sigma
        self.cov_mc = cov_mc
        self.samples_mc = samples_mc
        self.guess_offset = guess_offset
        self.arithmetics = arithmetics
        self.save_to_disk = save_to_disk
        self.norm_scans = norm_scans
        self.summed = summed
        self.linked = linked
        self.col_acol_config = col_acol_config
        self.save_to_db = save_to_db

        self.x_as_freq = x_as_freq
        self.fig_save_format = fig_save_format
        self.zoom_data = zoom_data
        self.fmt = fmt
        self.fontsize = fontsize

        self.file_types = {'.xml'}
        self.ascii_path = os.path.join(os.path.normpath(os.path.dirname(self.db)), 'saved_plots')
        self.plot_path = os.path.join(os.path.normpath(os.path.dirname(self.db)), 'saved_plots')

        self.splitter_models = []
        self.reset_model = None
        self.fitter = None
        self.gen_fitter()

    def _execute(self, command, *args):
        con = sqlite3.connect(self.db)
        cur = con.cursor()
        execute(cur, command, *args)
        con.commit()
        con.close()
        
    def load_filepaths(self):
        file_paths = []
        for file in self.files:
            var = TiTs.select_from_db(self.db, 'filePath', 'Files', [['file'], [file]], caller_name=__name__)
            if var is None:
                print(str(file) + ' not found in DB.')
            else:
                file_paths.append(os.path.join(os.path.dirname(self.db), var[0][0]))
                
        print('\nFile paths:')
        for i, path in enumerate(file_paths):
            print('{}: {}'.format(str(i).zfill(int(np.log10(len(file_paths)))), path))
        return file_paths

    def gen_model(self, config, iso):
        return gen_model(config, iso, self)

    def gen_config(self):
        return dict(x_axis=self.x_axis, routine=self.routine, absolute_sigma=self.absolute_sigma,
                    unc_from_fit=self.unc_from_fit, guess_offset=self.guess_offset, cov_mc=self.cov_mc,
                    samples_mc=self.samples_mc, arithmetics=self.arithmetics, save_to_disk=self.save_to_disk,
                    norm_scans=self.norm_scans, summed=self.summed, linked=self.linked,
                    col_acol_config=self.col_acol_config)

    def gen_fitter(self):
        if not self.files:
            self.fitter = None
            return

        self.splitter_models = []
        models, meas, st, iso = [], [], [], []
        for path, file, run, config in zip(self.file_paths, self.files, self.runs, self.configs):
            var = TiTs.select_from_db(self.db, 'isoVar, lineVar, scaler, track', 'Runs', [['run'], [run]],
                                      caller_name=__name__)
            if not var:
                raise ValueError('Run \'{}\' cannot be selected.'.format(run))

            linevar = var[0][1]
            softw_gates = self.load_trs(file, run)
            meas.append(MeasLoad.load(path, self.db, softw_gates=softw_gates))

            # st: tuple of PMTs and tracks from selected run
            st_str = [var[0][2], var[0][3]]
            n_scaler = min(meas[-1].nrScalers if isinstance(meas[-1].nrScalers, list) else [meas[-1].nrScalers])
            self.arithmetics = st_str[0].strip().lower()
            try:
                eval(self.arithmetics, {'s{}'.format(i): 4.2 for i in range(n_scaler)})
            except (ValueError, TypeError, SyntaxError, NameError) as e:
                raise ValueError('Run \'{}\' cannot be selected, due to {}.'.format(run, repr(e)))
            if 's' in self.arithmetics:
                st.append([[i for i in range(n_scaler)], ast.literal_eval(st_str[1])])
            else:
                st.append([ast.literal_eval(s) for s in st_str])

            size = len(config['offset_order'])
            n_tracks = len(meas[-1].x)
            if size < n_tracks:
                config['offset_order'] = config['offset_order'] + [max(config['offset_order']), ] * (n_tracks - size)
            elif size > n_tracks:
                config['offset_order'] = config['offset_order'][:(n_tracks - size)]

            if isinstance(meas[-1], MeasLoad.XMLImporter):
                if meas[-1].seq_type == 'kepco':
                    iso.append(None)
                    models.append(mod.Amplifier(order=config['offset_order'][0]))
                else:
                    iso.append(DBIsotope(self.db, meas[-1].type, lineVar=linevar))
                    models.append(self.gen_model(config, iso[-1]))
            else:
                raise ValueError('File type not supported. The supported types are {}.'.format(self.file_types))

        self.fitter = Fitter(models, meas, st, iso, self.gen_config(), run=self.runs[0])
        self.load_pars()
        self.reset_model = [[p for p in model.get_pars()] for model in models]

    def reset_st(self):
        st = []
        arithmetics = None
        for run, meas in zip(self.runs, self.fitter.meas):
            var = TiTs.select_from_db(self.db, 'isoVar, lineVar, scaler, track', 'Runs', [['run'], [run]],
                                      caller_name=__name__)
            if not var:
                raise ValueError('Run \'{}\' cannot be selected.'.format(run))
            # st: tuple of PMTs and tracks from selected run
            st_str = [var[0][2], var[0][3]]
            n_scaler = min(meas.nrScalers if isinstance(meas.nrScalers, list) else [meas.nrScalers])
            arithmetics = st_str[0].strip().lower()
            try:
                eval(arithmetics, {'s{}'.format(i): 4.2 for i in range(n_scaler)})
            except (ValueError, TypeError, SyntaxError, NameError) as e:
                raise ValueError('Run \'{}\' cannot be selected,'
                                 ' due to error in \'scaler\' column: {}.'.format(run, repr(e)))
            if 's' in arithmetics:
                st.append([[i for i in range(n_scaler)], ast.literal_eval(st_str[1])])
            else:
                st.append([ast.literal_eval(s) for s in st_str])
        self.arithmetics = arithmetics
        if self.fitter is not None:
            self.fitter.config['arithmetics'] = arithmetics
            self.fitter.st = st

    def set_softw_gates(self, i, softw_gates):
        if i is None:
            for j in range(len(self.fitter.meas)):
                self.fitter.meas[j].softw_gates = softw_gates[:len(self.fitter.meas[j].softw_gates)]
                self.fitter.meas[j] = TiTs.gate_specdata(self.fitter.meas[j])
        else:
            self.fitter.meas[i].softw_gates = softw_gates
            self.fitter.meas[i] = TiTs.gate_specdata(self.fitter.meas[i])
        self.fitter.gen_data()

    def set_norm_scans(self):
        if self.fitter is None:
            return
        self.fitter.config['norm_scans'] = self.norm_scans
        self.fitter.gen_data()

    def fit(self):
        if self.fitter is None:
            return
        self.fitter.config = self.gen_config()
        self.fitter.fit()
        return self.finish_fit()

    def finish_fit(self):
        popt, pcov, info = self.fitter.popt, self.fitter.pcov, self.fitter.info
        if self.save_to_db:
            self.save_fits_to_db(popt, pcov, info)
        # if self.save_figure:
        #     self.save_fits_as_fig(popt, pcov, info)
        return popt, pcov, info

    def get_pars(self, i):
        return self.fitter.get_pars(i)

    def set_val(self, i, j, val, force=False):
        self.fitter.set_val(i, j, val, force=force)

    def set_fix(self, i, j, fix, force=False):
        self.fitter.set_fix(i, j, fix, force=force)

    def set_link(self, i, j, link, force=False):
        self.fitter.set_link(i, j, link, force=force)

    def reset(self):
        if self.reset_model is None:
            return
        for i, reset_model in enumerate(self.reset_model):
            for j, (_, val, fix, link) in enumerate(reset_model):
                self.set_val(i, j, val)
                self.set_fix(i, j, fix)
                self.set_link(i, j, link)
            self.fitter.models[i].update()

    def load_trs(self, file, run, from_file=False):
        if from_file:
            return None
        softw_gates = (self.db, run)
        iso = TiTs.select_from_db(self.db, 'type', 'Files', [['file'], [file]], caller_name=__name__)
        pars = TiTs.select_from_db(self.db, 'softw_gates', 'FitPars', [['file', 'run'], [file, run]],
                                   caller_name=__name__)
        index = 0
        if pars is None:
            load_from_isotopes = True
            pars = TiTs.select_from_db(self.db, 'file, softw_gates', 'FitPars', [['run'], [run]], caller_name=__name__)
            if iso is not None and pars is not None:
                for i, par in enumerate(pars):
                    _iso = TiTs.select_from_db(self.db, 'type', 'Files', [['file'], [par[0]]], caller_name=__name__)
                    if _iso is None:
                        continue
                    if _iso[0][0] == iso:
                        index = i
                        load_from_isotopes = False
                        break
            if load_from_isotopes:
                return softw_gates
        try:
            softw_gates = ast.literal_eval(pars[index][-1])
        except ValueError:
            print('softw_gates could not be loaded from FitPars, loading from selected run.')
        return softw_gates

    def _pars_from_legacy_db(self, file, run):
        iso = TiTs.select_from_db(self.db, 'type', 'Files', [['file'], [file]], caller_name=__name__)
        if iso is None or not iso:
            return {}
        iso = iso[0][0]
        line = TiTs.select_from_db(self.db, 'lineVar', 'Runs', [['run'], [run]], caller_name=__name__)
        if line is None:
            return {}
        line = line[0][0]
        pars = TiTs.select_from_db(self.db, 'shape, fixShape', 'Lines', [['lineVar'], [line]], caller_name=__name__)
        fixes = ast.literal_eval(pars[0][1])
        pars = ast.literal_eval(pars[0][0])
        pars = {_legacy_par_to_par(par): (val, fixes.get(par, False), False) for par, val in pars.items()}
        if 'Gamma' in list(pars.keys()):
            pars['Gamma'] = (2 * pars['Gamma'][0], *pars['Gamma'][1:])
        columns = ['center', 'Al', 'Bl', 'Au', 'Bu', 'fixedArat', 'fixedBrat', 'm',
                   'fixedAl', 'fixedBl', 'fixedAu', 'fixedBu']

        m_flag = False
        while iso is not None and iso:
            pars_iso = TiTs.select_from_db(self.db, ', '.join(columns), 'Isotopes',
                                           [['iso'], [iso]], caller_name=__name__)
            if pars_iso is None:
                return {}
            pars_iso = {par: pars_iso[0][i] for i, par in enumerate(columns)}
            m = pars_iso['m']
            names = ['center', 'Al', 'Bl', 'Au', 'Bu']
            if m is not None or m_flag:
                m_flag = True
                names = ['{}({})'.format(par, iso) for par in names]
            _pars = {par: (_get_iso_par_from_dict(pars_iso, par),
                     pars_iso.get('fixed{}'.format(par[:2]), False), False) for par in names}
            pars = {**pars, **_pars}
            if pars_iso['fixedArat']:
                pars[names[3]] = (0., '{} * {}'.format(pars_iso[names[3]], names[1]), False)
            if pars_iso['fixedBrat']:
                pars[names[4]] = (0., '{} * {}'.format(pars_iso[names[4]], names[2]), False)
            iso = m
        return pars

    def _pars_from_db(self, file, run):
        iso = TiTs.select_from_db(self.db, 'type', 'Files', [['file'], [file]], caller_name=__name__)
        pars = TiTs.select_from_db(self.db, 'pars', 'FitPars', [['file', 'run'], [file, run]], caller_name=__name__)
        index = 0
        if pars is None:
            load_from_isotopes = True
            pars = TiTs.select_from_db(self.db, 'file, pars', 'FitPars', [['run'], [run]], caller_name=__name__)
            if iso is not None and pars is not None:
                for i, par in enumerate(pars):
                    _iso = TiTs.select_from_db(self.db, 'type', 'Files', [['file'], [par[0]]], caller_name=__name__)
                    if _iso is None:
                        continue
                    if _iso[0][0] == iso[0][0]:
                        index = i
                        load_from_isotopes = False
                        break

            # if pars is None:
            #     pars = TiTs.select_from_db(self.db, 'pars', 'FitPars', [['file'], [file]], caller_name=__name__)
            if load_from_isotopes:
                return self._pars_from_legacy_db(file, run)
        p = pars[index][-1].replace('(np.nan,', '(0.0,').replace('(nan,', '(0.0,') \
            .replace('(np.inf,', '(0.0,').replace('(inf,', '(0.0,')
        pars = ast.literal_eval(p)
        return pars

    def load_pars(self):
        self._execute('CREATE TABLE IF NOT EXISTS "FitPars"("file" TEXT, "run" TEXT, "softw_gates" TEXT, "config" TEXT,'
                      ' "pars" TEXT, PRIMARY KEY("file", "run"))')
        for i, (file, run) in enumerate(zip(self.files, self.runs)):
            pars = self._pars_from_db(file, run)
            reload_fix = []
            for j, (name, val, fix, link) in enumerate(self.get_pars(i)):
                par = pars.get(name, (val, fix, link))
                self.set_val(i, j, par[0])
                if isinstance(par[1], str):
                    self.set_fix(i, j, True)
                    reload_fix.append((j, par[1]))
                else:
                    self.set_fix(i, j, par[1])
                self.set_link(i, j, par[2])
            for j, fix in reload_fix:
                self.set_fix(i, j, fix)
            self.fitter.models[i].update()

    def save_pars(self):
        con = sqlite3.connect(self.db)
        cur = con.cursor()
        execute(cur, 'CREATE TABLE IF NOT EXISTS "FitPars"("file" TEXT, "run" TEXT, "softw_gates" TEXT, "config" TEXT,'
                     ' "pars" TEXT, PRIMARY KEY("file", "run"))')
        for i, (file, run, config) in enumerate(zip(self.files, self.runs, self.configs)):
            new_pars = {name: (val, fix, link) for (name, val, fix, link) in self.get_pars(i)}
            pars = TiTs.select_from_db(self.db, 'pars', 'FitPars', [['file', 'run'], [file, run]], caller_name=__name__)
            if pars is None:
                pars = {}
            else:
                pars = ast.literal_eval(pars[0][0].replace('(np.nan,', '(0.0,').replace('(nan,', '(0.0,')
                                        .replace('(np.inf,', '(0.0,').replace('(inf,', '(0.0,'))
            new_pars = {**pars, **new_pars}
            softw_gates = str(self.fitter.meas[i].softw_gates)
            if 'inf' in softw_gates:
                self.set_softw_gates(i, self.fitter.meas[i].softw_gates)
                softw_gates = str(self.fitter.meas[i].softw_gates)
            config['qi_config'].pop('qi_path', None)
            execute(cur, 'INSERT OR REPLACE INTO FitPars (file, run, softw_gates, config, pars)'
                         ' VALUES (?, ?, ?, ?, ?)', (file, run, softw_gates, str(config), str(new_pars)))
        con.commit()
        con.close()

    def save_fits_to_db(self, popt, pcov, info):
        con = sqlite3.connect(self.db)
        cur = con.cursor()
        for i, (file, run) in enumerate(zip(self.files, self.runs)):
            if i in info['err']:
                continue
            pars = {self.fitter.models[i].names[j]: (pt, np.sqrt(pc[j]), self.fitter.models[i].fixes[j])
                    for j, (pt, pc) in enumerate(zip(popt[i], pcov[i]))}
            execute(cur, 'INSERT OR REPLACE INTO FitRes (file, iso, run, rChi, pars) '
                         'VALUES (?, ?, ?, ?, ?)', (file, self.fitter.iso[i].name, run, info['chi2'][i], str(pars)))
        con.commit()
        con.close()

    def plot(self, index=None, clear=True, show=False, ascii_path='', plot_path=''):
        if self.fitter is None:
            return
        if clear:
            Plot.clear()

        fig = Plot.plot_model_fit(self.fitter, self.index_config if index is None else index, x_as_freq=self.x_as_freq,
                                  ascii_path=ascii_path, plot_path=plot_path, fig_save_format=self.fig_save_format,
                                  zoom_data=self.zoom_data, fmt=self.fmt, fontsize=self.fontsize)

        if show:
            Plot.show(True)
            fig.canvas.draw()

    """ Prints """

    def print_pars(self):
        print('Current parameters:')
        for i, file in enumerate(self.files):
            print('File: {}'.format(file))
            for pars in self.get_pars(i):
                print('\t'.join([str(p) for p in pars]))

    def print_files(self):
        print('\nFile paths:')
        for i, file in enumerate(self.files):
            print('{}: {}'.format(str(i).zfill(int(np.log10(len(self.files)))), file))
