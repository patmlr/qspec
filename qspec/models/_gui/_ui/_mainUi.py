"""
Created on 18.02.2022

@author: Patrick Mueller
"""

import os
import ast
import shutil
from copy import deepcopy
# noinspection PyUnresolvedReferences
from numpy import inf
from PyQt5 import QtWidgets, QtCore
# noinspection PyProtectedMember, PyUnresolvedReferences
from matplotlib.axes._base import _process_plot_format

from qspec.models._spectrum import SPECTRA
from qspec.models._convolved import CONVOLVE

from qspec.models._gui._core._db import select_from_db, write_to_db, get_documents_dir
from qspec.models._gui._core._core import SpectraFit

from qspec.models._gui._ui.Ui_main import Ui_main
from qspec.models._gui._ui._hfsConfigUi import HFSConfigUi

colors = ['b', 'g', 'r', 'x', 'm', 'y', 'k']
inf_str = ['PINF', 'Infinity', 'infty', 'Inf', 'inf']


class MainUi(QtWidgets.QMainWindow, Ui_main):
    def __init__(self, db_path=None, data_path=''):
        super(MainUi, self).__init__()
        self.setupUi(self)

        self.hf_mixing_config_ui = None
        self.load_lineshapes()
        self.load_convolves()
        self.db_path = None
        self.data_path = ''

        self.index_config = 0
        self.index_load = 0
        self.index_marked = 0
        self.fig = None
        self.spectra_fit = None
        self.thread = QtCore.QThread()

        self.db_select(db_path)
        self.connect()
        self.show()

    def connect(self):
        """ Connect all the GUI elements. """
        # Files.
        self.b_db.clicked.connect(self.open_db)
        self.c_db.currentTextChanged[str].connect(self.db_select)
        self.b_select_all.clicked.connect(
            lambda checked: self.select_from_list(self.get_items(subset='all'), selected=None))
        self.b_select_favorites.clicked.connect(
            lambda checked: self.select_from_list(self.get_items(subset='fav'), selected=None))
        self.check_multi.stateChanged.connect(self.multi)
        sel_model = self.list_files.selectionModel()
        sel_model.selectionChanged.connect(self.set_index)

        # Parameters.
        self.tab_pars.cellChanged.connect(self.set_par_multi)

        self.b_load_pars.clicked.connect(self.load_pars)
        self.b_up.clicked.connect(self.up)
        self.b_down.clicked.connect(self.down)
        self.b_copy.clicked.connect(self.copy_pars)
        self.b_reset_pars.clicked.connect(self.reset_pars)
        self.b_save_pars.clicked.connect(self.save_pars)

        # Options
        self.c_preset.currentIndexChanged.connect(self.set_preset)
        self.b_preset_minus.clicked.connect(self.remove_preset)
        self.b_preset_plus.clicked.connect(self.add_preset)

        # Model
        self.c_lineshape.currentIndexChanged.connect(self.set_lineshape)
        self.c_convolve.currentIndexChanged.connect(self.set_convolve)
        self.s_npeaks.valueChanged.connect(self.set_npeaks)
        self.check_offset_per_track.stateChanged.connect(self.toggle_offset_per_track)
        self.edit_offset_order.editingFinished.connect(self.set_offset_order)
        self.check_qi.stateChanged.connect(self.toogle_qi)
        # self.check_hf_mixing.stateChanged.connect(self.toogle_hf_mixing)
        self.b_hf_mixing.clicked.connect(self.open_hf_mixing)
        self.b_racah.clicked.connect(self.set_racah)

        # Fit.
        self.c_routine.currentIndexChanged.connect(self.set_routine)
        self.check_chi2.stateChanged.connect(self.toggle_chi2)
        self.check_delta_f.stateChanged.connect(self.toggle_delta_f)
        self.check_guess_offset.stateChanged.connect(self.toggle_guess_offset)
        self.check_cov_mc.stateChanged.connect(self.toggle_cov_mc)
        self.s_samples_mc.valueChanged.connect(self.set_samples_mc)
        # self.edit_arithmetics.editingFinished.connect(self.set_arithmetics_toggle_delta_f)
        # self.check_arithmetics.stateChanged.connect(self.toggle_arithmetics)
        self.check_norm_scans.stateChanged.connect(
            lambda state, _suppress=False: self.toggle_norm_scans(suppress_plot=_suppress))
        self.check_summed.stateChanged.connect(self.toogle_summed)
        self.check_linked.stateChanged.connect(self.toogle_linked)
        self.check_save_to_db.stateChanged.connect(self.toggle_save_to_db)
        self.check_save_to_disk.stateChanged.connect(self.toggle_save_to_disk)
        # self.check_save_figure.stateChanged.connect(self.toggle_save_figure)

        # Plot.
        self.b_plot.clicked.connect(self.plot)
        self.b_save_ascii.clicked.connect(self.save_ascii)
        self.b_save_figure.clicked.connect(self.save_plot)
        self.c_fig.currentIndexChanged.connect(self.set_fig_save_format)
        self.check_zoom_data.stateChanged.connect(
            lambda state, _suppress=False: self.set_zoom_data(suppress_plot=_suppress))
        self.edit_fmt.editingFinished.connect(self.set_fmt)
        self.s_fontsize.editingFinished.connect(self.set_fontsize)

        # Action (Fit).
        # self.b_fit.clicked.connect(self.fit)
        self.b_fit.clicked.connect(self.fit_threaded)

    def open_db(self):
        if self.db_path is None:
            default_path = get_documents_dir()
        else:
            default_path = os.path.dirname(self.db_path)
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, 'Open database directory', default_path)
        if not folder:
            return
        db_list = [os.path.splitext(db)[0] for db in os.listdir(folder) if '.sqlite' in db]
        if not db_list:
            src = os.path.join(os.path.dirname(__file__), 'default.sqlite')
            dst = os.path.join(folder, f'{os.path.basename(folder)}.sqlite')
            shutil.copy(src, dst)
            db_list = [os.path.splitext(os.path.basename(dst))[0]]

        self.c_db.blockSignals(True)
        self.c_db.addItems(db_list)
        self.c_db.setCurrentIndex(0)
        self.c_db.blockSignals(False)

        self.db_select(os.path.join(folder, f'{self.c_db.currentText()}.sqlite'))

    def db_select(self, db_path):
        self.db_path = db_path
        if db_path is None:
            return

        self.load_presets()
        self.load_files()

        self.index_config = 0
        self.index_load = 0
        self.index_marked = 0
        self.fig = None
        self.spectra_fit = self.gen_spectra_fit()

    """ Files """

    def load_presets(self):
        presets = select_from_db(self.db_path, 'Presets', 'preset')
        if not presets:
            self.add_preset('preset 1')
            return

        self.c_preset.blockSignals(True)
        self.c_preset.clear()
        for i, p in enumerate(presets):
            self.c_preset.insertItem(i, p[0])
        self.c_preset.setCurrentIndex(0)
        self.c_preset.blockSignals(False)
        self.set_preset()

    def set_preset(self):
        pass

    def remove_preset(self):
        pass

    def add_preset(self, preset):
        self.c_preset.blockSignals(True)
        self.c_preset.addItem(preset)
        self.c_preset.setCurrentText(preset)
        self.c_preset.blockSignals(False)
        qi_config = dict(qi=self.check_qi.isChecked(), qi_path=os.path.dirname(self.db_path))
        hfs_config = dict(enabled_l=False, enabled_u=False, Jl=[0.5, ], Ju=[0.5, ],
                          Tl=[[1.]], Tu=[[1.]], fl=[[0.]], fu=[[0.]], mu=0.)
        model_config = dict(lineshape=self.c_lineshape.currentText(),
                            convolve=self.c_convolve.currentText(),
                            npeaks=self.s_npeaks.value(),
                            offset_per_track=self.check_offset_per_track.isChecked(),
                            offset_order=ast.literal_eval(self.edit_offset_order.text()),
                            qi_config=qi_config,
                            hfs_config=hfs_config)

        data_config = dict(data_path=self.data_path)

        fit_config = dict(routine=self.c_routine.currentText(),
                          absolute_sigma=not self.check_chi2.isChecked(),
                          unc_from_fit=self.check_delta_f.isChecked(),
                          guess_offset=self.check_guess_offset.isChecked(),
                          cov_mc=self.check_cov_mc.isChecked(),
                          samples_mc=self.s_samples_mc.value(),
                          norm_scans=self.check_norm_scans.isChecked(),
                          summed=self.check_summed.isChecked(),
                          linked=self.check_linked.isChecked(),
                          save_to_db=self.check_save_to_db.isChecked(),)

        plot_config = dict(plot_auto=self.check_auto.isChecked(),
                           fig_save_format=self.c_fig.currentText(),
                           zoom_data=self.check_zoom_data.isChecked(),
                           fmt=self.edit_fmt.text(),
                           fontsize=self.s_fontsize.value())

        config = dict(model=model_config, data=data_config, fit=fit_config, plot=plot_config)
        write_to_db(self.db_path, 'Presets', ['preset', 'config'], [preset, str(config)])

    def load_lineshapes(self):
        for i, spec in enumerate(SPECTRA):
            self.c_lineshape.insertItem(i, spec)
        self.c_lineshape.setCurrentText('Voigt')

    def load_convolves(self):
        for i, spec in enumerate(CONVOLVE):
            self.c_convolve.insertItem(i, spec)
        self.c_convolve.setCurrentText('None')

    def load_files(self):
        self.list_files.clear()
        # files = select_from_db(self.db_path, 'file', 'Files', [['type'], [self.c_iso.currentText()]], 'ORDER BY date')
        # for f in files:
        #     self.list_files.addItem(f[0])
        files = os.listdir(os.path.join(os.path.dirname(self.db_path), self.data_path))
        for f in files:
            self.list_files.addItem(f)
        self.gen_item_lists()

    def gen_item_lists(self):
        pass
        # self.items_col = []
        # self.items_acol = []
        # for i in range(self.list_files.count()):
        #     it = TiTs.select_from_db(self.db_path, 'colDirTrue', 'Files',
        #                              [['file'], [self.list_files.item(i).text()]], caller_name=__name__)
        #     if it is None:
        #         continue
        #     if it[0][0]:
        #         self.items_col.append(self.list_files.item(i))
        #     else:
        #         self.items_acol.append(self.list_files.item(i))

    def get_items(self, subset='all'):
        """
        :param subset: The label of the subset.
        :returns: The specified subset of items from the files list.
        """
        if subset == 'col':
            return self.items_col
        elif subset == 'acol':
            return self.items_acol
        else:
            return [self.list_files.item(i) for i in range(self.list_files.count())]

    def set_index(self):
        self.index_load = 0
        # items = self.list_files.selectedItems()
        # if not items:
        #     return
        # try:
        #     self.index_load = items.index(self.list_files.currentItem())
        # except ValueError:
        #     self.index_load = 0

    def mark_loaded(self, items):
        item = self.list_files.item(self.index_marked)
        if item is not None:
            item.setForeground(QtCore.Qt.GlobalColor.black)
        if items:
            for item in items:
                item.setForeground(QtCore.Qt.GlobalColor.black)
            items[self.index_load].setForeground(QtCore.Qt.GlobalColor.blue)
            self.index_marked = self.list_files.row(items[self.index_load])
            model_file = items[self.index_load].text()
            self.l_model_file.setText(model_file)
            self.index_config = self.index_load
            self.spectra_fit.index_config = self.index_load

    def mark_warn(self, warn):
        for i in warn:
            items = self.list_files.findItems(self.spectra_fit.files[i], QtCore.Qt.MatchFlag.MatchExactly)
            items[0].setForeground(QtCore.Qt.GlobalColor.yellow)

    def mark_errs(self, errs):
        for i in errs:
            items = self.list_files.findItems(self.spectra_fit.files[i], QtCore.Qt.MatchFlag.MatchExactly)
            items[0].setForeground(QtCore.Qt.GlobalColor.red)

    def get_selected_items(self):
        items = [self.list_files.item(i) for i in range(self.list_files.count())]
        selected = self.list_files.selectedItems()
        return sorted(selected, key=lambda item: items.index(item))

    def select_from_list(self, items, selected=None):
        """
        :param items: The set of items to select.
        :param selected: Whether to select or deselect the 'items'.
         If None, the 'items' are (de-)selected based on there current selection status.
        :returns: None.
        """
        # if not self.list_files.selectedItems():
        #     return
        if self.check_multi.isChecked():  # If multi selection is enabled, (de-)select all items of the set.
            if selected is None:
                selected = True
                selected_items = self.get_selected_items()
                if len(selected_items) > 0 and len(selected_items) == len(items) \
                        and all(item is selected_items[i] for i, item in enumerate(items)):
                    selected = False
            self.list_files.clearSelection()
            for item in items:
                item.setSelected(selected)
        else:  # If multi selection is disabled, select the next item of the set or deselect if there are none.
            if selected is None or selected:
                selected = True
                i0 = self.list_files.currentRow()
                i = (i0 + 1) % self.list_files.count()
                item = self.list_files.item(i)
                while item not in items:
                    if i == i0:
                        selected = False
                        item = self.list_files.item(i0)
                        break
                    i = (i + 1) % self.list_files.count()
                    item = self.list_files.item(i)
                self.list_files.setCurrentItem(item)
                item.setSelected(selected)
            else:
                self.list_files.currentItem().setSelected(False)
        self.list_files.setFocus()

    def multi(self):
        if self.check_multi.isChecked():
            self.list_files.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        else:
            self.list_files.clearSelection()
            self.list_files.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
            item = self.list_files.currentItem()
            if item is None:
                item = self.list_files.item(0)
            item.setSelected(True)
        self.list_files.setFocus()

    """ Parameters """

    def _gen_configs(self, files, presets):
        configs = []
        qi_config = dict(qi=self.check_qi.isChecked(), qi_path=os.path.dirname(self.db_path))
        hf_config = dict(enabled_l=False, enabled_u=False, Jl=[0.5, ], Ju=[0.5, ],
                         Tl=[[1.]], Tu=[[1.]], fl=[[0.]], fu=[[0.]], mu=0.)
        current_config = dict(lineshape=self.c_lineshape.currentText(),
                              convolve=self.c_convolve.currentText(),
                              npeaks=self.s_npeaks.value(),
                              offset_per_track=self.check_offset_per_track.isChecked(),
                              offset_order=ast.literal_eval(self.edit_offset_order.text()),
                              qi_config=qi_config,
                              hf_config=hf_config)
        for file, preset in zip(files, presets):
            config = select_from_db(self.db_path, 'config', 'FitPars', [['file', 'preset'], [file, preset]])
            if config is None:
                config = select_from_db(self.db_path, 'config', 'FitPars', [['preset'], [preset]])
            if config is None:
                config = current_config
            else:
                config = {**current_config, **ast.literal_eval(config[0][0])}
                if config['lineshape'] not in SPECTRA:
                    config['lineshape'] = 'Voigt'
                if config['convolve'] not in CONVOLVE:
                    config['convolve'] = 'None'
                config['qi_config'] = {**qi_config, **config['qi_config']}
                config['hf_config'] = {**hf_config, **config['hf_config']}
            configs.append(config)
        if configs:
            self.set_model_gui(configs[self.index_load])
            self.mark_loaded(self.get_selected_items())
        return configs

    def gen_spectra_fit(self):
        if self.spectra_fit is not None:
            if self.spectra_fit.fitter is not None:
                try:
                    self.thread.disconnect()  # Disconnect thread from the old fitter.
                except TypeError:  # A TypeError is thrown if there are no connections.
                    pass
                self.spectra_fit.fitter.deleteLater()  # Make sure the QObject, which lives in another thread,
                # is deleted before creating a new one.

        files = [f.text() for f in self.get_selected_items()]
        presets = [self.c_preset.currentText() for _ in self.list_files.selectedItems()]
        configs = self._gen_configs(files, presets)
        kwargs = dict(routine=self.c_routine.currentText(),
                      absolute_sigma=not self.check_chi2.isChecked(),
                      unc_from_fit=self.check_delta_f.isChecked(),
                      guess_offset=self.check_guess_offset.isChecked(),
                      cov_mc=self.check_cov_mc.isChecked(),
                      samples_mc=self.s_samples_mc.value(),
                      norm_scans=self.check_norm_scans.isChecked(),
                      summed=self.check_summed.isChecked(),
                      linked=self.check_linked.isChecked(),
                      save_to_db=self.check_save_to_db.isChecked(),
                      fig_save_format=self.c_fig.currentText(),
                      zoom_data=self.check_zoom_data.isChecked(),
                      fmt=self.edit_fmt.text(),
                      fontsize=self.s_fontsize.value())
        return SpectraFit(self.db_path, files, presets, configs, self.index_load, **kwargs)

    def load_pars(self, suppress_plot=False):
        if not self.list_files.selectedItems():
            return
        self.spectra_fit = self.gen_spectra_fit()
        if self.check_arithmetics.isChecked():
            self.update_arithmetics()
        else:
            self.set_arithmetics_toggle_delta_f(suppress_plot=True)
        self.update_pars(suppress_plot=suppress_plot)
        self.edit_offset_order.setText(str(self.spectra_fit.configs[self.index_config]['offset_order']))

    def update_pars(self, suppress_plot=False):
        if not self.list_files.selectedItems():
            return
        self.tab_pars.blockSignals(True)
        self.tab_pars.setRowCount(self.spectra_fit.fitter.models[self.index_config].size)
        if self.check_x_as_freq.isChecked():
            pars = self.spectra_fit.get_pars(self.index_config)
        else:
            pars = self.spectra_fit.get_pars(self.index_config)  # TODO get_pars_e()?

        for i, (name, val, fix, link) in enumerate(pars):
            w = QtWidgets.QTableWidgetItem(name)
            # noinspection PyUnresolvedReferences
            w.setFlags(w.flags() & ~QtCore.Qt.ItemIsEditable)
            self.tab_pars.setItem(i, 0, w)

            w = QtWidgets.QTableWidgetItem(str(val))
            self.tab_pars.setItem(i, 1, w)

            w = QtWidgets.QTableWidgetItem(str(fix))
            self.tab_pars.setItem(i, 2, w)

            w = QtWidgets.QTableWidgetItem(str(link))
            self.tab_pars.setItem(i, 3, w)
        self.tab_pars.blockSignals(False)
        self.plot_auto(suppress_plot)

    def update_vals(self, suppress_plot=False):  # Call only if table signals are blocked.
        for i, val in enumerate(self.spectra_fit.fitter.models[self.index_config].vals):
            self.tab_pars.item(i, 1).setText(str(val))
        self.plot_auto(suppress_plot)

    def update_fixes(self, suppress_plot=False):  # Call only if table signals are blocked.
        for i, fix in enumerate(self.spectra_fit.fitter.models[self.index_config].fixes):
            self.tab_pars.item(i, 2).setText(str(fix))
        self.plot_auto(suppress_plot)

    def update_links(self, suppress_plot=False):  # Call only if table signals are blocked.
        for i, link in enumerate(self.spectra_fit.fitter.models[self.index_config].links):
            self.tab_pars.item(i, 3).setText(str(link))
        self.plot_auto(suppress_plot)

    def display_index_load(self):
        items = [self.list_files.findItems(file, QtCore.Qt.MatchFlag.MatchExactly)[0]
                 for file in self.spectra_fit.files]
        sel_model = self.list_files.selectionModel()
        sel_model.blockSignals(True)
        self.list_files.clearSelection()
        for item in items:
            item.setSelected(True)
        sel_model.blockSignals(False)
        self.set_model_gui(self.spectra_fit.configs[self.index_load])
        self.mark_loaded(items)
        self.update_pars()

    def up(self):
        self.index_load = (self.index_config - 1) % len(self.spectra_fit.configs)
        self.display_index_load()

    def down(self):
        self.index_load = (self.index_config + 1) % len(self.spectra_fit.configs)
        self.display_index_load()

    def _parse_fix(self, i, j):
        try:
            return ast.literal_eval(self.tab_pars.item(i, j).text())
        except (SyntaxError, ValueError):
            return self.tab_pars.item(i, j).text()

    def copy_pars(self):
        if self.spectra_fit.fitter is None:
            return
        for i, model in enumerate(self.spectra_fit.fitter.models):
            tab_dict = {self.tab_pars.item(_i, 0).text(): [self._parse_fix(_i, _j) for _j in range(1, 4)]
                        for _i in range(self.tab_pars.rowCount())}
            pars = [tab_dict.get(name, [val, fix, link]) for name, val, fix, link in model.get_pars()]
            model.set_pars(pars)

    def reset_pars(self):
        self.spectra_fit.reset()
        self.update_pars()

    def save_pars(self):
        self.spectra_fit.save_pars()

    def set_par_multi(self, i, j, suppress_plot=False):
        self.tab_pars.blockSignals(True)
        text = self.tab_pars.item(i, j).text()
        for item in self.tab_pars.selectedItems():
            item.setText(text)
            self._set_par(item.row(), item.column())
        for model in self.spectra_fit.fitter.models:
            model.update()
        self.update_vals(suppress_plot=suppress_plot)
        self.tab_pars.blockSignals(False)

    def _set_par(self, i, j):  # Call only if table signals are blocked.
        set_x = [self.spectra_fit.set_val, self.spectra_fit.set_fix, self.spectra_fit.set_link][j - 1]
        update_x = [self.update_vals, self.update_fixes, self.update_links][j - 1]

        try:
            text = self.tab_pars.item(i, j).text()
            for _inf in inf_str:
                text = text.replace('-{}'.format(_inf), '[]')
                text = text.replace(_inf, '{}')
            val = ast.literal_eval(text)
            if isinstance(val, list):
                val = [-inf if isinstance(v, list) else inf if isinstance(v, dict) else v for v in val] if val else -inf
            if isinstance(val, dict):
                val = inf

        except (ValueError, TypeError, SyntaxError):
            val = self.tab_pars.item(i, j).text()
        for index in range(len(self.spectra_fit.files)):
            try:
                _i = self.spectra_fit.fitter.models[index].names.index(self.tab_pars.item(i, 0).text())
                set_x(index, _i, val)
            except ValueError:
                continue
        update_x(suppress_plot=True)

    """ Model """

    def set_model_gui(self, config):
        self.c_lineshape.setCurrentText(config['lineshape'])
        self.c_convolve.setCurrentText(config['convolve'])
        self.s_npeaks.setValue(config['npeaks'])
        self.check_offset_per_track.setChecked(config['offset_per_track'])
        self.edit_offset_order.setText(str(config['offset_order']))
        self.check_qi.setChecked(config['qi_config']['qi'])
        self.check_hf_mixing.setChecked(config['hf_config']['enabled_l'] or config['hf_config']['enabled_u'])

    def set_lineshape(self):
        for config in self.spectra_fit.configs:
            config['lineshape'] = self.c_lineshape.currentText()

    def set_convolve(self):
        for config in self.spectra_fit.configs:
            config['convolve'] = self.c_convolve.currentText()

    def set_npeaks(self):
        for config in self.spectra_fit.configs:
            config['npeaks'] = self.s_npeaks.value()

    def toggle_offset_per_track(self):
        for config in self.spectra_fit.configs:
            config['offset_per_track'] = self.check_offset_per_track.isChecked()

    def set_offset_order(self):
        try:
            offset_order = list(ast.literal_eval(self.edit_offset_order.text()))
            size = len(offset_order)
            for i, config in enumerate(self.spectra_fit.configs):
                n_tracks = len(self.spectra_fit.fitter.meas[i].x)
                _offset_order = [order for order in offset_order]
                if size == 0:
                    _offset_order = [0, ] * n_tracks
                elif size < n_tracks:
                    if size < len(config['offset_order']):
                        _offset_order = offset_order + config['offset_order'][size:]
                    else:
                        _offset_order = offset_order + [max(offset_order), ] * (n_tracks - size)
                elif size > n_tracks:
                    _offset_order = offset_order[:(n_tracks - size)]
                config['offset_order'] = [order for order in _offset_order]
            self.edit_offset_order.setText(str(self.spectra_fit.configs[self.index_config]['offset_order']))
        except (ValueError, TypeError, SyntaxError, IndexError):
            try:
                self.edit_offset_order.setText(str(self.spectra_fit.configs[self.index_config]['offset_order']))
            except IndexError:
                self.edit_offset_order.setText('[0]')

    def toogle_qi(self):
        for config in self.spectra_fit.configs:
            config['qi_config']['qi'] = self.check_qi.isChecked()

    def toogle_hf_mixing(self):
        # for config in self.spectra_fit.configs:
        #     config['hf_mixing'] = self.check_hf_mixing.isChecked()
        pass

    def open_hf_mixing(self):
        if self.spectra_fit.fitter is None:
            return
        if self.hf_mixing_config_ui is not None:
            self.hf_mixing_config_ui.deleteLater()
        self.hf_mixing_config_ui = HFSConfigUi(self.spectra_fit.fitter.iso[self.index_config],
                                               self.spectra_fit.configs[self.index_config]['hf_config'])
        self.hf_mixing_config_ui.close_signal.connect(self.set_hf_config)
        self.hf_mixing_config_ui.show()

    def set_hf_config(self):
        hf_config = deepcopy(self.hf_mixing_config_ui.config)
        self.check_hf_mixing.setChecked(hf_config['enabled_l'] or hf_config['enabled_u'])
        for config in self.spectra_fit.configs:
            config['hf_config'] = hf_config

    def set_racah(self):
        for splitter_model in self.spectra_fit.splitter_models:
            splitter_model.racah()
        self.update_pars()

    """ Fit """

    def toggle_chi2(self):
        self.spectra_fit.absolute_sigma = not self.check_chi2.isChecked()

    def toggle_delta_f(self):
        self.spectra_fit.unc_from_fit = self.check_delta_f.isChecked()

    def set_x_axis(self, suppress_plot=False):
        self.spectra_fit = self.gen_spectra_fit()
        self.plot_auto(suppress_plot)

    def set_routine(self):
        self.spectra_fit.routine = self.c_routine.currentText()

    def toggle_guess_offset(self):
        self.spectra_fit.guess_offset = self.check_guess_offset.isChecked()

    def toggle_cov_mc(self):
        self.spectra_fit.cov_mc = self.check_cov_mc.isChecked()
        if self.check_cov_mc.isChecked():
            self.s_samples_mc.setEnabled(True)
            self.l_samples_mc.setEnabled(True)
        else:
            self.s_samples_mc.setEnabled(False)
            self.l_samples_mc.setEnabled(False)

    def set_samples_mc(self):
        self.spectra_fit.samples_mc = self.s_samples_mc.value()
    
    def _set_scaler(self, scaler):
        if scaler is None:
            self.spectra_fit.reset_st()
            return
        for i in range(len(self.spectra_fit.fitter.st)):
            self.spectra_fit.fitter.st[i][0] = scaler

    def _set_arithmetics(self, arithmetics):
        self.edit_arithmetics.blockSignals(True)
        self.edit_arithmetics.setText(arithmetics)
        if arithmetics == '':
            arithmetics = self.spectra_fit.arithmetics
            self.edit_arithmetics.setText(arithmetics)
        self.spectra_fit.arithmetics = arithmetics
        self.spectra_fit.fitter.config['arithmetics'] = arithmetics
        self.spectra_fit.fitter.gen_data()
        self.edit_arithmetics.blockSignals(False)

    def set_arithmetics(self, suppress_plot=False):
        if self.spectra_fit.fitter is None:
            self.edit_arithmetics.blockSignals(True)
            self.edit_arithmetics.setText('')
            self.edit_arithmetics.blockSignals(False)
            return
        arithmetics = self.edit_arithmetics.text().strip().lower()
        if arithmetics == self.spectra_fit.fitter.config['arithmetics']:
            return
        if arithmetics == '':
            self._set_scaler(None)
            self._set_arithmetics('')
            self.plot_auto(suppress_plot)
            return
        n = self.spectra_fit.fitter.n_scaler  # Only allow arithmetics with scalers that exist for all files and tracks.
        
        if arithmetics[0] == '[':  # Bracket mode (sum of specified scalers).
            if arithmetics[-1] != ']':
                arithmetics += ']'
            try:
                scaler = sorted(set(eval(arithmetics)))
                while scaler and scaler[-1] >= n:
                    scaler.pop(-1)
                if scaler:
                    self._set_scaler(scaler)
                    self._set_arithmetics(str(scaler))
                    self.plot_auto(suppress_plot)
                else:
                    self._set_arithmetics(self.spectra_fit.arithmetics)
            except (ValueError, SyntaxError):
                self._set_arithmetics(self.spectra_fit.arithmetics)
            return

        variables = {'s{}'.format(i): 4.2 for i in range(n)}
        try:  # Function mode (Specify scaler as variable s#).
            eval(arithmetics, variables)
            self._set_scaler(list(i for i in range(n)))
            self._set_arithmetics(arithmetics)
            self.plot_auto(suppress_plot)
        except (ValueError, TypeError, SyntaxError, NameError):
            self._set_arithmetics(self.spectra_fit.arithmetics)

    def set_arithmetics_toggle_delta_f(self, suppress_plot=False):
        self.set_arithmetics(suppress_plot=suppress_plot)
        if self.spectra_fit.arithmetics[0] == '[':
            self.check_delta_f.setEnabled(True)
        else:
            self.check_delta_f.setChecked(False)
            self.check_delta_f.setEnabled(False)

    def update_arithmetics(self):
        self.edit_arithmetics.blockSignals(True)
        self.edit_arithmetics.setText(self.spectra_fit.arithmetics)
        self.edit_arithmetics.blockSignals(False)

    def toggle_arithmetics(self):
        if self.check_arithmetics.isChecked():
            arithmetics = self.edit_arithmetics.text().strip().lower()
            self._set_scaler(None)
            self._set_arithmetics('')
            self.edit_arithmetics.setReadOnly(True)
            if arithmetics != self.spectra_fit.arithmetics:
                self.plot_auto()
        else:
            self.edit_arithmetics.setReadOnly(False)

    def toggle_norm_scans(self, suppress_plot=False):
        self.spectra_fit.norm_scans = self.check_norm_scans.isChecked()
        self.spectra_fit.set_norm_scans()
        self.plot_auto(suppress_plot)

    def toogle_summed(self):
        pass

    def toogle_linked(self):
        self.spectra_fit.linked = self.check_linked.isChecked()
        if self.check_linked.isChecked():
            if self.check_cov_mc.isChecked():
                self.check_cov_mc.setChecked(False)
            self.check_cov_mc.setEnabled(False)
        else:
            self.check_cov_mc.setEnabled(True)

    def toggle_save_to_db(self):
        self.spectra_fit.save_to_db = self.check_save_to_db.isChecked()

    def toggle_save_to_disk(self):
        self.spectra_fit.save_to_disk = self.check_save_to_disk.isChecked()

    """ Plot """

    def toggle_xlabel(self, suppress_plot=False):
        self.spectra_fit.x_as_freq = self.check_x_as_freq.isChecked()
        self.plot_auto(suppress_plot)

    def plot_auto(self, suppress=False):
        if not suppress and self.check_auto.isChecked():
            self.plot()

    def plot(self):
        self.spectra_fit.plot(clear=True, show=True)

    def save_ascii(self):
        self.spectra_fit.plot(ascii_path=self.spectra_fit.ascii_path)

    def save_plot(self):
        self.spectra_fit.plot(plot_path=self.spectra_fit.plot_path)

    def set_fig_save_format(self):
        self.spectra_fit.fig_save_format = self.c_fig.currentText()

    def set_zoom_data(self, suppress_plot=False):
        self.spectra_fit.zoom_data = self.check_zoom_data.isChecked()
        self.plot_auto(suppress_plot)

    def set_fmt(self, suppress_plot=False):
        try:
            fmt = self.edit_fmt.text()
            _process_plot_format(fmt)
            self.spectra_fit.fmt = fmt
            self.plot_auto(suppress_plot)
        except ValueError:
            self.edit_fmt.setText(self.spectra_fit.fmt)

    def set_fontsize(self, suppress_plot=False):
        fontsize = self.s_fontsize.value()
        if fontsize != self.spectra_fit.fontsize:
            self.spectra_fit.fontsize = fontsize
        self.plot_auto(suppress_plot)

    """ Action (Fit)"""

    def fit(self):
        self.mark_loaded(self.get_selected_items())
        _, _, info = self.spectra_fit.fit()
        self.tab_pars.blockSignals(True)
        self.update_vals()
        self.tab_pars.blockSignals(False)
        self.mark_warn(info['warn'])
        self.mark_errs(info['err'])

    def fit_threaded(self):
        self.mark_loaded(self.get_selected_items())
        if self.spectra_fit.fitter is None:
            return
        self.spectra_fit.fitter.config = self.spectra_fit.gen_config()

        if self.thread is not self.spectra_fit.fitter.thread():
            self.spectra_fit.fitter.moveToThread(self.thread)
            self.spectra_fit.fitter.finished.connect(self.thread.quit)
            self.thread.started.connect(self.spectra_fit.fitter.fit)
            self.thread.finished.connect(self.finish_fit)

        self.enable_gui(False)
        self.thread.start()

    def finish_fit(self):
        self.thread.wait()

        _, _, info = self.spectra_fit.finish_fit()

        self.tab_pars.blockSignals(True)
        self.update_vals(suppress_plot=True)
        self.tab_pars.blockSignals(False)
        self.mark_warn(info['warn'])
        self.mark_errs(info['err'])
        if self.check_save_figure.isChecked():
            for i, path in enumerate(self.spectra_fit.file_paths):
                self.spectra_fit.plot(index=i, clear=True, show=False, plot_path=os.path.split(path)[0])
        self.plot_auto(suppress=False)
        self.enable_gui(True)

    def enable_gui(self, a0):
        self.vert_files.setEnabled(a0)
        self.vert_parameters.setEnabled(a0)
        self.grid_model.setEnabled(a0)
        self.grid_fit.setEnabled(a0)
        self.grid_plot.setEnabled(a0)
        self.b_fit.setEnabled(a0)
        # self.b_abort.setEnabled(not a0)
        #  TODO: abort during fit. This may be difficult to implement cleanly
        #   since curve_fit actually does not allow intervention.
