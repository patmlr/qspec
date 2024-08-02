
import os
import string
import numpy as np
import matplotlib.pyplot as plt


def save_plot_as_ascii(x, y, y_res, yerr, x_fit, y_fit, *y_fit_extra, save_path=r'C:\\', filename='',
                       xlabel='', labels_extra=None):
    if not os.path.exists(save_path):
        try:
            os.makedirs(save_path)
        except Exception as e:
            print('Saving directory has not been created. Writing permission in DB directory?\n'
                  'Error msg: {}'.format(e))
            return
    if labels_extra is None:
        labels_extra = ''
    else:
        labels_extra = ', ' + ', '.join(labels_extra)
    # f = os.path.join(save_path, '{}.data.{}'.format(
    #     filename, datetime.datetime.today().strftime('_%Y-%m-%d_%H-%M-%S.txt')))
    f = os.path.join(save_path, '{}.data.txt'.format(filename))
    np.savetxt(f, np.array([x, y, y_res, yerr]).T, delimiter=', ',
               header='{}, data intensity (counts), fit residuals (counts), data uncertainty (counts)'.format(xlabel))
    # f = os.path.join(save_path, '{}.fit.{}'.format(
    #     filename, datetime.datetime.today().strftime('_%Y-%m-%d_%H-%M-%S.txt')))
    f = os.path.join(save_path, '{}.fit.txt'.format(filename))
    np.savetxt(f, np.array([x_fit, y_fit, *y_fit_extra]).T,
               delimiter=', ', header='{}, fit intensity (counts){}'.format(xlabel, labels_extra))
    # print('Saved plot as ASCII files in {}.'.format(save_path))


def plot_model_fit(fitter, index, x_as_freq=True, plot_summands=True, plot_npeaks=True, plot_fit_uncertainty=True,
                   ascii_path='', plot_path='', fig_save_format='png', zoom_data=False, fmt='.k', fontsize=10):
    fig = plt.figure(num=1, figsize=(8, 8))  # TODO: Fix figure resizing with large legends.
    ax1 = plt.axes([0.15, 0.35, 0.8, 0.50])
    ax2 = plt.axes([0.15, 0.1, 0.8, 0.2], sharex=ax1)

    ax1.tick_params(labelsize=fontsize)
    ax2.tick_params(labelsize=fontsize)
    ax1.get_xaxis().get_major_formatter().set_useOffset(False)
    ax2.get_xaxis().get_major_formatter().set_useOffset(False)
    ax2.locator_params(axis='y', nbins=5)

    model = fitter.models[index]
    vals = [val for val in model.vals]
    fixes = [fix for fix in model.fixes]
    model.set_fixes([False for _ in fixes])  # Reset after side peaks are created.

    x_volt, x, y, yerr = fitter.x_raw[index], fitter.x[index], fitter.y[index], fitter.yerr[index]
    y_res = model(x, *model.vals) - y

    x0 = [np.min(x), np.max(x)]
    x_fit = np.linspace(x0[0], x0[1], int((x0[1] - x0[0]) / model.dx)) if zoom_data else model.x()
    y_fit = model(x_fit, *model.vals)

    # if plot_fit_uncertainty:
    #     np.random.multivariate_normal(model.vals, fitter.pcov,)
    #     model()

    _x = x
    ax1.set_ylabel('intensity (counts)', fontsize=fontsize)
    ax2.set_ylabel('residuals (counts)', fontsize=fontsize)
    if fitter.meas[index].seq_type == 'kepco':
        _x = x_volt
        ax2.set_xlabel('DAC voltage (V)', fontsize=fontsize, labelpad=fontsize / 2)
        ax1.set_ylabel('voltage (V)', fontsize=fontsize)
        ax2.set_ylabel('residuals (V)', fontsize=fontsize)
    else:
        if fitter.config['x_axis'] in {'ion frequencies', 'lab frequencies'}:
            ax2.set_xlabel('relative frequency (MHz)', fontsize=fontsize, labelpad=fontsize / 2)
        elif fitter.config['x_axis'] == 'DAC voltages':
            ax2.set_xlabel('DAC voltage (V)', fontsize=fontsize, labelpad=fontsize / 2)
        else:
            ax2.set_xlabel('voltage (V)', fontsize=fontsize, labelpad=fontsize / 2)

    plot_data = ax1.errorbar(_x, y, yerr=yerr, fmt=fmt, label=fitter.meas[index].file)
    plot_fit = ax1.plot(x_fit, y_fit, '-C0', label='Full fit')

    # Create side peaks.
    y_fit_extra = []
    labels_extra = []
    plot_n = []
    n_pars = [name for name in model.names if name[0] == 'p' and name[1] in string.digits]
    if plot_npeaks and len(n_pars) > 1:  # Plot all side peaks.
        for i, par in enumerate(n_pars):
            for _par in n_pars:
                if _par != par:
                    vals[model.names.index(_par)] = 0.
            y_fit_extra.append(model(x_fit, *vals))
            labels_extra.append('fit {} intensity (counts)'.format(par))
            plot_n.append(ax1.plot(x_fit, y_fit_extra[-1], '--C0', label='Fit {}'.format(par))[0])
            for _par in n_pars:
                _i = model.names.index(_par)
                vals[_i] = model.vals[_i]

    plot_sum = []
    if plot_summands:  # Plot all isotopes.
        sum_pars = [name.replace('center', 'int') for name in model.names if 'center(' in name]
        for i, par in enumerate(sum_pars):
            for _par in sum_pars:
                if _par != par:
                    vals[model.names.index(_par)] = 0.
            label = par[(par.find('(') + 1):par.find(')')]
            y_fit_extra.append(model(x_fit, *vals))
            labels_extra.append('fit {} intensity (counts)'.format(label))
            plot_sum.append(ax1.plot(x_fit, y_fit_extra[-1], '-C{}'.format(i % 9 + 1), label=label)[0])
            if plot_npeaks and len(n_pars) > 1:  # Plot all side peaks for every isotope.
                for j, par_n in enumerate(n_pars):
                    for _par in n_pars:
                        if _par != par_n:
                            vals[model.names.index(_par)] = 0.
                    y_fit_extra.append(model(x_fit, *vals))
                    labels_extra.append('fit {}.{} intensity (counts)'.format(label, par_n))
                    plot_n.append(ax1.plot(x_fit, y_fit_extra[-1], '--C{}'.format(i % 9 + 1),
                                           label='{}.{}'.format(label, par_n))[0])
                    for _par in n_pars:
                        _i = model.names.index(_par)
                        vals[_i] = model.vals[_i]
            for _par in sum_pars:
                _i = model.names.index(_par)
                vals[_i] = model.vals[_i]
    model.set_fixes(fixes)
    # End of side peak creation.

    ax2.errorbar(_x, y_res, yerr=yerr, fmt=fmt, label='Residuals')

    lines = [plot_data, plot_fit[0], *plot_sum, *plot_n]
    labels = [each.get_label() for each in lines]
    fig.legend(lines, labels, loc='upper center', ncol=2, bbox_to_anchor=(0.15, 0.8, 0.8, 0.2), mode='expand',
               fontsize=fontsize + 2, numpoints=1)

    if zoom_data:
        x_min, x_max = np.min(x), np.max(x)
        dx = 0.05 * (x_max - x_min)
        y_min, y_max = np.min(y), np.max(y)
        dy = 0.05 * (y_max - y_min)
        ax1.set_xlim(x_min - dx, x_max + dx)
        ax1.set_ylim(y_min - dy, y_max + dy)

    run = '.{}'.format(fitter.run) if fitter.run else ''
    filename = '{}{}'.format(os.path.splitext(fitter.meas[index].file)[0], run)
    if ascii_path:
        save_plot_as_ascii(_x, y, y_res, yerr, x_fit, y_fit, *y_fit_extra, save_path=ascii_path,
                           filename=filename, xlabel=ax2.get_xlabel(), labels_extra=labels_extra)
    if plot_path:
        if not os.path.exists(plot_path):
            try:
                os.makedirs(plot_path)
            except Exception as e:
                print('Saving directory has not been created. Writing permission in DB directory?\n'
                      'Error msg: {}'.format(e))
                return
        if fig_save_format[0] == '.':
            fig_save_format = fig_save_format[1:]
        f = os.path.join(plot_path, '{}.{}'.format(filename, fig_save_format))
        fig.savefig(f)
    return fig
