
import os
import inspect
from pygments import highlight
from pygments.lexers import get_lexer_by_name
from pygments.formatters import HtmlFormatter
from pygments.styles import get_style_by_name


STYLE = 'tango'


def example_0():
    import numpy as np
    import qspec as qs
    # This imports the analyze, algebra,
    # physics, stats, and tools modules

    q = 1  # (e), Ion charge state
    m = 87.905612253 - q * qs.me_u  # (u) [1]
    # Mass of 87Sr+

    U = 20000  # (V), Acceleration voltage

    # Resonance frequency from NIST database
    f0 = qs.inv_cm_to_freq(24516.65)  # (MHz) [2]
    # >>> 734990676.5 MHz

    # Relativistic velocity of 88Sr+
    v = qs.v_el(U, q, m)  # (m/s)
    # >>> 209533.6 m/s

    # The anti-collinear lab. frequency
    f_laser = qs.doppler(f0, v, np.pi)  # (MHz)
    # >>> 735504562.3 MHz


def example_1():
    import numpy as np
    import qspec as qs

    I = 4.5  # Total nuclear spin quantum number of 87Sr+
    J = 2.5  # Total angular momentum quantum number of the 2D5/2 state
    g_i = -1.09316 / I  # The nuclear g-factor [1]
    g_j = qs.lande_j(s=0.5, l=2, j=2.5)
    # The nuclear g-factor, calculated from the nuclear magnetic moment

    A, B = 2.1743, 49.11  # (MHz), The hyperfine-structure constants [2]
    b_field = np.linspace(0., 4e-3, 4000)  # (T), The magnetic flux density

    e_eig, m_list, fm_list, mi_mj_list \
        = qs.hyper_zeeman_num(I, J, A, B, g_i, g_j, b_field)
    # The eigenvalues of the HFS + Zeeman-effect Hamiltonian
    # and the lists of m_F, F and (m_I, m_J) quantum numbers


def example_2():
    import numpy as np
    import qspec.models as mod
    import matplotlib.pyplot as plt

    # Generate the model used in this tutorial
    model = mod.Offset(  # Add a y-axis shift (off0e0)

        mod.NPeak(  # Add a single (n_peaks=1)
            # x-axis shift (x0) and an intensity (p0)

            mod.Voigt(), n_peaks=1)) # Add a Voigt lineshape
            # with Lorentzian (Gamma) and Gaussian (sigma) widths

    print(f'\nParameter names: {model.names}\n')
    # >>> Parameter names: ['Gamma', 'sigma', 'x0', 'p0', 'off0e0']

    # Change all parameter values and if they stay fixed during fitting at once
    model.set_vals([20., 6., 0., 10., 3.])
    model.set_fixes([False, '5(0.7)', False, False, False])

    # Generate some random data for this tutorial
    x = np.linspace(-80., 80., 81)  # x-values
    sigma_y = np.full_like(x, 0.5)  # y-uncertainties
    y = np.random.normal(model(x, *model.vals), sigma_y)
    # Random y-values around the model

    # Change the initial value of the peak position and intensity
    model.set_val(2, 15.)  # Parameter 2 (x0), the peak position
    model.set_val(3, 5.)  # Parameter 3 (p0), the peak intensity
    p_init = model.vals.copy()  # Copy the initial values for the plot

    # Fit the constructed model to the data
    popt, pcov, info = mod.fit(
        model, x, y, sigma_y=sigma_y, report=True)

    # We plot the data, the initial model and the fitted model
    plt.errorbar(x, y, yerr=sigma_y, fmt='.k', label='Data')
    plt.plot(x, model(x, *p_init), '-C0', label='Initial model')
    plt.plot(x, model(x, *popt), '-C1', label='Fitted model')

    # Improve the plot
    plt.legend()
    plt.xlabel('x'), plt.ylabel('y')
    plt.subplots_adjust(left=0.08, bottom=0.09, top=0.99, right=0.99)
    plt.show()


def example_3():
    import numpy as np
    import qspec.simulate as sim

    f_sp = 446810183.163  # Transition frequency (MHz)
    a_sp = 36.891  # Einstein coefficient (rad MHz)

    s_hyper = [401.75825]  # HFS constants (MHz)
    p_hyper = [-3.055038, -0.29670]

    s = sim.construct_electronic_state(
        0., s=0.5, l=0, j=0.5, i=1.5, hyper_const=s_hyper, label='s')
    p = sim.construct_electronic_state(
        f_sp, s=0.5, l=1, j=1.5, i=1.5, hyper_const=p_hyper, label='p')

    decay = sim.DecayMap(labels=[('s', 'p')], a=[a_sp])
    li7 = sim.Atom(s + p, decay)

    intensity = 1.  # uW /mm**2
    polarization = sim.Polarization([0, 1, 0])  # Linear polarization
    laser = sim.Laser(f_sp, intensity, polarization)

    inter = sim.Interaction(li7, [laser])
    inter.controlled = True  # Error controlled integrator

    t = 0.2  # Integration time (us)
    delta = np.linspace(-325, -275, 201)  # Frequency detunings (MHz)
    theta, phi = 0., 0.  # Angles from z-axis in x- and y-direction (rad)

    n = inter.rates(t, delta)  # Rate equations, 0.2 us
    y_rates = li7.scattering_rate(n, theta, phi, as_density_matrix=False)

    rho = inter.master(t, delta)  # Master equation, 0.2 us
    y_master = li7.scattering_rate(rho, theta, phi)

    rho = inter.master(0.4, delta)  # Master equation, 0.4 us
    y4_master = li7.scattering_rate(rho, theta, phi)

    sr = sim.ScatteringRate(li7, laser=laser)
    y_brown = sr.generate_y(delta, theta, phi)[:, 0, 0]  # Brown et al.

    import matplotlib.pyplot as plt

    scale = 1e3
    x_lim = delta[0], delta[-1]

    fig, (m, r) = plt.subplots(
        2, 1, sharex='all', height_ratios=[3, 1], figsize=(6, 5))

    m.plot(delta, y_brown * scale, '-k',
           label=r'Brown $et\,al.$', zorder=20)
    m.plot(delta, y_rates[:, -1] * scale, '-C0',
           label=r'rates, $t = 0.2\,\mu$s', zorder=0)
    m.plot(delta, y_master[:, -1] * scale, '-C1',
           label=r'master, $t = 0.2\,\mu$s', zorder=60)
    m.plot(delta, y4_master[:, -1] * scale, '--C3',
           label=r'master, $t = 0.4\,\mu$s', linewidth=1.5, zorder=30)

    r.plot(delta, (y_brown - y_rates[:, -1]) * scale,
           '-k', zorder=20)
    r.plot(delta, (y_master[:, -1] - y_rates[:, -1]) * scale,
           '-C1', zorder=60)
    r.plot(delta, (y4_master[:, -1] - y_rates[:, -1]) * scale,
           '--C3', linewidth=1.5, zorder=10)
    r.hlines(0, *x_lim, 'C0', '-', zorder=0)

    m.legend()
    m.set_ylabel(r'$\mathrm{d}\Gamma / \mathrm{d}\Omega$ (kHz)')
    m.set_xlim(*x_lim)
    r.set_xlabel('Relative frequency (MHz)')
    r.set_ylabel('Residuals (kHz)')

    y_lim = m.get_ylim()
    r.set_ylim(-(y_lim[1] - y_lim[0]) / 6, (y_lim[1] - y_lim[0]) / 6)

    plt.subplots_adjust(left=0.09, bottom=0.1, right=0.99, top=0.99, hspace=0.05)
    plt.show()


def pycode_to_html(code):
    style = get_style_by_name(STYLE)
    lexer = get_lexer_by_name('python', stripall=True)
    formatter = HtmlFormatter(linenos=False, cssclass='py-source', style=style)
    html = highlight(code, lexer, formatter)
    print(html)


def gen_pycode_css():
    style = get_style_by_name(STYLE)
    formatter = HtmlFormatter(cssclass='py-source', style=style)
    css = formatter.get_style_defs()
    with open(os.path.join(os.pardir, '_sass', 'pycode.scss'), 'w') as css_file:
        css_file.write(css)


def gen_example(n):
    # with open(os.path.join(os.pardir, os.pardir, 'examples', 'overview_tut.py'), 'r') as py_file:
    #     code = py_file.read()
    code = inspect.getsource(eval(f'example_{n}'))
    pycode_to_html(code)


if __name__ == '__main__':
    # gen_pycode_css()
    # gen_example(0)
    # gen_example(3)
    example_3()
