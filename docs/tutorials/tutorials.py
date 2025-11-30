
import os
import inspect
from pygments import highlight
from pygments.lexers import get_lexer_by_name
from pygments.formatters import HtmlFormatter
from pygments.styles import get_style_by_name


STYLE = 'tango'


def example_0():
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
    print(f'f0: {f0} MHz')

    # Relativistic velocity of 88Sr+
    v = qs.v_el(U, q, m)  # (m/s)
    # >>> 209533.6 m/s
    print(f'v: {v} MHz')

    # The anti-collinear lab. frequency
    f_laser = qs.doppler(f0, v, qs.pi, return_frame='lab')  # (MHz)
    # >>> 734477149.8 MHz
    print(f'f_laser: {f_laser} MHz')

    # The differential Doppler shift
    df_atom = qs.doppler_el_d1(f_laser, qs.pi, U, q, m)  # (MHz / V)
    # >>> +12.84 MHz / V
    print(f'df_atom: {df_atom} MHz')


def example_1():
    import numpy as np
    import qspec as qs

    I = 4.5  # Total nuclear spin quantum number of 87Sr+
    J = 2.5  # Total angular momentum quantum number of the 2D5/2 state
    g_i = -1.09316 / I  # The nuclear g-factor [1]
    g_j = qs.lande_j(s=0.5, l=2, j=2.5)
    # The nuclear g-factor, calculated from the nuclear magnetic moment

    A, B, C = 2.1743, 49.11, 0.  # (MHz), The hyperfine-structure constants [2]
    b_field = np.linspace(0., 4e-3, 4000)  # (T), The magnetic flux density

    e_eig, m_list, fm_list, mi_mj_list \
        = qs.hyper_zeeman_num(I, J, [A, B, C], g_i, g_j, b_field)
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
    import qspec as qs

    # The mass numbers of the Ca isotopes.
    a = [40, 42, 43, 44, 46, 48, 50, 52]

    # The masses of the isotopes (u, AME 2020).
    m = [(39.962590850, 22e-9), (41.958617780, 159e-9),  # 40Ca, 42Ca
         (42.958766381, 244e-9), (43.955481489, 348e-9),  # 43Ca, 44Ca
         (45.953687726, 2398e-9), (47.952522654, 18e-9),  # 46Ca, 48Ca
         (49.957499215, 1.7e-6), (51.963213646, 720e-9)]  # 50Ca, 52Ca

    # Use absolute values given in the shape (#isotopes, #observables, 2).
    # Frequencies for the (D1, D2) lines (MHz).
    x_abs = [[(755222765.66, 0.10), (761905012.53, 0.11)],  # 40Ca
             [(755223191.15, 0.10), (761905438.57, 0.10)],  # 42Ca
             [(755223443.57, 0.30), (761905691.89, 0.17)],  # 43Ca
             [(755223614.66, 0.10), (761905862.62, 0.09)],  # 44Ca
             [(755224063.27, 0.33), (761906311.60, 0.57)],  # 46Ca
             [(755224471.12, 0.10), (761906720.11, 0.11)],  # 48Ca
             [(        0.  , 0.  ), (        0.  , 0.  )],  # 50Ca
             [(        0.  , 0.  ), (        0.  , 0.  )]]  # 52Ca

    # Construct a King object. Optionally specify 'x_abs' here
    # to omit isotope shifts when fitting. 20 electron masses are subtracted
    # to perform the King plot analysis with the nuclear masses.
    king = qs.King(a=a, m=m, x_abs=x_abs, subtract_electrons=20)

    a_fit = [42, 43, 44, 46, 48]  # Choose the isotopes to fit.
    a_ref = [40, 48, 42, 40, 44]  # Choose individual reference isotopes.

    # Do a simple 2d King plot.
    # The 'mode' keyword is only used for the axis labels.
    popt, pcov = king.fit(a_fit, a_ref, mode='shifts')
    # >>> f(x) = (177.3 u MHz) + 1.00068 * x

    a_unknown = [50, 52]  # Specify the unknown isotopes
    a_unknown_ref = [40, 40]  # and their references.

    # Specify the isotope shifts of the D2 line.
    y = [(1969.2, 5.6), (2219.2, 7.0)]

    # Calculate the isotope shifts of the D1 line and their covariances.
    x, cov, cov_stat = king.get_unmodified(
        a_unknown, a_unknown_ref, y, axis=1, show=True, mode='shifts')

    for iso, c in zip(a_unknown, cov):
        qs.printh(f'\n{iso}Ca+:')  # Print colored headline.
        qs.print_cov(c)  # Print color-coded covariance matrix.


def example_4():
    import numpy as np
    import qspec.simulate as sim

    f_sp = 446810183.163  # Transition frequency (MHz)
    a_sp = 36.891  # Einstein coefficient (rad MHz)

    s_hyper = [401.75825]  # HFS constants (MHz)
    p_hyper = [-3.055038, -0.29670]

    s = sim.gen_electronic_ls_state(
        0., s=0.5, l=0, j=0.5, i=1.5, hyper_const=s_hyper, label='s')
    p = sim.gen_electronic_ls_state(
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
    y_rates = li7.scattering_rate(n, as_density_matrix=False,
                                  theta=theta, phi=phi)
    y_rates = y_rates[0, :, -1]  # Remove angle and time axes

    rho = inter.master(t, delta)  # Master equation, 0.2 us
    y_master = li7.scattering_rate(rho, theta=theta, phi=phi)
    y_master = y_master[0, :, -1]

    rho = inter.master(0.4, delta)  # Master equation, 0.4 us
    y4_master = li7.scattering_rate(rho, theta=theta, phi=phi)
    y4_master = y4_master[0, :, -1]

    sr = sim.ScatteringRate(li7, laser=laser)
    y_brown = sr.generate_y(delta, theta, phi)  # Brown et al.
    y_brown = y_brown[:, 0, 0]  # Remove (theta, phi) axes

    import matplotlib.pyplot as plt

    scale = 1e3
    x_lim = delta[0], delta[-1]

    fig, (m, r) = plt.subplots(
        2, 1, sharex='all', height_ratios=[3, 1], figsize=(6, 5))

    m.plot(delta, y_brown * scale, '-k',
           label=r'Brown $et\,al.$', zorder=20)
    m.plot(delta, y_rates * scale, '-C0',
           label=r'rates, $t = 0.2\,\mu$s', zorder=0)
    m.plot(delta, y_master * scale, '-C1',
           label=r'master, $t = 0.2\,\mu$s', zorder=60)
    m.plot(delta, y4_master * scale, '--C3',
           label=r'master, $t = 0.4\,\mu$s', linewidth=1.5, zorder=30)

    r.plot(delta, (y_brown - y_rates) * scale,
           '-k', zorder=20)
    r.plot(delta, (y_master - y_rates) * scale,
           '-C1', zorder=60)
    r.plot(delta, (y4_master - y_rates) * scale,
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


def example_5():
    import numpy as np
    import qspec as qs
    import qspec.simulate as sim
    import matplotlib.pyplot as plt

    f_eg = 7e8  # Transition frequency
    a_eg = 10.  # Einstein coefficient

    g_hyper = [0.]  # HFS A-constant of the g state
    e_hyper = [5.]  # HFS A-constant of the e state

    i, jg, je = 0., 2., 5.
    dm = 1  # The change of the m quantum number

    g = sim.State(0., parity='e', j=jg, i=i, f=jg + i, m=jg + i,
                  hyper_const=g_hyper, label='g')
    e = sim.State(f_eg, parity='o', j=je, i=i, f=je + i, m=jg + i + dm,
                  hyper_const=e_hyper, label='e')

    states = [g, e]
    decay = sim.DecayMap(labels=[('g', 'e')], a=[a_eg], k_max=int(je - jg))
    atom = sim.Atom(states=states, decay_map=decay)
    print(atom.get_multipole_types('g', 'e'))
    # >>> {'e3'}

    intensity = 1e3
    pol_eg = sim.Polarization([1., 0, 1j], vec_as_q=False)
    laser_eg = sim.Laser(freq=f_eg, polarization=pol_eg,
                         intensity=intensity, k=[0., 1., 0.])

    B = [0., 0., 1e-6]  # The B-field vector
    env = sim.Environment(B=B)
    inter = sim.Interaction(atom=atom, lasers=[laser_eg, ],
                            environment=env, delta_max=1000.)

    t = np.linspace(0., 1., 301)  # Create an array of times to simulate
    y0 = qs.unit_vector(0, 2, dtype=float)  # Create initial population [1., 0.]

    rho = inter.master(t, y0=y0)
    y = sim.density_matrix_diagonal(rho, axis=1)[0]

    plt.figure(figsize=(6, 4))
    plt.plot(t, y[0], label=g.label)
    plt.plot(t, y[1], label=e.label)
    plt.legend()
    plt.xlabel(r'Time ($\mu$s)')
    plt.ylabel('Population')
    plt.show()

    n_theta, n_phi = 128, 256
    theta = np.linspace(0., np.pi, n_theta)
    phi = np.linspace(0., 2 * np.pi, n_phi)
    theta, phi = np.meshgrid(theta, phi, indexing='ij')

    r = atom.scattering_rate(rho[0, :, :, -1], theta=theta, phi=phi,
                             x_vec=None, axis=0)
    r /= np.max(r)
    r = r.reshape((n_theta, n_phi))

    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})

    cm = plt.get_cmap('plasma')
    ax.plot_surface(x, y, z, facecolors=cm(r), rcount=128, ccount=256,
                    linewidth=0, antialiased=False)

    xyz_lim = 0.7
    ax.set_xlim(-xyz_lim, xyz_lim)
    ax.set_ylim(-xyz_lim, xyz_lim)
    ax.set_zlim(-xyz_lim, xyz_lim)
    ax.set_box_aspect((1., 1., 1.))
    ax.set_axis_off()
    plt.subplots_adjust(left=0., bottom=0., right=1., top=1.)
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
    gen_example(5)
    example_5()
