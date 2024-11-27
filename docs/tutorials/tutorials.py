
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

    U = 20000 # (V), Acceleration voltage

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
    g_n = -1.09316 / I  # The nuclear g-factor [1]
    g_j = qs.lande_j(s=0.5, l=2, j=2.5)
    # The nuclear g-factor, calculated from the nuclear magnetic moment

    A, B = 2.1743, 49.11  # (MHz), The hyperfine-structure constants [2]
    b_field = np.linspace(0., 4e-3, 4000)  # (T), The magnetic flux density

    e_eig, m_list, fm_list, mi_mj_list \
        = qs.hyper_zeeman_num(I, J, g_n, g_j, A, B, b_field)
    # The eigenvalues of the HFS + Zeeman-effect Hamiltonian
    # and the lists of m_F, F and (m_I, m_J) quantum numbers.


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
    gen_example(1)
    # example_0()
