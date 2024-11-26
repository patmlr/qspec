
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
    q = 1  # (e), Ion charge state
    m = 86.908877495 - q * qs.me_u  # (u), Mass of 87Sr+ [1]
    U = 20e3 # (kV), Acceleration voltage

    f0 = qs.inv_cm_to_freq(24516.65)  # 734990677 MHz [2]
    # Expected resonance frequency from NIST database

    v = qs.v_el(U, q, m)  # (m/s) Relativistic velocity of 87Sr+
    f_laser = qs.doppler(f0, v, np.pi)  # 735507502 MHz
    # (MHz) The required anti-collinear lab. frequency


def example_1():
    import numpy as np
    import qspec as qs

    I = 4.5  # Total nuclear spin quantum number of 87Sr+
    J = 2.5  # Total angular momentum quantum number of the 2D5/2 state
    g_n = -1.09316 / I  # The nuclear g-factor [3]
    g_j = qs.lande_j(s=0.5, l=2, j=2.5)
    # The nuclear g-factor, calculated from the nuclear magnetic moment

    A, B = 2.1743, 49.11  # (MHz), The hyperfine-structure constants [4]
    b_field = np.linspace(0., 4e-3, 1000)  # (T), The magnetic flux density

    e_eig, f_list, m_list = qs.hyper_zeeman_num(I, J, g_n, g_j, A, B, b_field)
    # The eigenvalues of the HFS + Zeeman-effect Hamiltonian
    # and the lists of F and m quantum numbers.


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
    gen_example(0)
    gen_example(1)
    # example_1()
