
import os
from pygments import highlight
from pygments.lexers import get_lexer_by_name
from pygments.formatters import HtmlFormatter
from pygments.styles import get_style_by_name


STYLE = 'tango'


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


def gen_tutorials():

    with open(os.path.join(os.pardir, os.pardir, 'examples', 'overview_tut.py'), 'r') as py_file:
        code = py_file.read()
    pycode_to_html(code)

if __name__ == '__main__':
    gen_pycode_css()
    gen_tutorials()
