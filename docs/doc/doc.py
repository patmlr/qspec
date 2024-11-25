
import importlib
import os
import inspect
from docutils.core import publish_parts


FOLDER_FILES = {'models', 'simulate', 'analyze'}
FILES = sorted(['simulate'])  # , 'analyze', 'algebra', 'physics', 'models', 'tools', 'stats'


def type_to_str(_type):
    return (str('None' if _type is inspect._empty else _type).replace('typing.', '').replace('<class ', '').replace('>', '')
            .replace("'", ""))


def rest_to_html(rest):
    html = publish_parts(rest, writer_name='html')['html_body']
    return html


def load_table_template():
    with open('_template-table.html', 'r') as f:
        ret = f.readlines()
    return ret


def load_table():
    with open('table.html', 'r') as f:
        ret = f.readlines()
    return ret


def load_module(file):
    with open(os.path.join('modules', f'_{file}.html'), 'r') as f:
        ret = f.readlines()
    return ret


def load_functions_template():
    with open(os.path.join('functions', '_template.html'), 'r') as f:
        ret = f.readlines()
    return ret


def gen_table():
    temp = [t.strip() for t in load_table_template()]
    i = temp.index('<!--p>tab-module</p-->') + 1
    html = '\n'.join(temp[:i-1])
    for file in FILES:
        directory = os.path.join('functions', file)
        if not os.path.exists(directory):
            os.makedirs(directory)

        namespace = f'qspec{f".{file}" if file in FOLDER_FILES else ""}'
        mod = importlib.import_module(f'qspec.{file}')
        func_str = sorted(f for f in mod.__all__ if callable(eval(f'mod.{f}', {'mod': mod})))
        funcs = {f: eval(f'mod.{f}') for f in func_str}

        j = temp.index('<!--p>tab-func</p-->') + 1
        html += '\n'.join(temp[i:j-1]).replace('_title_', f'qspec.{file}').replace('_file_', file)
        for f in func_str:
            if f[0].isupper():
                i = temp.index('<!--p>tab-class</p-->') + 1
                j = temp.index('<!--p>tab-class-func</p-->') + 1
                html += ('\n'.join(temp[i:j-1]).replace('_file_', file).replace('_class_', f)
                         .replace('_namespace-class_', f'{namespace}.{f}'))

                directory = os.path.join('functions', file, f)
                if not os.path.exists(directory):
                    os.makedirs(directory)

                class_funcs = [m[0] for m in inspect.getmembers(funcs[f], inspect.isfunction)
                               if not m[0].startswith('_')]

                for cf in class_funcs:
                    html += ('\n' + temp[j].replace('_file_', file).replace('_class_', f).replace('_class-func_', cf)
                             .replace('_namespace-class-func_', f'{namespace}.{f}.{cf}'))
                i = temp.index('<!--p>tab-func-end</p-->')
                html += '\n'.join(temp[j+1:i])
                j = temp.index('<!--p>tab-func</p-->') + 1
            else:
                html += ('\n' + temp[j].replace('_file_', file).replace('_func_', f)
                         .replace('_namespace-func_', f'{namespace}.{f}'))


        i = temp.index('<!--p>tab-func-end</p-->') + 1
        html += '\n'.join(temp[i:-1])
        i = temp.index('<!--p>tab-module</p-->') + 1
    html += '\n' + temp[-1]

    with open('table.html', 'w') as html_file:
        html_file.write(html)


def gen_modules():
    html_table = ''.join(load_table())
    for file in FILES:
        html = '\n'.join([t.strip() for t in load_module(file)])
        html = html.replace('_table_', html_table)
        i = html.find(f'{file}.html')

        i = html[:i].rfind('<li')
        j = i + html[i + 1:].find('>') + 2
        html = (html[:i] + f'<li class="has-children spaced table-current">' + html[j:])

        j += html[j:].find('<details')
        html = html[:j + 8] + ' open=""' + html[j + 8:]

        html = '\n'.join(['---', 'layout: default', f'title: qspec.{file}', '---', '']) + html

        with open(os.path.join('modules', f'{file}.html'), 'w') as html_file:
            html_file.write(html)


def _gen_func(f, file, temp, namespace, funcs, func_sig, func_doc):
    j = namespace.rfind('.')
    class_flag = 1 if f[0].isupper() else 2 if namespace[j+1:][0].isupper() else 0

    i = temp.index('<!--p>sig</p-->')
    html = '\n'.join(temp[:i])
    html += '\n'.join(temp[i+1:i+3])

    if class_flag == 1:
        namespace_part = namespace
        _class = f
    elif class_flag == 2:
        namespace_part = namespace[:j]
        _class = namespace[j+1:]
        html += '\n' + temp[i+3]
    else:
        namespace_part = namespace
        _class = ''

    html += '\n'.join(temp[i+4:i+6])
    html = (html.replace('_title_', f'{namespace}.{f}')
            .replace('_namespace_', namespace_part)
            .replace('_class_', _class).replace('_func_', f)
            .replace('sig-name', 'sig-classname' if class_flag == 1 else 'sig-name'))

    # Signature
    i = temp.index('<!--p>sig-pars</p-->') + 1
    for p, p_sig in func_sig[f].parameters.items():
        if p == 'self':
            continue
        p_str = p_sig.__str__()
        html += '\n' + temp[i]
        if '**' in p_str:
            html += '\n' + temp[i + 4]
        elif '*' in p_str:
            html += '\n' + temp[i + 2]
        html += '\n' + temp[i + 6].replace('_par_', p)
        if '=' in p_str:
            default = p_sig.default
            if callable(default):
                default = default.__name__
            html += '\n' + temp[i + 8] + '\n' + temp[i + 10].replace('_default_', str(default))
        html += f'\n{temp[i + 11]}\n,&nbsp;'
    lines, i_start = inspect.getsourcelines(funcs[f])
    i_stop = i_start + len(lines) - 1
    if file in FOLDER_FILES:
        file_py = os.path.join(file, os.path.splitext(os.path.basename(inspect.getfile(funcs[f])))[0])
    else:
        file_py = file
    del_comma_flag = bool(len([p for p in func_sig[f].parameters.keys() if p != 'self']))
    html = (html[:(-7 if del_comma_flag else None)] + '\n'
            .join(temp[i+12:i+14]).replace('_file_', file_py)
            .replace('_start_', str(i_start)).replace('_stop_', str(i_stop)))

    # Description
    i = temp.index('<!--p>desc</p-->') + 1
    desc = func_doc[f]
    if desc is None:
        desc = ''
    else:
        j = desc.find(':param')
        if j == -1:
            j = desc.find(':return')
            if j == -1:
                desc = ''
            else:
                desc = desc[:j].strip().strip('\n')
        else:
            desc = desc[:j].strip().strip('\n')
    html += '\n' + temp[i].replace('_description_', desc)

    # Parameters
    html += '\n' + temp[i + 1]
    i = temp.index('<!--p>pars-h</p-->') + 1
    if func_sig[f].parameters:
        html += '\n' + temp[i]
    i = temp.index('<!--p>pars</p-->') + 1
    html += '\n' + '\n'.join(temp[i:i+2])
    for p, p_sig in func_sig[f].parameters.items():
        if p == 'self':
            continue
        p_desc = func_doc[f]
        if p_desc is None:
            p_desc = ''
        else:
            j = p_desc.find(f':param {p}:') + len(f':param {p}:')
            if j == -1:
                p_desc = ''
            else:
                k = p_desc[j:].find(':param')
                if k != -1:
                    k += j
                else:
                    k = p_desc[j:].find(':return')
                    if k != -1:
                        k += j
            p_desc = p_desc[j:k].strip().strip('\n')
        anno = p_sig.annotation
        html += ('\n' + '\n'.join(temp[i+2:i+4])
                 .replace('_par_', p)
                 .replace('_par-type_', type_to_str(anno))
                 .replace('_par-description_', p_desc))
    html += '\n'.join(temp[i+4:i+6])

    # Returns
    if class_flag == 1:
        i = temp.index('<!--p>rets</p-->') + 1
    else:
        i = temp.index('<!--p>rets-h</p-->') + 1
        html += temp[i]
        i = temp.index('<!--p>rets</p-->') + 1
        html += '\n' + '\n'.join(temp[i:i+2])
        r_desc = func_doc[f]
        if r_desc is None:
            r_desc = ''
        else:
            k = r_desc.find(':return')
            if k != -1:
                k += r_desc[k+1:].find(':') + 2
                r_desc = r_desc[k:].strip().strip('\n')
            else:
                r_desc = ''
        anno = func_sig[f].return_annotation
        html += ('\n' + '\n'.join(temp[i+2:i+4])
                 .replace('_ret_', 'out')
                 .replace('_ret-type_', type_to_str(anno))
                 .replace('_ret-description_', r_desc))
        html += '\n'.join(temp[i+4:i+6])

    html += '\n' + temp[i+6]
    html += '\n'.join(temp[i+8:])

    # Content table
    html_table = ''.join(load_table())
    i = html_table.find(f'{namespace}.{f}')
    i = html_table[:i].rfind('<li')
    j = i + html_table[i+1:].find('>') + 2
    html_table = (html_table[:i] + f'<li class="{'has-children ' if class_flag == 1 else ''}'
                                   f'{'spaced ' if class_flag != 2 else ''}table-current">' + html_table[j:])
    if class_flag == 2:
        j = html_table[:i].rfind('<details')
        html_table = html_table[:j+8] + ' open=""' + html_table[j+8:]
    if class_flag == 1:
        j = i + html_table[i:].find('<details')
        html_table = html_table[:j+8] + ' open=""' + html_table[j+8:]

    j = html_table[:j].rfind(f'{file}.html')
    j += html_table[j:].find('<details')
    html_table = html_table[:j + 8] + ' open=""' + html_table[j + 8:]


    html = html.replace('_table_', html_table)
    return html


def _gen_class_functions(directory, file, f, class_funcs, namespace):
    temp = [t.strip() for t in load_functions_template()]

    func_str = [cf[0] for cf in class_funcs]
    funcs = {cf[0]: cf[1] for cf in class_funcs}
    func_sig = {f: inspect.signature(funcs[f]) for f in func_str}
    func_doc = {f: funcs[f].__doc__ for f in func_str}

    for cf in func_str:
        html = _gen_func(cf, file, temp, f'{namespace}.{f}', funcs, func_sig, func_doc)
        with open(os.path.join(directory, f'{os.path.join(f, cf)}.html'), 'w') as html_file:
            html_file.write(html)

def gen_functions():
    for file in FILES:
        directory = os.path.join('functions', file)
        if not os.path.exists(directory):
            os.makedirs(directory)

        namespace = f'qspec{f".{file}" if file in FOLDER_FILES else ""}'
        mod = importlib.import_module(f'qspec.{file}')
        func_str = sorted(f for f in mod.__all__ if callable(eval(f'mod.{f}', {'mod': mod})))
        funcs = {f: eval(f'mod.{f}') for f in func_str}
        func_sig = {f: inspect.signature(funcs[f]) for f in func_str}
        func_doc = {f: funcs[f].__doc__ for f in func_str}

        temp = [t.strip() for t in load_functions_template()]
        for f in func_str:
            html = _gen_func(f, file, temp, namespace, funcs, func_sig, func_doc)
            if f[0].isupper():
                class_funcs = [m for m in inspect.getmembers(funcs[f], inspect.isfunction)
                               if not m[0].startswith('_')]
                _gen_class_functions(directory, file, f, class_funcs, namespace)

                f_path = os.path.join(f, f)
            else:
                f_path = f
            with open(os.path.join(directory, f'{f_path}.html'), 'w') as html_file:
                html_file.write(html)



if __name__ == '__main__':
    # gen_table()
    # gen_modules()
    gen_functions()
