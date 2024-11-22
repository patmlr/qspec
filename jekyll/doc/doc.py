
import importlib
import os
import inspect


FILES = ['tools']  # , 'physics'


def load_table_template():
    with open('_template-table.html', 'r') as f:
        ret = f.readlines()
    return ret


def load_table():
    with open('table.html', 'r') as f:
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

        namespace = f'qspec{file if file in {"models", "simulate"} else ""}'
        mod = importlib.import_module(f'qspec.{file}')
        func_str = sorted(f for f in mod.__all__ if callable(eval(f'mod.{f}', {'mod': mod})))
        j = temp.index('<!--p>tab-func</p-->') + 1
        html += '\n'.join(temp[i:j-1]).replace('_file_', file)
        for f in func_str:
            html += '\n' + temp[j].replace('_file_', file).replace('_func_', f)
        i = temp.index('<!--p>tab-func-end</p-->') + 1
        html += '\n'.join(temp[i:-1])
        i = temp.index('<!--p>tab-module</p-->') + 1
    html += '\n' + temp[-1]

    with open('table.html', 'w') as html_file:
        html_file.write(html)


def gen_functions():
    for file in FILES:
        directory = os.path.join('functions', file)
        if not os.path.exists(directory):
            os.makedirs(directory)

        namespace = f'qspec{file if file in {"models", "simulate"} else ""}'
        mod = importlib.import_module(f'qspec.{file}')
        func_str = [f for f in mod.__all__ if callable(eval(f'mod.{f}', {'mod': mod}))]
        funcs = [eval(f'mod.{f}') for f in func_str]
        func_sig = {k: inspect.signature(f) for k, f in zip(func_str, funcs)}
        func_doc = {k: f.__doc__ for k, f in zip(func_str, funcs)}

        temp = [t.strip() for t in load_functions_template()]
        for f in func_str:
            i = temp.index('<!--p>sig</p-->')
            html = '\n'.join(temp[:i])
            html += '\n'.join(temp[i+1:i+5])
            html = html.replace('_title_', f'{namespace}.{f}').replace('_func_', f)

            # Signature
            i = temp.index('<!--p>sig-pars</p-->') + 1
            for p, p_sig in func_sig[f].parameters.items():
                p_str = p_sig.__str__()
                html += '\n' + temp[i]
                if '**' in p_str:
                    html += '\n' + temp[i + 4]
                elif '*' in p_str:
                    html += '\n' + temp[i + 2]
                html += '\n' + temp[i + 6].replace('_par_', p)
                if '=' in p_str:
                    html += '\n' + temp[i + 8] + '\n' + temp[i + 10].replace('_default_', repr(p_sig.default))
                html += f'\n{temp[i + 11]}\n,'
            html = html[:-1] + '\n'.join(temp[i+12:i+14])

            # Description
            i = temp.index('<!--p>desc</p-->') + 1
            desc = func_doc[f]
            if desc is None:
                desc = ''
            else:
                j = desc.find(':param')
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
                p_desc = func_doc[f]
                j = p_desc.find(f':param {p}:') + len(f':param {p}:')
                k = p_desc[j:].find(':param')
                if k != -1:
                    k += j
                else:
                    k = p_desc[j:].find(':return')
                    if k != -1:
                        k += j
                anno = p_sig.annotation
                html += ('\n' + '\n'.join(temp[i+2:i+4])
                         .replace('_par_', p)
                         .replace('_par-type_', str('' if anno is inspect._empty else anno))
                         .replace('_par-description_', p_desc[j:k].strip().strip('\n')))
            html += '\n'.join(temp[i+4:i+6])

            # Returns
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
                     .replace('_ret-type_', str('' if anno is inspect._empty else anno))
                     .replace('_ret-description_', r_desc))
            html += '\n'.join(temp[i+4:i+7])
            html += '\n'.join(temp[i+8:])

            # Content table
            html_table = '\n'.join(load_table())
            i = html_table.find(f)
            i = html_table[:i].rfind('<li>')
            html_table = html_table[:i] + '<li class="table-current">' + html_table[i + 4:]
            html = html.replace('_table_', html_table)

            with open(os.path.join(directory, f'{f}.html'), 'w') as html_file:
                html_file.write(html)



if __name__ == '__main__':
    gen_table()
    gen_functions()
