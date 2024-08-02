
import ctypes
import ctypes.wintypes
import sqlite3


def get_documents_dir():
    """
    :returns: The absolute path to the documents folder.
    """
    csidl_personal = 5  # My Documents
    shgfp_type_current = 0  # Get current, not default value

    buf = ctypes.create_unicode_buffer(ctypes.wintypes.MAX_PATH)
    ctypes.windll.shell32.SHGetFolderPathW(None, csidl_personal, None, shgfp_type_current, buf)
    return buf.value


def select_from_db(db, var_from, vars_select, var_where=None, add_cond=''):
    """
    Connects to database 'db' and finds attributes 'vars_select' (string) in table 'var_from' (string).
    The command will convert be 'SELECT vars_select FROM var_from WHERE varwhere[0][i] == varwhere[1][i] ... addCond'.

    :param db: The database (str).
    :param var_from: The table to select the variables from (str), e.g. 'Isotopes'.
    :param vars_select: The variables to select (str), e.g. vars_select='mass, mass_d'.
    :param var_where: Conditions to filter the selected entries ([list, list] -> var_where[0][i]
    == var_where[1][i] for all i), e.g. [['I'], [1.5]] to select isotopes with nuclear spin I==1.5.
    :param add_cond: An additional condition (str), e.g. 'ORDER BY date'.
    :returns: List of tuples with values [(vars_select0, vars_select1...), (...)].
    """
    sql_cmd = ''
    con = None
    if var_where is None:
        var_where = []
    var = []

    try:
        con = sqlite3.connect(db)
        cur = con.cursor()
        if var_where:
            where = var_where[0][0] + ' = ?'
            list_where_is = [var_where[1][0]]
            var_where[0].remove(var_where[0][0])
            var_where[1].remove(var_where[1][0])
            for i, j in enumerate(var_where[0]):
                where = where + ' and ' + j + ' = ?'
                list_where_is.append(var_where[1][i])
            sql_cmd = f'SELECT {vars_select} FROM {var_from} WHERE {where} {add_cond}'
            cur.execute(sql_cmd, tuple(list_where_is))
        else:
            sql_cmd = f'SELECT {vars_select} FROM {var_from} {add_cond}'
            cur.execute(sql_cmd, ())
        var = cur.fetchall()
        return var
    except Exception as e:
        print(f'Error in database access while trying to execute:\n{sql_cmd}\n{e}')

    if con is not None:
        con.close()

    return var


def write_to_db(db, var_to, var_select, values):
    con = sqlite3.connect(db)
    cur = con.cursor()

    var_select_str = ', '.join(var_select)
    q = ', '.join(['?'] * len(var_select))
    cur.execute(f'INSERT OR REPLACE INTO {var_to} ({var_select_str}) VALUES ({q})', values)
    con.commit()
    con.close()
