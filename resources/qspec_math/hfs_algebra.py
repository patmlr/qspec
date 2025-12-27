from typing import Any

import sympy as sy
import sympy.physics.wigner as wi

I = sy.symbols("I")
J = sy.symbols("J")

MI = sy.symbols("m_I")
MJ = sy.symbols("m_J")

I_set = sy.symbols("I_-, I_z, I_+", commutative=False)
I_set = {-1: I_set[0] / sy.sqrt(2), 0: I_set[1], 1: -I_set[2] / sy.sqrt(2)}

J_set = sy.symbols("J_-, J_z, J_+", commutative=False)
J_set = {-1: J_set[0] / sy.sqrt(2), 0: J_set[1], 1: -J_set[2] / sy.sqrt(2)}


def t_rank_k(k: int, spin_set: dict[int, Any]) -> dict[int, Any]:
    if k == 1:
        return spin_set
    if k > 1:
        t0 = t_rank_k(k - 1, spin_set)
        return {
            q: sy.Add(
                *[
                    wi.clebsch_gordan(k - 1, 1, k, m0, m1, q) * t0[m0] * spin_set[m1]
                    for m0 in range(-k + 1, k)
                    for m1 in range(-1, 2)
                    if m0 + m1 == q
                ]
            )
            for q in range(-k, k + 1)
        }

    raise ValueError(f"There is no rank k={k}.")


def hfs_rank_k(k: int) -> Any:
    ti = t_rank_k(k, I_set)
    # for q in range(-k, k + 1):
    #     print(f'{q}: {ti[q]}')

    tj = t_rank_k(k, J_set)
    h = sy.Add(*[sy.S(-1) ** sy.S(q) * ti[q] * tj[-q] for q in range(-k, k + 1)])
    return sy.simplify(h).expand()


def norm_rank_k(k: int) -> Any:
    ti = t_rank_k(k, I_set)
    tj = t_rank_k(k, J_set)
    return sy.simplify(ti[0] * tj[0]).expand()


def act_jm(j: float, m: float) -> tuple[Any, float]:
    coef = sy.sqrt((j + m) * (j - m + 1))
    return coef, m - 1


def act_jz(j: float, m: float) -> tuple[float, float]:
    return m, m


def act_jp(j: float, m: float) -> tuple[Any, float]:
    coef = sy.sqrt((j - m) * (j + m + 1))
    return coef, m + 1


def act_ops_im_jm(ops: list[str], i: float, j: float, mi: float, mj: float) -> tuple[Any, float, float]:
    coef = sy.S(1)
    _mi = mi
    _mj = mj

    for op in ops:
        if "I_-" in op:
            c, _mi = act_jm(i, _mi)
        elif "I_z" in op:
            c, _mi = act_jz(i, _mi)
        elif "I_+" in op:
            c, _mi = act_jp(i, _mi)
        elif "J_-" in op:
            c, _mj = act_jm(j, _mj)
        elif "J_z" in op:
            c, _mj = act_jz(j, _mj)
        elif "J_+" in op:
            c, _mj = act_jp(j, _mj)
        else:
            c = sy.S(op)
        coef *= c

    return sy.simplify(coef), _mi, _mj


def act_sum_ops_im_jm(ops: list[list[str]], i: float, j: float, mi: float, mj: float) -> Any:
    return sy.Add(*[act_ops_im_jm(op, i, mi, j, mj) for op in ops])


def matrix_jm(op: str, i: float, j: float, mi0: float, mi1: float, mj0: float, mj1: float) -> Any:
    # print(f'Input: {op}')
    ops = word_to_list(op)
    vec = [act_ops_im_jm(_op, i, j, mi1, mj1) for _op in ops]
    return sy.simplify(sy.Add(*[c for (c, _mi1, _mj1) in vec if mi0 == _mi1 and mj0 == _mj1]))


def word_to_list(op: str) -> list[list[str]]:
    if not isinstance(op, str):
        op = str(op)
    op = op.strip().replace(" - ", " + -1*")
    w_list = op.split(" + ")
    ops = []

    for w in w_list:
        _ops = []
        _ops0 = w.strip().replace("**", "^").replace("/", "*1/").split("*")
        for i, _op in enumerate(_ops0):
            index = _op.find("^")
            if index == -1:
                _ops.append(_op)
                continue
            n = ""
            for s in _op[index + 1 :]:
                if s.isnumeric():
                    n += s
                    continue
                break
            n = int(n)
            s = _op[index - 3 : index]
            _ops0[i] = _op.replace(f"{s}^{n}", "*".join([s] * n))
            _ops += _ops0[i].split("*")
        ops.append(_ops)

    return ops


def convert_to_latex(s: str) -> str:
    return f"${s.replace('*', ' ')}$"


def convert_to_python(s: Any) -> str:
    return (
        str(s)
        .replace("m_I", "mi1")
        .replace("m_J", "mj1")
        .replace("I", "i")
        .replace("J", "j")
        .replace("*", " * ")
        .replace(" *  * ", " ** ")
        .replace("/", " / ")
        .replace("sqrt", "np.sqrt")
    )


if __name__ == "__main__":
    norm = [sy.S(1), sy.S(6), sy.S(10)]
    hfs = [(norm[_k - 1] * hfs_rank_k(_k)).expand() for _k in range(1, 4)]
    for _k, _h in enumerate(hfs):
        print(f"\n{_k + 1}: {_h}")
        print(f"/: {matrix_jm(norm_rank_k(_k + 1), I, J, I, I, J, J)}")

        print(f"mi0 == mi1     and mj0 == mj1    : {convert_to_python(matrix_jm(_h, I, J, MI, MI, MJ, MJ))}")

        for __k in range(1, _k + 2):
            print(
                f"mi0 == mi1 + {__k} and mj0 == mj1 - {__k}:"
                f" {convert_to_python(matrix_jm(_h, I, J, MI + __k, MI, MJ - __k, MJ))}"
            )
            print(
                f"mi0 == mi1 - {__k} and mj0 == mj1 + {__k}:"
                f" {convert_to_python(matrix_jm(_h, I, J, MI - __k, MI, MJ + __k, MJ))}"
            )
