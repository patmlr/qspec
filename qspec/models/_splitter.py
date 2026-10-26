"""
qspec.models._splitter
======================

Splitter classes for lineshape models.
"""

import os
from string import ascii_uppercase

import numpy as np

from qspec.algebra import a, b, c, clebsch_gordan, wigner_6j
from qspec.models._base import Model, Summed
from qspec.models._spectrum import LorentzQI
from qspec.physics import get_f, get_m, hyper_zeeman_num, lande_f, zeeman_linear
from qspec.qtypes import (
    Any,
    Iterable,
    Sequence,
    array_like,
    cast,
    fix_type,
    floating,
    has_getitem,
    is_scalar,
    ndarray,
    quant,
    scalar,
)
from qspec.tools import merge_intervals, unit_vector

__all__ = [
    "Hyperfine",
    "HyperfineMixed",
    "HyperfineQI",
    "HyperfineZeeman",
    "Splitter",
    "SplitterSummed",
    "gen_splitter_model",
    "get_all_f",
    "hf_coeff",
    "hf_int",
    "hf_shift",
    "hf_trans",
]


def gen_splitter_model(qi: bool = False, hf_mixing: bool = False) -> type["Splitter"]:
    """
    Choose the `Splitter` type: [`HyperfineQI`], [`HyperfineMixed`] or [`Hyperfine`].

    :param qi: Whether to consider quantum interference (QI) effects.
    :param hf_mixing: Whether to consider hyperfine-induced mixing.
    :returns: (splitter) The appropriate `Splitter` type.
    :raises ValueError: If both `qi` and `hf_mixing` are `True`.
    """
    if qi and hf_mixing:
        pass
    elif qi:
        return HyperfineQI
    elif hf_mixing:
        return HyperfineMixed
    else:
        return Hyperfine
    raise ValueError("Specified splitter model not available.")


def get_all_f(i: scalar, j: scalar) -> list[quant]:
    """
    All total angular momentum quantum numbers $F$ of one or several fine-structure states.

    :param i: The nuclear spin(s) $I$.
    :param j: The electronic total angular momentum quantum number(s) $J$.
    :returns: (F_list) The sorted list of all possible $F$ quantum numbers of all combinations of `i` and `j`.
    """
    i, j = np.array(i, float).flatten(), np.array(j, float).flatten()
    return sorted(set(quant(f + abs(_i - _j)) for _i in i for _j in j for f in range(int(_i + _j - abs(_i - _j) + 1))))


def hf_coeff(i: scalar, j: scalar, f: scalar) -> tuple[float, ...]:
    """
    The coefficients of the hyperfine structure constants $A$, $B$ and $C$ of the magnetic dipole,
    electric quadrupole and magnetic octupole order, such that the hyperfine structure shift of the state
    is $\\Delta_\\mathrm{hfs} = c_AA + c_BB + c_CC$, see [`qspec.physics.hyperfine`].
    Only the orders allowed by $I$ and $J$ are returned.

    :param i: The nuclear spin quantum number $I$.
    :param j: The electronic total angular momentum quantum number $J$.
    :param f: The total angular momentum quantum number $F$.
    :returns: (coeffs) The tuple of hyperfine coefficients $(c_A, c_B, c_C)$ of the quantum state.
    """
    i, j, f = cast(i, j, f, dtype=float)

    if i < 0.5 or j < 0.5:
        return tuple()

    k = f * (f + 1) - i * (i + 1) - j * (j + 1)
    co_a = 0.5 * k
    if i < 1 or j < 1:
        return (co_a,)

    co_b = (0.75 * k * (k + 1) - j * (j + 1) * i * (i + 1)) / (2 * i * (2 * i - 1) * j * (2 * j - 1))
    if i < 1.5 or j < 1.5:
        return co_a, co_b

    co_c = (
        k**3
        + 4 * k**2
        + 0.8 * k * (-3 * i * (i + 1) * j * (j + 1) + i * (i + 1) + j * (j + 1) + 3)
        - 4 * i * (i + 1) * j * (j + 1)
    )
    co_c /= i * (i - 1) * (2 * i - 1) * j * (j - 1) * (2 * j - 1)
    co_c *= 1.25
    if i < 2 or j < 2:
        return co_a, co_b, co_c

    return co_a, co_b, co_c


def hf_trans(i: scalar, j_l: scalar, j_u: scalar) -> list[list[tuple]]:
    r"""
    All allowed electric dipole transitions $F_l\rightarrow F_u$ between two hyperfine-structure multiplets
    and their hyperfine coefficients, see [`hf_coeff`].

    :param i: The nuclear spin quantum number $I$.
    :param j_l: The electronic total angular momentum quantum number $J_l$ of the lower state.
    :param j_u: The electronic total angular momentum quantum number $J_u$ of the upper state.
    :returns: (transitions) A list of transitions `[(f_l, f_u), coeffs_l, coeffs_u]`.
    """
    return [
        [(f_l, f_u), hf_coeff(i, j_l, f_l), hf_coeff(i, j_u, f_u)]
        for f_l in get_f(i, j_l)
        for f_u in get_f(i, j_u)
        if abs(f_u - f_l) < 1.1 and not f_u == f_l == 0
    ]


def hf_shift(
    hyper_l: Iterable[scalar],
    hyper_u: Iterable[scalar],
    coeff_l: Iterable[scalar],
    coeff_u: Iterable[scalar],
) -> ndarray[floating]:
    r"""
    The hyperfine structure shift of an optical transition

    $$
    \Delta_\mathrm{hfs} = \sum\limits_k K_k^u c_k^u - \sum\limits_k K_k^l c_k^l,
    $$

    with the hyperfine structure constants $K_k \in \{A, B, C\}$ and the coefficients $c_k$, see [`hf_coeff`].

    :param hyper_l: The hyperfine structure constants $K_k^l$ of the lower state `(Al, Bl, Cl, ...)`.
    :param hyper_u: The hyperfine structure constants $K_k^u$ of the upper state `(Au, Bu, Cu, ...)`.
    :param coeff_l: The coefficients $c_k^l$ of the lower state.
    :param coeff_u: The coefficients $c_k^u$ of the upper state.
    :returns: (dnu_hfs) The hyperfine structure shift $\Delta_\mathrm{hfs}$ (unit of `hyper_l`).
    """
    return np.sum([np.asarray(const * coeff, dtype=float) for const, coeff in zip(hyper_u, coeff_u)]) - np.sum(
        [np.asarray(const * coeff, dtype=float) for const, coeff in zip(hyper_l, coeff_l)]
    )


def hfm_lin_shift(
    hyper_l: Iterable[scalar],
    hyper_u: Iterable[scalar],
    coeff_l: Iterable[scalar],
    coeff_u: Iterable[scalar],
    m_l: scalar,
    m_u: scalar,
    g_l: array_like,
    g_u: array_like,
    b_field: array_like,
) -> ndarray:
    r"""
    The hyperfine structure and linear Zeeman shift of an optical transition,
    see [`hf_shift`] and [`qspec.physics.zeeman_linear`].

    :param hyper_l: The hyperfine structure constants of the lower state `(Al, Bl, Cl, ...)` (MHz).
    :param hyper_u: The hyperfine structure constants of the upper state `(Au, Bu, Cu, ...)` (MHz).
    :param coeff_l: The coefficients of the lower state, see [`hf_coeff`].
    :param coeff_u: The coefficients of the upper state, see [`hf_coeff`].
    :param m_l: The lower state magnetic quantum number.
    :param m_u: The upper state magnetic quantum number.
    :param g_l: The Landé g-factor $g_F$ of the lower state.
    :param g_u: The Landé g-factor $g_F$ of the upper state.
    :param b_field: The magnetic flux density $\mathcal{B}$ (T).
    :returns: (dnu) The hyperfine structure and Zeeman shift of the optical transition (MHz).
    """
    return (
        sum(const * coeff for const, coeff in zip(hyper_u, coeff_u))
        - sum(const * coeff for const, coeff in zip(hyper_l, coeff_l))
        + zeeman_linear(m_u, g_u, b_field, as_freq=True)
        - zeeman_linear(m_l, g_l, b_field, as_freq=True)
    )


def hf_int(i: scalar, j_l: scalar, j_u: scalar, transitions: Iterable) -> list[float]:
    r"""
    The relative line intensities (Racah intensities) of hyperfine transitions $F_l\rightarrow F_u$

    $$
    I_{F_lF_u} = (2F_l + 1)(2F_u + 1)\begin{Bmatrix}J_l & F_l & I\\ F_u & J_u & 1\end{Bmatrix}^2.
    $$

    :param i: The nuclear spin quantum number $I$.
    :param j_l: The electronic total angular momentum quantum number $J_l$ of the lower state.
    :param j_u: The electronic total angular momentum quantum number $J_u$ of the upper state.
    :param transitions: A list of transitions, see [`hf_trans`].
     The first element of each entry must be the pair `(f_l, f_u)`.
    :returns: (intensities) The relative line intensities $I_{F_lF_u}$.
    """
    return [
        float(
            np.around(
                (2 * f_u + 1) * (2 * f_l + 1) * (wigner_6j(j_l, f_l, i, f_u, j_u, 1, as_sympy=False) ** 2), decimals=9
            )
        )
        for (f_l, f_u), *_ in transitions
    ]


class Splitter(Model):
    def __init__(
        self, model: Model, i: scalar, j_l: scalar, j_u: scalar, label: str | None = None, **kwargs: Any
    ) -> None:
        r"""
        The abstract base class for hyperfine structure models.

        :param model: A submodel whose parameters are adopted by this model.
        :param i: The nuclear spin quantum number $I$.
        :param j_l: The electronic total angular momentum quantum number $J_l$ of the lower state.
        :param j_u: The electronic total angular momentum quantum number $J_u$ of the upper state.
        :param label: A label for the `Splitter` (isotope / isomer / transition).
        :param kwargs: Additional keyword arguments of the specific `Splitter` model.
        """
        super().__init__(model=model)
        self.type = "Splitter"

        self.i = float(i)
        self.j_l = float(j_l)
        self.j_u = float(j_u)
        if label is None:
            label = "None"
        self.label = label

        self.racah_indices: list[int] = []
        self.racah_intensities: list[float] = []

    def racah(self) -> None:
        """
        Set the intensity parameters to the Racah intensities, see [`hf_int`].
        """
        for i, intensity in zip(self.racah_indices, self.racah_intensities):
            self.vals[i] = intensity

    def racah_fixed(self, fix: fix_type) -> None:
        """
        Set the fix state of all intensity parameters.

        :param fix: The new fix state of the intensity parameters.
        """
        for i in self.racah_indices:
            self.fixes[i] = fix


class SplitterSummed(Summed):
    def __init__(self, splitter_models: list[Splitter]) -> None:
        """
        A [`Summed`] model of `Splitter` submodels, e.g., of several isotopes.

        :param splitter_models: The `Splitter` submodels to sum over.
        :raises TypeError: If any of the submodels is not a `Splitter`.
        """
        if any(not isinstance(model, Splitter) for model in splitter_models):
            raise TypeError("All models passed to 'SplitterSummed' must have type 'Splitter'.")

        self.models: Sequence[Splitter]
        super().__init__(
            splitter_models,
            labels=[f"({i if model.label is None else model.label})" for i, model in enumerate(splitter_models)]
            if len(splitter_models) > 1
            else None,
        )

    def racah(self) -> None:
        """
        Set the intensity parameters of all submodels to the Racah intensities, see [`hf_int`].
        """
        i0 = 0
        for model in self.models:
            for i, intensity in zip(model.racah_indices, model.racah_intensities):
                self.set_val(i0 + i, intensity, force=True)
            i0 += model.size
        self.set_vals(self.vals, force=True)


class Hyperfine(Splitter):
    def __init__(self, model: Model, i: scalar, j_l: scalar, j_u: scalar, label: str | None = None) -> None:
        r"""
        A hyperfine structure model, summing the submodel $g$ over all allowed transitions $F_l\rightarrow F_u$

        $$
        f(x) = \sum\limits_{F_l, F_u} p_{F_lF_u}\,g(x - \Delta_\mathrm{hfs}^{F_lF_u}),
        $$

        with the hyperfine structure shifts $\Delta_\mathrm{hfs}^{F_lF_u}$, see [`hf_shift`].
        The parameters are the hyperfine structure constants `Al`, `Bl`, ... and `Au`, `Bu`, ...
        (up to the orders allowed by $I$ and $J$), the ratios `A_ratio`, ... defined as `Au / Al`, ...,
        and the intensities $p_{F_lF_u}$ named `int(F_l, F_u)`, initialized with the Racah intensities [`hf_int`].
        The first intensity is fixed by default.

        :param model: A submodel whose parameters are adopted by this model.
        :param i: The nuclear spin quantum number $I$.
        :param j_l: The electronic total angular momentum quantum number $J_l$ of the lower state.
        :param j_u: The electronic total angular momentum quantum number $J_u$ of the upper state.
        :param label: A label for the `Splitter` (isotope / isomer / transition).
        """
        self.model: Model
        super().__init__(model, i, j_l, j_u, label=label)
        self.type = "Hyperfine"

        self.transitions = hf_trans(self.i, self.j_l, self.j_u)
        self.racah_intensities = hf_int(self.i, self.j_l, self.j_u, self.transitions)

        self.n_l = len(self.transitions[0][1])
        self.n_u = len(self.transitions[0][2])
        for i in range(self.n_l):
            self._add_arg(f"{ascii_uppercase[i]}l", 0.0, False, False)
        for i in range(self.n_u):
            self._add_arg(f"{ascii_uppercase[i]}u", 0.0, False, False)

        for i in range(min([self.n_l, self.n_u])):
            fix = "{} / {}".format(f"{ascii_uppercase[i]}u", f"{ascii_uppercase[i]}l")
            self._add_arg(f"{ascii_uppercase[i]}_ratio", 0.0, fix, False)

        for i, (t, intensity) in enumerate(zip(self.transitions, self.racah_intensities)):
            self.racah_indices.append(self._index)
            self._add_arg(f"int({t[0][0]}, {t[0][1]})", intensity, i == 0, False)

    def evaluate(self, x: ndarray, *args: scalar, **kwargs: Any) -> ndarray:
        const_l = tuple(args[self.model.size + i] for i in range(self.n_l))
        const_u = tuple(args[self.model.size + self.n_l + i] for i in range(self.n_u))
        return np.sum(
            [
                args[i] * self.model.evaluate(x - hf_shift(const_l, const_u, t[1], t[2]), *args, **kwargs)
                for i, t in zip(self.racah_indices, self.transitions)
            ],
            axis=0,
        )

    def min(self) -> float:
        const_l = tuple(self.vals[self.model.size + i] for i in range(self.n_l))
        const_u = tuple(self.vals[self.model.size + self.n_l + i] for i in range(self.n_u))
        return self.model.min() + min(float(hf_shift(const_l, const_u, t[1], t[2])) for t in self.transitions)

    def max(self) -> float:
        const_l = tuple(self.vals[self.model.size + i] for i in range(self.n_l))
        const_u = tuple(self.vals[self.model.size + self.n_l + i] for i in range(self.n_u))
        return self.model.max() + max(float(hf_shift(const_l, const_u, t[1], t[2])) for t in self.transitions)

    def intervals(self) -> list[list[float]]:
        const_l = tuple(self.vals[self.model.size + i] for i in range(self.n_l))
        const_u = tuple(self.vals[self.model.size + self.n_l + i] for i in range(self.n_u))
        shifts = [float(hf_shift(const_l, const_u, t[1], t[2])) for t in self.transitions]
        return merge_intervals([[self.model.min() + shift, self.model.max() + shift] for shift in shifts]).tolist()

    def racah(self) -> None:
        for i, intensity in zip(self.racah_indices, self.racah_intensities):
            self.vals[i] = intensity


def load_qi(filepath: str) -> tuple[list[float], list[float], list[list[float]]]:
    """
    :param filepath: The path to the saved model coefficients.
    :returns: The QI model coefficients A, B and C.
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"QI configuration file at {filepath} not found.")

    with open(os.path.join(filepath)) as file:
        ret: dict[str, Any] = eval(file.read())

    return ret["a"], ret["b"], ret["c"]


class HyperfineQI(Splitter):
    def __init__(
        self, _: Model, i: scalar, j_l: scalar, j_u: scalar, label: str | None = None, qi_path: str | None = None
    ) -> None:
        r"""
        A perturbative quantum interference (QI) hyperfine-structure model

        $$
        f(x) = \frac{1}{\max A_F^{F^\prime}}\sum\limits_{F, F^\prime}\left[\left(A_F^{F^\prime}
        + B_F^{F^\prime}G\right)L(x - \Delta_{FF^\prime}) + G\sum\limits_{F^{\prime\prime} > F^\prime}
        C_F^{F^\prime F^{\prime\prime}}L_\mathrm{qi}(x - \Delta_{FF^\prime}, \Delta_{FF^{\prime\prime}}
        - \Delta_{FF^\prime})\right],
        $$

        with the coefficients $A_F^{F^\prime}$, $B_F^{F^\prime}$ and $C_F^{F^\prime F^{\prime\prime}}$
        [`qspec.algebra.abc`], the Lorentzian $L$ and its dispersive term $L_\mathrm{qi}$ [`LorentzQI`],
        the hyperfine structure shifts $\Delta_{FF^\prime}$ [`hf_shift`] and the geometry parameter $G$ (`geo`),
        based on [<a href="https://doi.org/10.1103/PhysRevA.87.032504">
        R. C. Brown <i>et al.</i>, Phys. Rev. A <b>87</b>, 032504 (2013)</a>].
        The other parameters are those of [`Hyperfine`], without the intensities.

        :param _: Empty `model` parameter. This hyperfine-structure model always uses `LorentzQI` lineshapes.
         To use other lineshapes, combine it with a [`Convolved`] model.
        :param i: The nuclear spin quantum number $I$.
        :param j_l: The electronic total angular momentum quantum number $J_l$ of the lower state.
        :param j_u: The electronic total angular momentum quantum number $J_u$ of the upper state.
        :param label: A label for the `Splitter` (isotope / isomer / transition).
        :param qi_path: Specify a directory to save and load the calculated geometric parts of the
         dipole matrix elements for faster repeated calculations.
         These depend only on the quantum numbers `i`, `j_l` and `j_u`.
        """
        self.model: LorentzQI
        super().__init__(LorentzQI(), i, j_l, j_u, label)
        self.type = "HyperfineQI"

        self.qi_path = qi_path
        self.file = f"qi_{label}.txt"

        self.transitions = hf_trans(self.i, self.j_l, self.j_u)
        self.racah_intensities = [
            0.0,
        ]

        save = False
        if self.qi_path is None or not os.path.isfile(os.path.join(self.qi_path, self.file)):
            print(f"Calculating QI A-matrix of isotope {self.label} ... ")
            self.a_qi: list[float] = [
                float(a(self.i, self.j_l, f_l, self.j_u, f_u, as_sympy=False))
                for f_l in get_f(self.i, self.j_l)
                for f_u in get_f(self.i, self.j_u)
                if abs(f_u - f_l) < 1.1 and not f_u == f_l == 0
            ]
            print(f"Calculating QI B-matrix of isotope {self.label} ... ")
            self.b_qi: list[float] = [
                float(b(self.i, self.j_l, f_l, self.j_u, f_u, as_sympy=False))
                for f_l in get_f(self.i, self.j_l)
                for f_u in get_f(self.i, self.j_u)
                if abs(f_u - f_l) < 1.1 and not f_u == f_l == 0
            ]
            print(f"Calculating QI C-matrix of isotope {self.label} ... ")
            self.c_qi: list[list[float]] = [
                [
                    float(c(self.i, self.j_l, f_l, self.j_u, f1_u, f2_u, as_sympy=False))
                    for i2, f2_u in enumerate(get_f(self.i, self.j_u))
                    if i2 > i1
                    if abs(f1_u - f_l) < 1.1 and abs(f2_u - f_l) < 1.1 and not f1_u == f_l == 0 and not f2_u == f_l == 0
                ]
                for f_l in get_f(self.i, self.j_l)
                for i1, f1_u in enumerate(get_f(self.i, self.j_u))
            ]
            if self.qi_path is not None:
                save = True
        else:
            self.a_qi, self.b_qi, self.c_qi = load_qi(os.path.join(self.qi_path, self.file))

        if save and self.qi_path is not None:
            qi_dict = {"a": self.a_qi, "b": self.b_qi, "c": self.c_qi}
            with open(os.path.join(self.qi_path, self.file), "w") as file:
                file.write(str(qi_dict))

        self.transitions_qi = [
            [
                [(f_l, f2_u), hf_coeff(self.i, self.j_l, f_l), hf_coeff(self.i, self.j_u, f2_u)]
                for i2, f2_u in enumerate(get_f(self.i, self.j_u))
                if i2 > i1
                if abs(f1_u - f_l) < 1.1 and abs(f2_u - f_l) < 1.1 and not f1_u == f_l == 0 and not f2_u == f_l == 0
            ]
            for f_l in get_f(self.i, self.j_l)
            for i1, f1_u in enumerate(get_f(self.i, self.j_u))
        ]

        self.n_l = len(self.transitions[0][1])
        self.n_u = len(self.transitions[0][2])
        for i in range(self.n_l):
            self._add_arg(f"{ascii_uppercase[i]}l", 0.0, False, False)
        for i in range(self.n_u):
            self._add_arg(f"{ascii_uppercase[i]}u", 0.0, False, False)

        for i in range(min([self.n_l, self.n_u])):
            fix = "{} / {}".format(f"{ascii_uppercase[i]}u", f"{ascii_uppercase[i]}l")
            self._add_arg(f"{ascii_uppercase[i]}_ratio", 0.0, fix, False)

        self.racah_indices.append(self._index)
        self._add_arg("geo", 0.0, False, False)

    def evaluate(self, x: ndarray, *args: scalar, **kwargs: Any) -> ndarray:
        const_l = tuple(args[self.model.size + i] for i in range(self.n_l))
        const_u = tuple(args[self.model.size + self.n_l + i] for i in range(self.n_u))
        return np.sum(
            [
                (_a + _b * args[self.racah_indices[0]])
                * self.model.evaluate(x - hf_shift(const_l, const_u, t[1], t[2]), *args, **kwargs)
                + np.sum(
                    [
                        _c
                        * args[self.racah_indices[0]]
                        * self.model.evaluate_qi(
                            x - hf_shift(const_l, const_u, t[1], t[2]),
                            hf_shift(const_l, const_u, _t[1], _t[2]) - hf_shift(const_l, const_u, t[1], t[2]),
                            *args,
                            **kwargs,
                        )
                        for _t, _c in zip(t_list, c_list)
                    ],
                    axis=0,
                )
                for t, _a, _b, t_list, c_list in zip(
                    self.transitions, self.a_qi, self.b_qi, self.transitions_qi, self.c_qi
                )
            ],
            axis=0,
        ) / np.max(self.a_qi)

    def min(self) -> float:
        const_l = tuple(self.vals[self.model.size + i] for i in range(self.n_l))
        const_u = tuple(self.vals[self.model.size + self.n_l + i] for i in range(self.n_u))
        return self.model.min() + min(float(hf_shift(const_l, const_u, t[1], t[2])) for t in self.transitions)

    def max(self) -> float:
        const_l = tuple(self.vals[self.model.size + i] for i in range(self.n_l))
        const_u = tuple(self.vals[self.model.size + self.n_l + i] for i in range(self.n_u))
        return self.model.max() + max(float(hf_shift(const_l, const_u, t[1], t[2])) for t in self.transitions)

    def intervals(self) -> list[list[float]]:
        const_l = tuple(self.vals[self.model.size + i] for i in range(self.n_l))
        const_u = tuple(self.vals[self.model.size + self.n_l + i] for i in range(self.n_u))
        shifts = [hf_shift(const_l, const_u, t[1], t[2]) for t in self.transitions]
        return merge_intervals([[self.model.min() + shift, self.model.max() + shift] for shift in shifts]).tolist()

    def racah(self) -> None:
        self.vals[self.racah_indices[0]] = 0.0


class HyperfineMixed(Splitter):
    def __init__(
        self, model: Model, i: scalar, j_l: scalar, j_u: scalar, label: str | None = None, config: dict | None = None
    ) -> None:
        r"""
        A hyperfine structure model including the hyperfine-induced mixing of fine-structure states
        with the same $F$, based on [<a href="https://doi.org/10.1103/PhysRevA.55.2728">
        W. R. Johnson <i>et al.</i>, Phys. Rev. A <b>55</b>, 2728 (1997)</a>].
        For the mixed lower or upper states, the peak positions are the eigenvalues of Eq. (2.5) of the reference,
        parameterized by the fine-structure energies `FS_l{i}(J)` and `FS_u{i}(J)`. For states without mixing,
        the hyperfine structure constants `A_l{i}(J)`, `B_l{i}(J)`, ... are used as in [`Hyperfine`].
        The intensities are named `int{i}(...)`.

        :param model: A submodel whose parameters are adopted by this model.
        :param i: The nuclear spin quantum number $I$.
        :param j_l: The electronic total angular momentum quantum number $J_l$ of the lower state.
        :param j_u: The electronic total angular momentum quantum number $J_u$ of the upper state.
        :param label: A label for the `Splitter` (isotope / isomer / transition).
        :param config: The configuration of the hyperfine-induced mixing. This is a dictionary such as
         `{"enabled_l": False, "enabled_u": False, "Jl": [0.5], "Ju": [0.5], "Tl": [[1.0]], "Tu": [[1.0]],
         "fl": [0.0], "fu": [0.0], "mu": 0.0}`, where
         - `"enabled_l"`/`"enabled_u"` decide if the corresponding lower/upper states mix,
         - `"Jl"`/`"Ju"` are lists of the lower/upper state $J$, including `j_l` and `j_u`,
         - `"Tl"`/`"Tu"` are the matrix representations of the electronic magnetic dipole operator, see the reference,
         - `"fl"`/`"fu"` are lists of the initial fine-structure energies (can be omitted) and
         - `"mu"` is the magnetic dipole moment of the nucleus ($\mu_\mathrm{N}$).

         If `None`, the mixing is disabled.
        """
        self.model: Model
        super().__init__(model, i, j_l, j_u, label)
        self.type = "HyperfineMixed"
        if config is None:
            config = {
                "enabled_l": False,
                "enabled_u": False,
                "Jl": [
                    self.j_l,
                ],
                "Ju": [
                    self.j_u,
                ],
                "Tl": [[1.0]],
                "Tu": [[1.0]],
                "fl": [
                    0.0,
                ],
                "fu": [
                    0.0,
                ],
                "mu": 0.0,
            }
        self.config = config
        self.states = ["l", "u"]
        to_mhz = 13074.70

        self.enabled = {s: self.config[f"enabled_{s}"] for s in self.states}

        self.J = {s: np.array(self.config[f"J{s}"]) for s in self.states}

        self.F = {s: get_all_f(self.i, self.J[s]) for s in self.states}

        self.T = {s: np.array(self.config[f"T{s}"], dtype=float) * to_mhz for s in self.states}

        self.M = 0.0 if self.i == 0 else np.sqrt((2 * self.i + 1) * (self.i + 1) / self.i) * self.config["mu"]

        self.fs = {s: np.array(self.config.get(f"f{s}", np.zeros(self.J[s].shape))) for s in self.states}

        self.mask_J = {
            s: [
                np.array(
                    [i for i, j in enumerate(self.config[f"J{s}"]) if abs(self.i - j) - 0.1 < f < self.i + j + 0.1],
                    dtype=int,
                )
                for f in self.F[s]
            ]
            for s in self.states
        }

        self.W = {
            s: [
                np.array(
                    [
                        [
                            (-1) ** (self.i + self.J[s][i] + f)
                            * wigner_6j(self.i, self.J[s][i], f, self.J[s][j], self.i, 1)
                            for j in self.mask_J[s][k]
                        ]
                        for i in self.mask_J[s][k]
                    ],
                    dtype=float,
                )
                * self.T[s][np.ix_(self.mask_J[s][k], self.mask_J[s][k])]
                * self.M
                for k, f in enumerate(self.F[s])
            ]
            for s in self.states
        }

        self.transitions = [
            [
                (_m0, _m1),
                (self.J["l"][_m0], self.J["u"][_m1]),
                (f0, f1),
                hf_coeff(self.i, self.J["l"][_m0], f0),
                hf_coeff(self.i, self.J["u"][_m1], f1),
            ]
            for i, (m0, f0) in enumerate(zip(self.mask_J["l"], self.F["l"]))
            for j, (m1, f1) in enumerate(zip(self.mask_J["u"], self.F["u"]))
            for _m0 in m0
            for _m1 in m1
            if abs(f1 - f0) < 1.1 and abs(self.J["u"][_m1] - self.J["l"][_m0]) < 1.1
        ]

        self.wt_map = {
            s: np.array(
                [
                    [
                        (1 - 2 * int(s == "l")) * int(t[0][int(s == "u")] == _m and t[2][int(s == "u")] == f)
                        for m, f in zip(self.mask_J[s], self.F[s])
                        for _m in m
                    ]
                    for t in self.transitions
                ],
                dtype=float,
            )
            for s in self.states
        }

        self.racah_intensities = [a(self.i, t[1][0], t[2][0], t[1][1], t[2][1]) for t in self.transitions]

        self.order = {s: [int(min((j // 0.5, self.i // 0.5))) for j in self.J[s]] for s in self.states}

        self.hf_args_map: list[list[int]] = [[] for _ in self.transitions]
        for s in self.states:
            if self.enabled[s] and self.i > 0:
                for i, j in enumerate(self.J[s]):
                    self._add_arg(f"FS_{s}{i}({j})", self.fs[s][i], i == 0, False)
            else:
                for i, j in enumerate(self.J[s]):
                    for k in range(self.order[s][i]):
                        for t, m in zip(self.transitions, self.hf_args_map):
                            if t[0][int(s == "u")] == i:
                                m.append(self._index)
                        self._add_arg(f"{ascii_uppercase[k]}_{s}{i}({j})", 0.0, False, False)

        for i, (t, intensity) in enumerate(zip(self.transitions, self.racah_intensities)):
            self.racah_indices.append(self._index)
            self._add_arg(f"int{i}([{t[1][0]}, {t[2][0]}] -> [{t[1][1]}, {t[2][1]}])", intensity, i == 0, False)

    def x0(self, *args: scalar) -> ndarray:
        """
        The peak positions of all transitions.

        :param args: The function parameters.
        :returns: (x0) The peak positions.
        """
        x0 = np.zeros(len(self.transitions))
        for s in self.states:
            if self.enabled[s]:
                eigs = [
                    np.linalg.eigh(
                        np.diag([np.asarray(args[self.p[f"FS_{s}{_m}({self.J[s][_m]})"]], dtype=float) for _m in m]) + w
                    )
                    for m, w in zip(self.mask_J[s], self.W[s])
                ]
                orders = [np.argmax(np.abs(w[1]), axis=0) for w in eigs]
                orders = [
                    np.array([j for i in range(order.size) for j, o in enumerate(order) if i == o])
                    for w, order in zip(eigs, orders)
                ]
                _x0 = np.concatenate(tuple(w[0][order] for w, order in zip(eigs, orders)))
                _x0 = self.wt_map[s] @ _x0
            else:
                _x0 = np.array(
                    [
                        sum(args[i] * coeff for i, coeff in zip(m, t[3 + int(s == "u")]))
                        for m, t in zip(self.hf_args_map, self.transitions)
                    ]
                )
                _x0 *= 1 - 2 * int(s == "l")
            x0 += _x0
        return x0

    def evaluate(self, x: ndarray, *args: scalar, **kwargs: Any) -> ndarray:
        return np.sum(
            [
                args[i] * self.model.evaluate(x - _x0, *args, **kwargs)
                for i, _x0 in zip(self.racah_indices, self.x0(*args))
            ],
            axis=0,
        )

    def min(self) -> float:
        return self.model.min() + np.min(self.x0(*self.vals))

    def max(self) -> float:
        return self.model.max() + np.max(self.x0(*self.vals))

    def intervals(self) -> list[list[float]]:
        return merge_intervals(
            [[self.model.min() + _x0, self.model.max() + _x0] for _x0 in self.x0(*self.vals)]
        ).tolist()

    def racah(self) -> None:
        for i, intensity in zip(self.racah_indices, self.racah_intensities):
            self.vals[i] = intensity


class HyperfineZeeman(Splitter):
    def __init__(
        self,
        model: Model,
        i: scalar,
        j_l: scalar,
        j_u: scalar,
        gi: scalar,
        gj_l: scalar,
        gj_u: scalar,
        f_l: scalar | None = None,
        f_u: scalar | None = None,
        q: array_like | None = None,
        rho_ml: Iterable[array_like] | None = None,
        rho_as_par: bool = False,
        scale_par_with_shift: str | None = None,
        linear: bool = False,
        label: str | None = None,
    ) -> None:
        r"""
        A hyperfine structure model in an external magnetic field, summing the submodel over all allowed transitions
        $|F_lm_l\rangle\rightarrow|F_um_u\rangle$. The Zeeman shifts are calculated with
        [`qspec.physics.zeeman_linear`] if `linear` is `True`, else with [`qspec.physics.hyper_zeeman_num`].
        The relative intensities are the Racah intensities [`hf_int`], multiplied by the squared Clebsch-Gordan
        coefficients, the polarization components $q_{m_u - m_l}$ and the lower state populations $\rho(F_l, m_l)$.
        In addition to the parameters of [`Hyperfine`], the magnetic field is the parameter `B_field` (T).

        :param model: A submodel whose parameters are adopted by this model.
        :param i: The nuclear spin quantum number $I$.
        :param j_l: The electronic total angular momentum quantum number $J_l$ of the lower state.
        :param j_u: The electronic total angular momentum quantum number $J_u$ of the upper state.
        :param gi: The nuclear g-factor $g_I$.
        :param gj_l: The electronic g-factor $g_J$ of the lower state.
        :param gj_u: The electronic g-factor $g_J$ of the upper state.
        :param f_l: The total angular momentum quantum number(s) $F_l$ of the lower state to consider.
         If `None`, all are considered.
        :param f_u: The total angular momentum quantum number(s) $F_u$ of the upper state to consider.
         If `None`, all are considered.
        :param q: The polarization vector $\vec{q} = (q_{-1}, q_0, q_1)$ in the helicity basis,
         or an integer in `{-1, 0, 1}` for pure $\sigma^-$, $\pi$ or $\sigma^+$ light.
         If `None`, all components are equal. The vector is normalized to a sum of 1.
        :param rho_ml: The lower state populations. Must be an iterable of arrays.
         If `None`, the default list is given by `rho_ml = [np.ones(2 * f + 1) for f in f_l]`.
        :param rho_as_par: Whether to parameterize the transition intensities by the lower state populations
         `rho(F, m)` and the polarization components `q(-1)`, `q(0)` and `q(1)` (`True`)
         instead of one intensity parameter per transition (`False`).
        :param scale_par_with_shift: Choose a parameter that is scaled with the Zeeman shift
         such as the Lorentz width `"Gamma"`.
        :param linear: Whether the Zeeman shift is linear (`True`) or nonlinear (`False`). If nonlinear,
         the Zeeman shift is calculated numerically, decreasing computing speed.
        :param label: A label for the `Splitter` (isotope / isomer / transition).
        :raises ValueError: If `q` is neither a 3d-vector nor in `{-1, 0, 1}`.
        """
        self.model: Model
        super().__init__(model, i, j_l, j_u, label=label)
        self.type = "HyperfineZeeman"

        self.gi = gi
        self.gj_l = gj_l
        self.gj_u = gj_u

        self.linear = linear

        if f_l is None:
            self.f_l = np.asarray(get_f(self.i, self.j_l), dtype=float)
        elif has_getitem(f_l):
            self.f_l = np.array(f_l, dtype=float)
        else:
            self.f_l = np.array([f_l], dtype=float)

        if f_u is None:
            self.f_u = np.asarray(get_f(self.i, self.j_u), dtype=float)
        elif has_getitem(f_u):
            self.f_u = np.array(f_u, dtype=float)
        else:
            self.f_u = np.array([f_u], dtype=float)

        if q is None:
            self.q = np.array([1.0, 1.0, 1.0])
        elif has_getitem(q):
            self.q = np.array(q, dtype=float)
        elif is_scalar(q) and int(q) in {-1, 0, 1}:
            self.q = unit_vector(int(q) + 1, 3, dtype=float)
        else:
            raise ValueError("The polarization vector `q` must be a 3d vector or in `{-1, 0, 1}`")
        self.q /= np.sum(self.q)

        if rho_ml is None:
            self.rho_ml = [np.ones(int(2 * f + 1), dtype=float) for f in self.f_l]
        else:
            _rho_ml = [np.asarray(r, dtype=float) for r in rho_ml]
            self.rho_ml = [r / np.sum(r) * (2 * f + 1) for f, r in zip(self.f_l, _rho_ml)]

        self.rho_as_par = rho_as_par
        self.rho_indexes: list[tuple[float, float]] = []

        self.transitions = hf_trans(self.i, self.j_l, self.j_u)
        self.transitions_m: list[list[Any]] = []
        self._racah_intensities = hf_int(self.i, self.j_l, self.j_u, self.transitions)
        self.racah_intensities = []

        self.n_l = len(self.transitions[0][1])
        self.n_u = len(self.transitions[0][2])
        for i in range(self.n_l):
            self._add_arg(f"{ascii_uppercase[i]}l", 0.0, False, False)
        for i in range(self.n_u):
            self._add_arg(f"{ascii_uppercase[i]}u", 0.0, False, False)

        for i in range(min([self.n_l, self.n_u])):
            fix = "{} / {}".format(f"{ascii_uppercase[i]}u", f"{ascii_uppercase[i]}l")
            self._add_arg(f"{ascii_uppercase[i]}_ratio", 0.0, fix, False)

        self._add_arg("B_field", 0.0, False, False)

        for i, (t, intensity) in enumerate(zip(self.transitions, self._racah_intensities)):
            if t[0][0] not in self.f_l or t[0][1] not in self.f_u:
                continue

            _f_l, _f_u = t[0]
            index_fl = np.argmax(self.f_l == _f_l)
            ml_list = get_m(_f_l)
            g_l = lande_f(self.i, self.j_l, _f_l, self.gi, self.gj_l)
            g_u = lande_f(self.i, self.j_u, _f_u, self.gi, self.gj_u)

            self._index_0 = self._index

            if self.rho_as_par:
                for iq, _q in enumerate(self.q):
                    self._add_arg(f"q({iq - 1})", _q, True, False)

            for iq, _q in enumerate(self.q):
                if _q == 0.0:
                    continue

                for im, m_l in enumerate(ml_list):
                    population = self.rho_ml[index_fl][im]

                    if (_f_l, m_l) not in self.rho_indexes:
                        self._add_arg(f"rho({_f_l}, {m_l})", population, [0.0, 2 * _f_l + 1], False)
                        self.rho_indexes.append((_f_l, m_l))

                    m_u = quant(m_l + iq - 1)
                    if abs(m_u) > _f_u:
                        continue

                    cg = clebsch_gordan(_f_l, 1, _f_u, m_l, iq - 1, m_u) ** 2
                    _intensity = float(3 * intensity * cg / (2 * _f_u + 1) * 1e4)

                    if self.rho_as_par:
                        self.racah_intensities.append(population)
                        self.racah_indices.append(self._index_0 + im + 3)
                        self.transitions_m.append([(_f_l, _f_u), (m_l, m_u), (g_l, g_u), _intensity, t[1], t[2]])
                    else:
                        self.racah_indices.append(self._index)
                        self._add_arg(f"int[({_f_l}, {m_l}), ({_f_u}, {m_u})]", _intensity, True, False)
                        self.racah_intensities.append(_intensity * _q * population)
                        self.transitions_m.append([(_f_l, _f_u), (m_l, m_u), (g_l, g_u), population, t[1], t[2]])

        self.scale_par_with_shift = ""
        self.index_scale_with_sift = -1
        self.index_shift_scale_par = -1
        if scale_par_with_shift is not None:
            self.scale_par_with_shift = scale_par_with_shift
            self.index_scale_with_sift = self.p.get(self.scale_par_with_shift, -1)
            self.index_shift_scale_par = self._index
            self._add_arg("shift_scale_par", 0.0, False, False)

    def evaluate(self, x: ndarray, *args: scalar, **kwargs: Any) -> ndarray:
        const_l = tuple(args[self.model.size + i] for i in range(self.n_l))
        const_u = tuple(args[self.model.size + self.n_l + i] for i in range(self.n_u))

        if self.linear:
            return np.sum(
                [
                    args[i]
                    * t[3]
                    * self.model.evaluate(
                        x
                        - hfm_lin_shift(
                            const_l, const_u, t[4], t[5], t[1][0], t[1][1], t[2][0], t[2][1], args[self._index_0 - 1]
                        ),
                        *args,
                        **kwargs,
                    )
                    for i, t in zip(self.racah_indices, self.transitions_m)
                ],
                axis=0,
            )

        hyper_const = np.array(
            [
                0.0 if len(const_l) < 1 else const_l[0],
                0.0 if len(const_l) < 2 else const_l[1],
                0.0 if len(const_l) < 3 else const_l[2],
            ],
            dtype=float,
        )
        shifts_l = hyper_zeeman_num(
            self.i,
            self.j_l,
            hyper_const,
            gi=self.gi,
            gj=self.gj_l,
            b_field=args[self._index_0 - 1],
            g_n_as_gyro=False,
            as_freq=True,
        )
        hyper_const = np.array(
            [
                0.0 if len(const_u) < 1 else const_u[0],
                0.0 if len(const_u) < 2 else const_u[1],
                0.0 if len(const_u) < 3 else const_u[2],
            ],
            dtype=float,
        )
        shifts_u = hyper_zeeman_num(
            self.i,
            self.j_u,
            hyper_const,
            gi=self.gi,
            gj=self.gj_u,
            b_field=args[self._index_0 - 1],
            g_n_as_gyro=False,
            as_freq=True,
        )

        shifts, args_list = [], []
        intensities = []
        for i, t in zip(self.racah_indices, self.transitions_m):
            il_m = shifts_l[1].index(t[1][0])
            il_f = shifts_l[2][il_m].index(t[0][0])

            iu_m = shifts_u[1].index(t[1][1])
            iu_f = shifts_u[2][iu_m].index(t[0][1])

            shift0 = hf_shift(const_l, const_u, t[4], t[5])
            shifts.append(shifts_u[0][iu_m][:, iu_f] - shifts_l[0][il_m][:, il_f])
            args_list.append([arg for arg in args])

            iq = int(t[1][1] - t[1][0] + 1)
            if self.rho_as_par:
                intensities.append(args[i] * t[3] * args[self._index_0 + iq])
            else:
                intensities.append(args[i] * t[3])

            if self.index_scale_with_sift != -1:
                args_list[-1][self.index_scale_with_sift] *= 1 + args[self.index_shift_scale_par] * np.abs(
                    shifts[-1] - shift0
                )

        return np.sum(
            [
                intensities[index] * self.model.evaluate(x - shifts[index], *args_list[index], **kwargs)
                for index, (i, t) in enumerate(zip(self.racah_indices, self.transitions_m))
            ],
            axis=0,
        )

    def min(self) -> float:
        const_l = tuple(self.vals[self.model.size + i] for i in range(self.n_l))
        const_u = tuple(self.vals[self.model.size + self.n_l + i] for i in range(self.n_u))
        return self.model.min() + np.min(
            [
                hfm_lin_shift(
                    const_l, const_u, t[4], t[5], t[1][0], t[1][1], t[2][0], t[2][1], self.vals[self._index_0 - 1]
                )
                for t in self.transitions_m
            ]
        )

    def max(self) -> float:
        const_l = tuple(self.vals[self.model.size + i] for i in range(self.n_l))
        const_u = tuple(self.vals[self.model.size + self.n_l + i] for i in range(self.n_u))
        return self.model.max() + np.max(
            [
                hfm_lin_shift(
                    const_l, const_u, t[4], t[5], t[1][0], t[1][1], t[2][0], t[2][1], self.vals[self._index_0 - 1]
                )
                for t in self.transitions_m
            ]
        )

    def intervals(self) -> list[list[float]]:
        const_l = tuple(self.vals[self.model.size + i] for i in range(self.n_l))
        const_u = tuple(self.vals[self.model.size + self.n_l + i] for i in range(self.n_u))
        shifts = [
            hfm_lin_shift(
                const_l, const_u, t[4], t[5], t[1][0], t[1][1], t[2][0], t[2][1], self.vals[self._index_0 - 1]
            )
            for t in self.transitions_m
        ]
        return merge_intervals([[self.model.min() + shift, self.model.max() + shift] for shift in shifts]).tolist()

    def racah(self) -> None:
        for i, intensity in zip(self.racah_indices, self.racah_intensities):
            self.vals[i] = intensity
