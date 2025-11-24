/*
* PyCLS.Physics.cpp
* 
* Factorial, GCcoeff, ThreeJSymbol, SixJSymbol and NineJSymbol taken and modified from
* https://nukephysik101.wordpress.com/2019/01/30/3j-6j-9j-symbol-fro-c/
* by Tsz Leung Tang (Also known as Ryan Tang)
* Orcid: 0000-0001-5527-076X
*/

#include "pch.h"
#include "Physics.h"

using namespace std::complex_literals;
using namespace WignerSymbols;

extern std::complex<double> sc::i = 1i;
extern double sc::h = 6.62607015e-34;
extern double sc::hbar = 1.054571817e-34;
extern double sc::c = 299792458.;
extern double sc::e = 1.602176634e-19;
extern double sc::amu = 1.66053906660e-27;
extern double sc::epsilon_0 = 8.8541878128e-12;
extern double sc::g_s = -2.00231930436256;
extern double sc::mu_B = 9.2740100783e-24;
extern double sc::mu_N = 5.0507837461e-27;


double wigner_d_qm(size_t j, int q, int m, double theta)
{
    int _j = static_cast<int>(j);
    double jd = static_cast<double>(j);
    double qd = static_cast<double>(q);
    double md = static_cast<double>(m);

    int N = min(j + q, min(j - q, min(j + m, j - m))) + 1;
    std::vector<int> k_list;
    for (int k = 0; k <= 2 * _j; ++k)
        if (   _j - q - k >= 0 
            && _j - m - k >= 0 
            &&  q + m + k >= 0) k_list.push_back(k);

    // printf("j, q, m, theta: %zi, %d, %d, %.3f\n", j, q, m, theta);

    double ret = 0.;
    for (int k: k_list)
    {
        double kd = static_cast<double>(k);
        ret += pow(-1., k) * pow(cos(0.5 * theta), q + m + 2 * k) * pow(sin(0.5 * theta), 2 * _j - q - m - 2 * k) 
            / (factorial(kd) * factorial(jd - qd - kd) * factorial(jd - md - kd) * factorial(qd + md + kd));
        // printf("fac: %.3f\n", factorial(kd) * factorial(jd - qd - kd) * factorial(jd - md - kd) * factorial(qd + md + kd));
    }
    // printf("sqrt_arg: %.3f\n", factorial(jd + qd) * factorial(jd - qd) * factorial(jd + md) * factorial(jd - md));
    // printf("-1: %.3f\n", pow(-1, _j - m));
    return ret * pow(-1., _j - m) * sqrt(factorial(jd + qd) * factorial(jd - qd) * factorial(jd + md) * factorial(jd - md));
}


std::complex<double> wigner_D_qm(size_t j, int q, int m, double theta, double phi)
{
    // printf("q, m: %d, %d\n", q, m);
    // printf("d_qm: %.6f\n", wigner_d_qm(j, q, m, theta));
    return exp(sc::i * static_cast<double>(m) * phi) * wigner_d_qm(j, q, m, theta);
}

VectorXcd spherical_tensor(bool electric, size_t k, Vector3cd qk, double theta, double phi)
{
    // printf("theta, phi: %.3f, %.3f\n", theta, phi);
    VectorXcd ret = VectorXcd::Zero(2 * k + 1);

    double pm = -1.;
    std::complex<double> phase = -sc::i;
    if (electric) {
        pm = 1.;
        phase = 1.;
    }

    for (size_t im = 0; im < 2 * k + 1; ++im)
    {
        int m = static_cast<int>(im) - static_cast<int>(k);
        ret(im) = qk(0) * wigner_D_qm(k, 1, -m, theta, phi) + pm * qk(2) * wigner_D_qm(k, -1, -m, theta, phi);

        if (abs(ret(im).real()) < 1e-15) ret.real()(im) = 0.;
        if (abs(ret(im).imag()) < 1e-15) ret.imag()(im) = 0.;
    }
    ret *= phase;
    ret /= ret.norm();

    // double _k = static_cast<double>(k);
    return ret;  // sqrt((_k + 1.) / (2. * _k)) / double_factorial(2. * _k - 1.)
}

double d_e1(double a, double freq_0, double freq_1)
{
    return sqrt(3 * a * pow(sc::c, 2) / (2 * sc::pi * sc::h * pow( abs(freq_0 - freq_1), 3))) * 1e-12;
}

double d_m1(double a, double freq_0, double freq_1)
{
    return d_e1(a, freq_0, freq_1);  // d_m1 = d_e1 * c, but c cancels in calculation of Rabi frequency.
}

double a_dipole(double i, double j_l, double f_l, double m_l, double j_u, double f_u, double m_u, double q)
{
    if (abs(m_u - m_l - q) > 0.1) return 0.;
    double sqrt_f = sqrt(2 * f_l + 1);
    double sqrt_j = sqrt(2 * j_u + 1);
    double exp = f_l + i + 1 + j_u;
    // double w6j = wigner_6j(j_u, j_l, 1, f_l, f_u, i);
    // double cg = clebsch_gordan(f_l, m_l, 1, q, f_u, m_u);
    double w6j = wigner6j(j_u, j_l, 1, f_l, f_u, i);
    double cg = clebschGordan(f_l, 1, f_u, m_l, q, m_u);
    return pow(-1, exp) * sqrt_f * sqrt_j * w6j * cg;  // wigner6j(j_u, j_l, 1, f_l, f_u, i) * clebschGordan(f_u, f_l, 1, m_u, m_l, q)
}

double d_emk(size_t k, bool parity_equal, double a, double freq_0, double freq_1)
{
    double k_mult = 1.;
    // if (k > 1) k_mult = 0.5 * pow(abs(freq_0 - freq_1) / sc::c, k - 1) / (factorial(k - 1));
    // return k_mult * sqrt(3 * a * pow(sc::c, 2) / (2 * sc::pi * sc::h * pow(abs(freq_0 - freq_1), 3))) * 1e-12;
    return k_mult * sc::c * sqrt(a * (2 * k + 1) / (2 * sc::pi * sc::h * pow(abs(freq_0 - freq_1), 3))) * 1e-12;
}

double a_multipole(double k, double i, double j_l, double f_l, double m_l, double j_u, double f_u, double m_u, double q)
{
    double sqrt_f = sqrt(2 * f_l + 1);
    double sqrt_j = sqrt(2 * j_u + 1);
    double exp = f_l + i + k + j_u;
    // printf("W6j: {%3.3f, %3.3f, %3.3f, %3.3f, %3.3f, %3.3f}\n", j_u, j_l, k, f_l, f_u, i);
    // printf("CGC: {%3.3f, %3.3f, %3.3f, %3.3f, %3.3f, %3.3f}\n", f_l, k, f_u, m_l, q, m_u);
    // double w6j_n = wigner_6j(j_u, j_l, k, f_l, f_u, i);
    // double cg_n = clebsch_gordan(f_l, k, f_u, m_l, q, m_u);
    double w6j = wigner6j(j_u, j_l, k, f_l, f_u, i);
    double cg = clebschGordan(f_l, k, f_u, m_l, q, m_u);
    // if (w6j_n != w6j) printf("w6j: %s\n", std::format("{:.3e}", w6j_n - w6j).c_str());
    // if (cg_n != cg) printf("cg: %s\n", std::format("{:.3e}", cg_n - cg).c_str());
    // printf("%s\n", std::format("  w6j,   cg: {:.3e}, {:.3e}", w6j, cg).c_str());
    // if (k == 1.) printf("%s\n", std::format("w6j_n, cg_n: {:.3e}, {:.3e}", w6j_n, cg_n).c_str());
    return pow(-1, exp) * sqrt_f * sqrt_j * w6j * cg;  // wigner6j(j_u, j_l, 1, f_l, f_u, i) * clebschGordan(f_u, f_l, 1, m_u, m_l, q)
}

double lande_n(double gyro)
{
    return gyro * sc::h / sc::mu_N;
}

double lande_j(double s, double l, double j)
{
    if (j == 0) return 0.;
    double jj = j * (j + 1);
    double ls = l * (l + 1) - s * (s + 1);
    double ret = -(jj + ls) / (2 * jj);
    ret += (jj - ls) / (2 * jj) * sc::g_s;
    return ret;
}

double lande_f(double i, double j, double f, double g_n, double g_j)
{
    if (f == 0) return 0.;
    double ff = f * (f + 1.);
    double ji = j * (j + 1.) - i * (i + 1.);
    double ret = (ff + ji) / (2 * ff) * g_j;
    ret += (ff - ji) / (2 * ff) * g_n * sc::mu_N / sc::mu_B;
    return ret;
}

double hyperfine(double i, double j, double f, double* hyper_const)
{
    if (i == 0 || j == 0) return 0.;
    double k_0 = f * (f + 1) - i * (i + 1) - j * (j + 1);
    double shift = 0.5 * hyper_const[0] * k_0;
    if (i > 0.5 && j > 0.5)
    {
        double k_1 = 3 * k_0 * (k_0 + 1) / 2 - 2 * i * (i + 1) * j * (j + 1);
        k_1 /= i * (2 * i - 1) * j * (2 * j - 1);
        shift += 0.25 * hyper_const[1] * k_1;
    }
    if (i > 1 && j > 1)
    {
        double k_2 = std::pow(k_0, 3) + 4 * std::pow(k_0, 2) + 0.8 * k_0 * (-3 * i * (i + 1) * j * (j + 1) + i * (i + 1) + j * (j + 1) + 3)
            - 4 * i * (i + 1) * j * (j + 1);
            k_2 /= i * (i - 1) * (2 * i - 1) * j * (j - 1) * (2 * j - 1);
        shift += 1.25 * hyper_const[2] * k_2;
    }
    return shift;
}

double zeeman(double m, double b, double g)
{
    return -g * m * sc::mu_B * b / sc::h * 1e-6;
}

double hyper_zeeman_linear(double i, double j, double f, double m, double g_j, double g_n, double* hyper_const, double b, bool g_n_as_gyro)
{
    double _g_n = g_n;
    if (g_n_as_gyro) _g_n = lande_n(g_n);
    double g_f = 0.;
    if (i != 0 || j != 0) g_f = lande_f(i, j, f, _g_n, g_j);
    double ret = hyperfine(i, j, f, hyper_const);
    ret += zeeman(m, b, g_f);
    return ret;
}

/*double hyper_zeeman_ij(double mi0, double mj0, double mi1, double mj1, double i, double j, double g_j, double g_n, double* hyper_const, double b)
{
    double b_field = b * 1e-6 / sc::h;
    double b_hyper_n = 0.;
    if (i > 0.5 && j > 0.5)
    {
        b_hyper_n = hyper_const[1] / (2 * i * (2 * i - 1) * j * (2 * j - 1));
    }

    if (mi0 + mj0 != mi1 + mj1) return 0;

    else if (mi0 == mi1 && mj0 == mj1)
    {
        double ret = hyper_const[0] * mi0 * mj0 - (mi0 * g_n * sc::mu_N + mj0 * g_j * sc::mu_B) * b_field;
        ret += b_hyper_n * (3 * pow(mi0 * mj0, 2) - i * (i + 1) * j * (j + 1) + 1.5 * mi0 * mj0
            + 0.75 * (j - mj0) * (j + mj0 + 1) * (i + mi0) * (i - mi0 + 1)
            + 0.75 * (i - mi0) * (i + mi0 + 1) * (j + mj0) * (j - mj0 + 1));
        return ret;
    }

    else if (mi0 == mi1 + 1 && mj0 == mj1 - 1)
    {
        return (0.5 * hyper_const[0] + 1.5 * b_hyper_n * (0.5 + mi0 * mj0 + mi1 * mj1))
            * sqrt((i - mi1) * (i + mi1 + 1) * (j + mj1) * (j - mj1 + 1));
    }

    else if (mi0 == mi1 - 1 && mj0 == mj1 + 1)
    {
        return (0.5 * hyper_const[0] + 1.5 * b_hyper_n * (0.5 + mi0 * mj0 + mi1 * mj1))
            * sqrt((i + mi1) * (i - mi1 + 1) * (j - mj1) * (j + mj1 + 1));
    }

    else if (mi0 == mi1 + 2 && mj0 == mj1 - 2)
    {
        return 0.75 * b_hyper_n * sqrt((j + mj1) * (j - mj1 + 1) * (j + mj1 - 1) * (j - mj1 + 2)
            * (i - mi1) * (i + mi1 + 1) * (i - mi1 - 1) * (i + mi1 + 2));
    }

    else if (mi0 == mi1 - 2 && mj0 == mj1 + 2)
    {
        return 0.75 * b_hyper_n * sqrt((i + mi1) * (i - mi1 + 1) * (i + mi1 - 1) * (i - mi1 + 2)
            * (j - mj1) * (j + mj1 + 1) * (j - mj1 - 1) * (j + mj1 + 2));
    }

    return 0;
}*/

double hyper_zeeman_ij(double mi0, double mj0, double mi1, double mj1, double i, double j, double g_j, double g_n, double* hyper_const, double b)
{   
    // The matrix element expressions have been generated using the sympy python package, see qspec/resources/hfs_algebra.py.
    double b_field = b * 1e-6 / sc::h;
    double a_hyper = hyper_const[0];
    double b_hyper_n = 0.;
    double c_hyper_n = 0.;
    if (i > 0.5 && j > 0.5)
    {
        b_hyper_n = hyper_const[1] / (4 * i * (2 * i - 1) * j * (2 * j - 1));
    }

    if (i > 1. && j > 1.)
    {
        c_hyper_n = hyper_const[2] / (i * (2 * i - 1) * (i - 1) * j * (2 * j - 1) * (j - 1));
    }

    if (mi0 + mj0 != mi1 + mj1) return 0;

    else if (mi0 == mi1 && mj0 == mj1)
    {
        double ret =  -(mi1 * g_n * sc::mu_N + mj1 * g_j * sc::mu_B) * b_field;

        ret += a_hyper * mi1 * mj1;
        ret += b_hyper_n * (i * j
            + pow(i, 2) * j
            + i * pow(j, 2)
            + pow(i, 2) * pow(j, 2)
            - 3 * j * pow(mi1, 2)
            - 3 * i * pow(mj1, 2)
            - 3 * pow(j, 2) * pow(mi1, 2)
            - 3 * pow(i, 2) * pow(mj1, 2)
            + 9 * pow(mi1, 2) * pow(mj1, 2));
        ret += c_hyper_n * (mi1 * mj1 * (1
            - 3 * i
            - 3 * j
            - 3 * pow(i, 2)
            - 3 * pow(j, 2)
            + 9 * i * j
            + 9 * pow(i, 2) * j
            + 9 * i * pow(j, 2)
            + 9 * pow(i, 2) * pow(j, 2)
            + 5 * pow(mi1, 2)
            + 5 * pow(mj1, 2)
            - 15 * i * pow(mj1, 2)
            - 15 * j * pow(mi1, 2)
            - 15 * pow(i, 2) * pow(mj1, 2)
            - 15 * pow(j, 2) * pow(mi1, 2)
            + 25 * pow(mi1, 2) * pow(mj1, 2)));
        return ret;
    }

    else if (mi0 == mi1 + 1 && mj0 == mj1 - 1)
    {
        double ret = a_hyper * 0.5
            * sqrt((i - mi1) * (i + mi1 + 1))
            * sqrt((j + mj1) * (j - mj1 + 1));
        ret += (b_hyper_n * 3.
            * sqrt(pow(i, 2) + i - pow(mi1, 2) - mi1)
            * sqrt(pow(j, 2) + j - pow(mj1, 2) + mj1)
            * (2 * mi1 * mj1 - mi1 + mj1 - 0.5));
        ret += (c_hyper_n * 0.75
            * sqrt(pow(i, 2) + i - pow(mi1, 2) - mi1)
            * sqrt(pow(j, 2) + j - pow(mj1, 2) + mj1)
            * (pow(i, 2) * pow(j, 2) + pow(i, 2) * j - 5 * pow(i, 2) * pow(mj1, 2) + 5 * pow(i, 2) * mj1 - 2 * pow(i, 2) + i * pow(j, 2)
                + i * j - 5 * i * pow(mj1, 2) + 5 * i * mj1 - 2 * i - 5 * pow(j, 2) * pow(mi1, 2) - 5 * pow(j, 2) * mi1
                - 2 * pow(j, 2) - 5 * j * pow(mi1, 2) - 5 * j * mi1 - 2 * j + 25 * pow(mi1, 2) * pow(mj1, 2)
                - 25 * pow(mi1, 2) * mj1 + 10 * pow(mi1, 2) + 25 * mi1 * pow(mj1, 2) - 25 * mi1 * mj1 + 10 * mi1
                + 10 * pow(mj1, 2) - 10 * mj1 + 4));
        return ret;
    }

    else if (mi0 == mi1 - 1 && mj0 == mj1 + 1)
    {
        double ret = (a_hyper * 0.5
            * sqrt((i + mi1) * (i - mi1 + 1))
            * sqrt((j - mj1) * (j + mj1 + 1)));
        ret += (b_hyper_n * 3.
            * sqrt(pow(i, 2) + i - pow(mi1, 2) + mi1)
            * sqrt(pow(j, 2) + j - pow(mj1, 2) - mj1)
            * (2 * mi1 * mj1 + mi1 - mj1 - 0.5));
        ret += (c_hyper_n * 0.75
            * sqrt(pow(i, 2) + i - pow(mi1, 2) + mi1)
            * sqrt(pow(j, 2) + j - pow(mj1, 2) - mj1)
            * (pow(i, 2) * pow(j, 2) + pow(i, 2) * j - 5 * pow(i, 2) * pow(mj1, 2) - 5 * pow(i, 2) * mj1 - 2 * pow(i, 2) + i * pow(j, 2)
                + i * j - 5 * i * pow(mj1, 2) - 5 * i * mj1 - 2 * i - 5 * pow(j, 2) * pow(mi1, 2) + 5 * pow(j, 2) * mi1
                - 2 * pow(j, 2) - 5 * j * pow(mi1, 2) + 5 * j * mi1 - 2 * j + 25 * pow(mi1, 2) * pow(mj1, 2)
                + 25 * pow(mi1, 2) * mj1 + 10 * pow(mi1, 2) - 25 * mi1 * pow(mj1, 2) - 25 * mi1 * mj1 - 10 * mi1
                + 10 * pow(mj1, 2) + 10 * mj1 + 4));
        return ret;
    }

    else if (mi0 == mi1 + 2 && mj0 == mj1 - 2)
    {
        double ret = (b_hyper_n * 1.5
            * sqrt((i - mi1) * (i + mi1 + 1))
            * sqrt((j + mj1) * (j - mj1 + 1))
            * sqrt((j - mj1 + 2) * (j + mj1 - 1))
            * sqrt(-(-i + mi1 + 1) * (i + mi1 + 2)));
        ret += (c_hyper_n * 7.5
            * sqrt(pow(i, 2) + i - pow(mi1, 2) - mi1)
            * sqrt(pow(j, 2) + j - pow(mj1, 2) + mj1)
            * sqrt(pow(i, 2) + i - pow(mi1, 2) - 3 * mi1 - 2)
            * sqrt(pow(j, 2) + j - pow(mj1, 2) + 3 * mj1 - 2)
            * (mi1 * mj1 - mi1 + mj1 - 1));
        return ret;
    }

    else if (mi0 == mi1 - 2 && mj0 == mj1 + 2)
    {
        double ret = (b_hyper_n * 1.5
            * sqrt((i + mi1) * (i - mi1 + 1))
            * sqrt((j - mj1) * (j + mj1 + 1))
            * sqrt((i - mi1 + 2) * (i + mi1 - 1))
            * sqrt(-(-j + mj1 + 1) * (j + mj1 + 2)));
        ret += (c_hyper_n * 7.5
            * sqrt(pow(i, 2) + i - pow(mi1, 2) + mi1)
            * sqrt(pow(j, 2) + j - pow(mj1, 2) - mj1)
            * sqrt(pow(i, 2) + i - pow(mi1, 2) + 3 * mi1 - 2)
            * sqrt(pow(j, 2) + j - pow(mj1, 2) - 3 * mj1 - 2)
            * (mi1 * mj1 + mi1 - mj1 - 1));
        return ret;
    }

    else if (mi0 == mi1 + 3 && mj0 == mj1 - 3)
    {
        double ret = (c_hyper_n * 1.25
            * sqrt((i - mi1) * (i + mi1 + 1))
            * sqrt((j + mj1) * (j - mj1 + 1))
            * sqrt((j - mj1 + 2) * (j + mj1 - 1))
            * sqrt((j - mj1 + 3) * (j + mj1 - 2))
            * sqrt(-(-i + mi1 + 1) * (i + mi1 + 2))
            * sqrt(-(-i + mi1 + 2) * (i + mi1 + 3)));
        return ret;
    }

    else if (mi0 == mi1 - 3 && mj0 == mj1 + 3)
    {
        double ret = (c_hyper_n * 1.25
            * sqrt((i + mi1) * (i - mi1 + 1))
            * sqrt((j - mj1) * (j + mj1 + 1))
            * sqrt((i - mi1 + 2) * (i + mi1 - 1))
            * sqrt((i - mi1 + 3) * (i + mi1 - 2))
            * sqrt(-(-j + mj1 + 1) * (j + mj1 + 2))
            * sqrt(-(-j + mj1 + 2) * (j + mj1 + 3)));
        return ret;
    }

    return 0;
}

EigenReturn hyper_zeeman_num(double i, double j, double m, double g_j, double g_n, double* hyper_const, double b)
{
    double f_min = max(abs(m), abs(i - j));
    double f_max = i + j;
    size_t n = static_cast<size_t>(f_max - f_min + 1);

    EigenReturn ret;

    ret.values = std::vector<double>(n);
    ret.vectors = std::vector<VectorXd>(n, VectorXd::Zero(n));

    MatrixXd h = MatrixXd::Zero(n, n);
    size_t k0 = 0;
    for (double mi0 = -i; mi0 <= i; ++mi0)
    {
        if (abs(m - mi0) <= j)
        {
            size_t k1 = 0;
            for (double mi1 = -i; mi1 <= i; ++mi1)
            {
                if (abs(m - mi1) <= j)
                {
                    h(k0, k1) = hyper_zeeman_ij(mi0, m - mi0, mi1, m - mi1, i, j, g_j, g_n, hyper_const, b);
                    ++k1;
                }
            }
            ++k0;
        }
    }

    SelfAdjointEigenSolver<MatrixXd> eigen_solver(h);
    VectorXd e_eig = eigen_solver.eigenvalues();
    MatrixXd e_vec = eigen_solver.eigenvectors();

    // Find indexes to sort eigenvalues in ascending order regarding F quantum number.
    std::vector<double> e_ref(n);
    size_t k = 0;
    for (double f = f_min; f <= f_max; ++f)
    {
        e_ref.at(k) = hyperfine(i, j, f, hyper_const);
        ++k;
    }
    std::vector<size_t> indexes = invert_order(argsort(e_ref));

    double f = f_min;
    for (size_t k = 0; k < n; ++k)
    {
        ret.values.at(k) = e_eig(indexes.at(k));
        ret.vectors.at(k) = e_vec.col(indexes.at(k));
        ++f;
    }

    return ret;
}

double lorentz(double w, double w0, double a, double rabi_square)
{
    // if (a == 0) return 0;
    return rabi_square * a / (std::pow(w - w0, 2) + std::pow(a, 2) / 4) / 4;
}

double gamma(double v)
{
    return 1 / std::sqrt(1 - std::pow(v / sc::c, 2));
}

double doppler(double x, double v, double angle)
{
    return x * gamma(v) * (1 - v / sc::c * std::cos(angle));
}

double recoil(double freq, double mass)
{
    return sc::h * std::pow(freq, 2) / (2 * mass * sc::amu * std::pow(sc::c, 2)) * 1e6;
}
