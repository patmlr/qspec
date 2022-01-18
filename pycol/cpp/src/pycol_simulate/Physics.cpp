/*
* PyCLS.Physics.cpp
*
* Created on 15.11.2021
*
* @author: Patrick Mueller
* 
* Factorial, GCcoeff, ThreeJSymbol, SixJSymbol and NineJSymbol taken from
* https://nukephysik101.wordpress.com/2019/01/30/3j-6j-9j-symbol-fro-c/
* by Tsz Leung Tang (Also known as Ryan Tang)
* Orcid: 0000-0001-5527-076X
*/

#include "pch.h"
#include "Physics.h"
using namespace std::complex_literals;

extern std::complex<double> sc::i = 1i;
extern double sc::pi = 3.14159265359;
extern double sc::h = 6.62607015e-34;
extern double sc::hbar = 1.054571817e-34;
extern double sc::c = 299792458.;
extern double sc::e = 1.602176634e-19;
extern double sc::amu = 1.66053906660e-27;
extern double sc::epsilon_0 = 8.8541878128e-12;
extern double sc::g_s = -2.00231930436256;
extern double sc::mu_B = 9.2740100783e-24;
extern double sc::mu_N = 5.0507837461e-27;

double factorial(double n) {
    if (n < 0) return -100.;
    return (n == 1. || n == 0.) ? 1. : factorial(n - 1) * n;
}

double CGcoeff(double J, double m, double J1, double m1, double J2, double m2) {
    // (J1,m1) + (J2, m2) = (J, m)

    if (m != m1 + m2) return 0;
    if (abs(m1) > J1 || abs(m2) > J2 || abs(m) > J) return 0;  // Added.

    double Jmin = abs(J1 - J2);
    double Jmax = J1 + J2;

    if (J < Jmin || Jmax < J) return 0;

    double s0 = (2 * J + 1.) * factorial(J + J1 - J2) * factorial(J - J1 + J2) * factorial(J1 + J2 - J) / factorial(J + J1 + J2 + 1.);
    s0 = sqrt(s0);

    double s = factorial(J + m) * factorial(J - m);
    double s1 = factorial(J1 + m1) * factorial(J1 - m1);
    double s2 = factorial(J2 + m2) * factorial(J2 - m2);
    s = sqrt(s * s1 * s2);

    //printf(" s0, s = %f , %f \n", s0, s);

    double kMax = min(min(J1 + J2 - J, J1 - m1), J2 + m2);

    double CG = 0.;
    for (double k = 0; k <= kMax; k++) {
        double k1 = factorial(J1 + J2 - J - k);
        double k2 = factorial(J1 - m1 - k);
        double k3 = factorial(J2 + m2 - k);
        double k4 = factorial(J - J2 + m1 + k);
        double k5 = factorial(J - J1 - m2 + k);
        double temp = pow(-1, k) / (factorial(k) * k1 * k2 * k3 * k4 * k5);
        if (k1 == -100. || k2 == -100. || k3 == -100. || k4 == -100. || k5 == -100.) temp = 0.;

        //printf(" %d | %f \n", k, temp);
        CG += temp;
    }

    return s0 * s * CG;

}

double ThreeJSymbol(double J1, double m1, double J2, double m2, double J3, double m3) {

    // ( J1 J2 J3 ) = (-1)^(J1-J2 - m3)/ sqrt(2*J3+1) * CGcoeff(J3, -m3, J1, m1, J2, m2) 
    // ( m1 m2 m3 )

    return pow(-1, J1 - J2 - m3) / sqrt(2 * J3 + 1) * CGcoeff(J3, -m3, J1, m1, J2, m2);

}

double SixJSymbol(double J1, double J2, double J3, double J4, double J5, double J6) {

    // coupling of j1, j2, j3 to J-J1
    // J1 = j1
    // J2 = j2
    // J3 = j12 = j1 + j2
    // J4 = j3
    // J5 = J = j1 + j2 + j3
    // J6 = j23 = j2 + j3

    //check triangle condition
    if (J3 < abs(J1 - J2) || J1 + J2 < J3) return 0;
    if (J6 < abs(J2 - J4) || J2 + J4 < J6) return 0;
    if (J5 < abs(J1 - J6) || J1 + J6 < J5) return 0;
    if (J5 < abs(J3 - J4) || J3 + J4 < J5) return 0;

    double sixJ = 0;

    for (double m1 = -J1; m1 <= J1; m1 = m1 + 1) {
        for (double m2 = -J2; m2 <= J2; m2 = m2 + 1) {
            for (double m3 = -J3; m3 <= J3; m3 = m3 + 1) {
                for (double m4 = -J4; m4 <= J4; m4 = m4 + 1) {
                    for (double m5 = -J5; m5 <= J5; m5 = m5 + 1) {
                        for (double m6 = -J6; m6 <= J6; m6 = m6 + 1) {

                            double f = (J1 - m1) + (J2 - m2) + (J3 - m3) + (J4 - m4) + (J5 - m5) + (J6 - m6);

                            double a1 = ThreeJSymbol(J1, -m1, J2, -m2, J3, -m3); // J3 = j12 
                            double a2 = ThreeJSymbol(J1, m1, J5, -m5, J6, m6); // J5 = j1 + (J6 = j23)
                            double a3 = ThreeJSymbol(J4, m4, J2, m2, J6, -m6); // J6 = j23
                            double a4 = ThreeJSymbol(J4, -m4, J5, m5, J3, m3); // J5 = j3 + j12

                            double a = a1 * a2 * a3 * a4;
                            //if( a != 0 ) printf("%4.1f %4.1f %4.1f %4.1f %4.1f %4.1f | %f \n", m1, m2, m3, m4, m5, m6, a);

                            sixJ += pow(-1, f) * a1 * a2 * a3 * a4;

                        }
                    }
                }
            }
        }
    }

    return sixJ;
}

double NineJSymbol(double J1, double J2, double J3, double J4, double J5, double J6, double J7, double J8, double J9) {

    double gMin = min(min(min(abs(J1 - J2), abs(J4 - J5)), abs(J4 - J6)), abs(J7 - J8));
    double gMax = max(max(max(J1 + J2, J4 + J5), J3 + J6), J7 + J8);

    //printf(" gMin, gMax = %f %f \n", gMin, gMax);

    double nineJ = 0;
    for (double g = gMin; g <= gMax; g = g + 1) {
        double f = pow(-1, 2 * g) * (2 * g + 1);
        double s1 = SixJSymbol(J1, J4, J7, J8, J9, g);
        if (s1 == 0) continue;
        double s2 = SixJSymbol(J2, J5, J8, J4, g, J6);
        if (s2 == 0) continue;
        double s3 = SixJSymbol(J3, J6, J9, g, J1, J2);
        if (s3 == 0) continue;
        nineJ += f * s1 * s2 * s3;
    }

    return nineJ;
}

double j_dipole(double a, double freq_0, double freq_1)
{
    return sqrt(3 * a * pow(sc::c, 2) / (2 * sc::pi * sc::h * pow( abs(freq_0 - freq_1), 3))) * 1e-12;
}

double a_dipole(double i, double j_l, double f_l, double m_l, double j_u, double f_u, double m_u, double q)
{
    if (m_u - m_l != q) return 0.;
    double sqrt_f = sqrt(2 * f_l + 1);
    double sqrt_j = sqrt(2 * j_u + 1);
    double exp = f_l + i + 1 + j_u;
    return pow(-1, exp) * sqrt_f * sqrt_j * SixJSymbol(j_u, j_l, 1, f_l, f_u, i) * CGcoeff(f_u, m_u, f_l, m_l, 1, q);
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
    double shift = hyper_const[0] * k_0 / 2;
    if (i > 0.5 && j > 0.5)
    {
        double k_1 = 3 * k_0 * (k_0 + 1) / 2 - 2 * i * (i + 1) * j * (j + 1);
        k_1 /= i * (2 * i - 1) * j * (2 * j - 1);
        shift += hyper_const[1] * k_1 / 4;
    }
    if (i > 1 && j > 1)
    {
        shift += 0;  // 3. order
    }
    return shift;
}

double zeeman(double m, double b, double g)
{
    return -g * m * sc::mu_B * b / sc::h * 1e-6;
}

double hyper_zeeman(double i, double s, double l, double j,
    double f, double m, double g_n, double* hyper_const, double b, bool g_n_as_gyro)
{
    double g_j = lande_j(s, l, j);
    double _g_n = g_n;
    if (g_n_as_gyro) _g_n = lande_n(g_n);
    double g_f = 0.;
    if (i != 0 || j != 0) g_f = lande_f(i, j, f, _g_n, g_j);
    double ret = hyperfine(i, j, f, hyper_const);
    ret += zeeman(m, b, g_f);
    return ret;
}

double lorentz(double w, double w0, double a, double rabi_square)
{
    // if (a == 0) return 0;
    return rabi_square * a / (std::pow(w - w0, 2) + std::pow(a, 2) / 4) / 12;
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
    return sc::h * std::pow(freq, 2) / (2 * mass * sc::amu * std::pow(sc::c, 2)) * 1e12;
}
