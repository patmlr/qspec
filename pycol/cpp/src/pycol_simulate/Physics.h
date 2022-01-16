#pragma once

// #define NOMINMAX

#include <stdlib.h>
#include <complex>
#include <cmath>
#include <algorithm>

namespace sc
{
extern std::complex<double> i;
extern double pi;
extern double h;
extern double c;
extern double e;
extern double amu;
extern double epsilon_0;
extern double g_s;
extern double mu_B;
extern double mu_N;
}

double factorial(double n);
double CGcoeff(double J, double m, double J1, double m1, double J2, double m2);
double ThreeJSymbol(double J1, double m1, double J2, double m2, double J3, double m3);
double SixJSymbol(double J1, double J2, double J3, double J4, double J5, double J6);
double NineJSymbol(double J1, double J2, double J3, double J4, double J5, double J6, double J7, double J8, double J9);

double j_dipole(double a, double freq_0, double freq_1);
double a_dipole(double i, double j_l, double f_l, double m_l, double j_u, double f_u, double m_u, double q);

double lande_n(double g_n);
double lande_j(double s, double l, double j);
double lande_f(double i, double j, double f, double g_n, double g_j);
double hyperfine(double i, double j, double f, double* hyper_const);
double zeeman(double m, double b, double g);
double hyper_zeeman(double i, double s, double ll, double j,
	double f, double m, double g_n, double* hyper_const, double b, bool g_n_as_gyro);
double lorentz(double w, double w0, double a, double rabi_square);
double gamma(double v);
double doppler(double x, double v, double angle);
