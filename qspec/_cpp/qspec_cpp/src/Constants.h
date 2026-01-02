#pragma once

#include <complex>

using namespace std::complex_literals;

namespace sc {
    inline constexpr std::complex<double> i = 1i;
    inline constexpr double pi = 3.141592653589793;
    inline constexpr double h = 6.62607015e-34;
    inline constexpr double hbar = 1.0545718176461565e-34;
    inline constexpr double c = 299792458.0;
    inline constexpr double e = 1.602176634e-19;
    inline constexpr double amu = 1.66053906892e-27;
    inline constexpr double epsilon_0 = 8.8541878188e-12;
    inline constexpr double g_s = -2.00231930436092;
    inline constexpr double mu_B = 9.2740100657e-24;
    inline constexpr double mu_N = 5.0507837393e-27;
}
