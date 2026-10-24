/*
* qspec.Algebra.cpp
*
* Clebsch-Gordan coefficients and Wigner-6j symbols from the explicit Racah formulas.
* All factorials are combined into a single exp() of tabulated log-factorials, which avoids overflows.
* The remaining terms of the alternating sums follow from the first one through the ratio of consecutive terms.
*/

#include "Algebra.h"

#include <algorithm>
#include <array>
#include <cmath>


namespace
{
    constexpr int LOG_FACTORIAL_SIZE = 1024;

    const std::array<double, LOG_FACTORIAL_SIZE> log_factorials = []
    {
        std::array<double, LOG_FACTORIAL_SIZE> table{};
        for (int n = 0; n < LOG_FACTORIAL_SIZE; ++n) table[n] = std::lgamma(n + 1.);
        return table;
    }();

    inline double log_factorial(int n)
    {
        return n < LOG_FACTORIAL_SIZE ? log_factorials[n] : std::lgamma(n + 1.);
    }

    // Returns 2 * x as an integer. Sets valid to false if x is not a multiple of 1/2.
    inline int twice(double x, bool& valid)
    {
        double tx = std::round(2. * x);
        valid &= std::abs(2. * x - tx) < 1e-9;
        return static_cast<int>(tx);
    }

    // |a - b| <= c <= a + b and a + b + c is an integer, for the doubled quantum numbers a, b and c.
    // This is never satisfied if one of them is negative.
    inline bool triangle(int a, int b, int c)
    {
        return std::abs(a - b) <= c && c <= a + b && (a + b + c) % 2 == 0;
    }

    // ln[(a + b - c)! (a - b + c)! (-a + b + c)! / (a + b + c + 1)!], for the doubled quantum numbers a, b and c.
    inline double log_triangle_coefficient(int a, int b, int c)
    {
        return log_factorial((a + b - c) / 2) + log_factorial((a - b + c) / 2) + log_factorial((b + c - a) / 2)
            - log_factorial((a + b + c) / 2 + 1);
    }
}


double clebsch_gordan(double j1, double j2, double j3, double m1, double m2, double m3)
{
    bool valid = true;
    int tj1 = twice(j1, valid), tj2 = twice(j2, valid), tj3 = twice(j3, valid);
    int tm1 = twice(m1, valid), tm2 = twice(m2, valid), tm3 = twice(m3, valid);

    if (!valid || tm1 + tm2 != tm3 || !triangle(tj1, tj2, tj3)) return 0.;
    if (std::abs(tm1) > tj1 || std::abs(tm2) > tj2 || std::abs(tm3) > tj3) return 0.;
    if ((tj1 + tm1) % 2 != 0 || (tj2 + tm2) % 2 != 0) return 0.;

    // sum_k (-1)^k / [k! (n1 - k)! (n2 - k)! (n3 - k)! (n4 + k)! (n5 + k)!]
    int n1 = (tj1 + tj2 - tj3) / 2;  // j1 + j2 - j3
    int n2 = (tj1 - tm1) / 2;  // j1 - m1
    int n3 = (tj2 + tm2) / 2;  // j2 + m2
    int n4 = (tj3 - tj2 + tm1) / 2;  // j3 - j2 + m1
    int n5 = (tj3 - tj1 - tm2) / 2;  // j3 - j1 - m2
    int k_min = std::max({0, -n4, -n5});
    int k_max = std::min({n1, n2, n3});
    if (k_min > k_max) return 0.;

    double log_prefactor = 0.5 * (log_triangle_coefficient(tj1, tj2, tj3)
        + log_factorial((tj1 + tm1) / 2) + log_factorial((tj1 - tm1) / 2)
        + log_factorial((tj2 + tm2) / 2) + log_factorial((tj2 - tm2) / 2)
        + log_factorial((tj3 + tm3) / 2) + log_factorial((tj3 - tm3) / 2));
    double log_first = log_factorial(k_min) + log_factorial(n1 - k_min) + log_factorial(n2 - k_min)
        + log_factorial(n3 - k_min) + log_factorial(n4 + k_min) + log_factorial(n5 + k_min);

    double term = 1., sum = 1.;
    for (double k = k_min; k < k_max; ++k)
    {
        term *= -(n1 - k) * (n2 - k) * (n3 - k) / ((k + 1.) * (n4 + k + 1.) * (n5 + k + 1.));
        sum += term;
    }
    double sign = k_min % 2 == 0 ? 1. : -1.;
    return sign * std::sqrt(tj3 + 1.) * std::exp(log_prefactor - log_first) * sum;
}


double wigner_6j(double j1, double j2, double j3, double j4, double j5, double j6)
{
    bool valid = true;
    int tj1 = twice(j1, valid), tj2 = twice(j2, valid), tj3 = twice(j3, valid);
    int tj4 = twice(j4, valid), tj5 = twice(j5, valid), tj6 = twice(j6, valid);

    if (!valid || !triangle(tj1, tj2, tj3) || !triangle(tj1, tj5, tj6)
        || !triangle(tj4, tj2, tj6) || !triangle(tj4, tj5, tj3)) return 0.;

    // sum_t (-1)^t (t + 1)! / [(t - a1)! (t - a2)! (t - a3)! (t - a4)! (b1 - t)! (b2 - t)! (b3 - t)!]
    int a1 = (tj1 + tj2 + tj3) / 2;
    int a2 = (tj1 + tj5 + tj6) / 2;
    int a3 = (tj4 + tj2 + tj6) / 2;
    int a4 = (tj4 + tj5 + tj3) / 2;
    int b1 = (tj1 + tj2 + tj4 + tj5) / 2;
    int b2 = (tj2 + tj3 + tj5 + tj6) / 2;
    int b3 = (tj3 + tj1 + tj6 + tj4) / 2;
    int t_min = std::max({a1, a2, a3, a4});
    int t_max = std::min({b1, b2, b3});
    if (t_min > t_max) return 0.;

    double log_prefactor = 0.5 * (log_triangle_coefficient(tj1, tj2, tj3) + log_triangle_coefficient(tj1, tj5, tj6)
        + log_triangle_coefficient(tj4, tj2, tj6) + log_triangle_coefficient(tj4, tj5, tj3));
    double log_first = log_factorial(t_min + 1)
        - log_factorial(t_min - a1) - log_factorial(t_min - a2) - log_factorial(t_min - a3) - log_factorial(t_min - a4)
        - log_factorial(b1 - t_min) - log_factorial(b2 - t_min) - log_factorial(b3 - t_min);

    double term = 1., sum = 1.;
    for (double t = t_min; t < t_max; ++t)
    {
        term *= -(t + 2.) * (b1 - t) * (b2 - t) * (b3 - t)
            / ((t + 1. - a1) * (t + 1. - a2) * (t + 1. - a3) * (t + 1. - a4));
        sum += term;
    }
    double sign = t_min % 2 == 0 ? 1. : -1.;
    return sign * std::exp(log_prefactor + log_first) * sum;
}
