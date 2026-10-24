#pragma once

// Angular momentum coupling coefficients with the same argument order as in qspec.algebra.
// Both functions return 0 if the arguments violate a selection rule or are not multiples of 1/2.

double clebsch_gordan(double j1, double j2, double j3, double m1, double m2, double m3);
double wigner_6j(double j1, double j2, double j3, double j4, double j5, double j6);
