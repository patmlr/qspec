#pragma once

#include <format>
#include <complex>
#include <numeric>
#include <algorithm>
#include <vector>
#include <random>
#include <Eigen/Dense>

using namespace Eigen;

namespace sc
{
extern double pi;
}

template <typename T> int sgn(T val)
{
    return (T(0) < val) - (val < T(0));
}

template <typename T>
T factorial(T n) {
    if (n < T(0)) return T(0);
    return (n == T(0) || n == T(1)) ? T(1) : factorial(n - T(1)) * n;
}

template <typename T>
T double_factorial(T n) {
    if (n < T(0)) return T(0);
    if (n == T(0) || n == T(1)) return T(1);
    return n * double_factorial(n - T(2));
}

Vector3d cast_Vector3d(double* x);
VectorXd cast_VectorXd(double* x, size_t size);
VectorXcd cast_VectorXcd(std::complex<double>* x, size_t size);
MatrixXd cast_MatrixXd(double* x, size_t size);
MatrixXcd cast_MatrixXcd(std::complex<double>* x, size_t size);

std::vector<double> cast_samples_double(double* x, size_t sample_size);
std::vector<size_t> cast_samples_size_t(size_t* x, size_t sample_size);
std::vector<Vector3d> cast_samples_Vector3d(double* x, size_t sample_size);
std::vector<Vector3cd> cast_samples_Vector3cd(std::complex<double>* x, size_t sample_size);
std::vector<VectorXd> cast_samples_VectorXd(double* x, size_t sample_size, size_t size);
std::vector<VectorXcd> cast_samples_VectorXcd(std::complex<double>* x, size_t sample_size, size_t size);
std::vector<MatrixXcd> cast_samples_VectorXcd_as_MatrixXcd(std::complex<double>* x, size_t sample_size, size_t size);
std::vector<MatrixXd> cast_samples_MatrixXd(double* x, size_t sample_size, size_t size);
std::vector<MatrixXcd> cast_samples_MatrixXcd(std::complex<double>* x, size_t sample_size, size_t size);

size_t gen_index(VectorXd p, std::uniform_real_distribution<double>& d, std::mt19937& gen);
bool check_loop(size_t i, size_t j, size_t m, int pm, std::vector<MatrixXd>& shifts);

std::vector<size_t> argsort(const std::vector<double>& array);
std::vector<size_t> invert_order(const std::vector<size_t>& indexes);

double rotation_theta(Vector3d vec);
double rotation_phi(Vector3d vec);
Matrix3d rotation_matrix(Vector3d vec);

Vector3d cast_theta_phi_er(double theta, double phi);
std::vector<Vector3d> cast_samples_theta_phi_er(double* theta, double* phi, size_t sample_size);

Vector3d cast_theta_phi_et(double theta, double phi);
std::vector<Vector3d> cast_samples_theta_phi_et(double* theta, double* phi, size_t sample_size);

Vector3d cast_theta_phi_ep(double theta, double phi);
std::vector<Vector3d> cast_samples_theta_phi_ep(double* theta, double* phi, size_t sample_size);
