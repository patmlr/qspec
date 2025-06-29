#pragma once

#include <complex>
#include <numeric>
#include <algorithm>
#include <vector>
#include <random>
#include <Eigen/Dense>

using namespace Eigen;

template <typename T> int sgn(T val)
{
    return (T(0) < val) - (val < T(0));
}

Vector3d cast_Vector3d(double* x);
VectorXd cast_VectorXd(double* x, size_t size);
VectorXcd cast_VectorXcd(std::complex<double>* x, size_t size);
MatrixXd cast_MatrixXd(double* x, size_t size);
MatrixXcd cast_MatrixXcd(std::complex<double>* x, size_t size);

std::vector<double> cast_samples_double(double* x, size_t sample_size);
std::vector<Vector3d> cast_samples_Vector3d(double* x, size_t sample_size);
std::vector<VectorXd> cast_samples_VectorXd(double* x, size_t sample_size, size_t size);
std::vector<VectorXcd> cast_samples_VectorXcd(std::complex<double>* x, size_t sample_size, size_t size);
std::vector<MatrixXd> cast_samples_MatrixXd(double* x, size_t sample_size, size_t size);
std::vector<MatrixXcd> cast_samples_MatrixXcd(std::complex<double>* x, size_t sample_size, size_t size);

size_t gen_index(VectorXd p, std::uniform_real_distribution<double>& d, std::mt19937& gen);
bool check_loop(size_t i, size_t j, size_t m, int pm, std::vector<MatrixXd>& shifts);

std::vector<size_t> argsort(const std::vector<double>& array);
std::vector<size_t> invert_order(const std::vector<size_t>& indexes);
