#pragma once

#include <complex>
#include <vector>
#include <random>
#include <Eigen/Dense>

using namespace Eigen;


std::vector<VectorXd> cast_delta(double* delta, size_t delta_size, size_t lasers_size);
std::vector<Vector3d> cast_v(double* delta, size_t v_size);
VectorXd cast_y0_vectord(double* y0, size_t size);
VectorXcd cast_y0_vectorcd(std::complex<double>* y0, size_t size);
MatrixXcd cast_y0_matrixcd(std::complex<double>* y0, size_t size);
std::vector<VectorXcd> cast_y0_vector_vectorcd(std::complex<double>* y0, size_t y0_size, size_t size);
size_t gen_index(VectorXd p, std::uniform_real_distribution<double>& d, std::mt19937& gen);
bool check_loop(size_t i, size_t j, size_t m, int pm, std::vector<MatrixXd>& shifts);
