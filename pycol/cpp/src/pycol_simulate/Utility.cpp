
#include "pch.h"
#include "Utility.h"


std::vector<VectorXd> cast_delta(double* delta, size_t delta_size, size_t lasers_size)
{
    std::vector<VectorXd> _delta = std::vector<VectorXd>(delta_size, VectorXd::Zero(lasers_size));
    for (size_t i = 0; i < delta_size; ++i)
    {
        for (size_t m = 0; m < lasers_size; ++m) _delta.at(i)(m) = delta[lasers_size * i + m];
    }
    return _delta;
}


std::vector<Vector3d> cast_v(double* delta, size_t v_size)
{
    std::vector<Vector3d> _v = std::vector<Vector3d>(v_size, Vector3d::Zero());
    for (size_t i = 0; i < v_size; ++i)
    {
        for (size_t r = 0; r < 3; ++r) _v.at(i)(r) = delta[3 * i + r];
    }
    return _v;
}


VectorXd cast_y0_vectord(double* y0, size_t size)
{
    VectorXd _y0(size);
    _y0 = VectorXd::Zero(size);
    for (size_t i = 0; i < size; ++i) _y0(i) = y0[i];
    return _y0;
}

VectorXcd cast_y0_vectorcd(std::complex<double>* y0, size_t size)
{
    VectorXcd _y0(size);
    _y0 = VectorXcd::Zero(size);
    for (size_t i = 0; i < size; ++i) _y0(i) = y0[i];
    return _y0;
}

MatrixXcd cast_y0_matrixcd(std::complex<double>* y0, size_t size)
{
    MatrixXcd _y0(size, size);
    _y0 = MatrixXcd::Zero(size, size);
    for (size_t j = 0; j < size; ++j)
    {
        for (size_t i = 0; i < size; ++i) _y0(i, j) = y0[size * j + i];
    }
    return _y0;
}
