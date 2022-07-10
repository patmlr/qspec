
#include "pch.h"
#include "Utility.h"


std::vector<VectorXd> cast_delta(double delta[], size_t delta_size, size_t lasers_size)
{
    std::vector<VectorXd> _delta = std::vector<VectorXd>(delta_size, VectorXd::Zero(lasers_size));
    for (size_t i = 0; i < delta_size; ++i)
    {
        for (size_t m = 0; m < lasers_size; ++m) _delta.at(i)(m) = delta[lasers_size * i + m];
    }
    return _delta;
}


std::vector<Vector3d> cast_v(double delta[], size_t v_size)
{
    std::vector<Vector3d> _v = std::vector<Vector3d>(v_size, Vector3d::Zero());
    for (size_t i = 0; i < v_size; ++i)
    {
        for (size_t r = 0; r < 3; ++r) _v.at(i)(r) = delta[3 * i + r];
    }
    return _v;
}


VectorXd cast_y0_vectord(double y0[], size_t size)
{
    VectorXd _y0(size);
    _y0 = VectorXd::Zero(size);
    for (size_t i = 0; i < size; ++i) _y0(i) = y0[i];
    return _y0;
}

VectorXcd cast_y0_vectorcd(std::complex<double> y0[], size_t size)
{
    VectorXcd _y0(size);
    _y0 = VectorXcd::Zero(size);
    for (size_t i = 0; i < size; ++i) _y0(i) = y0[i];
    return _y0;
}

MatrixXcd cast_y0_matrixcd(std::complex<double> y0[], size_t size)
{
    MatrixXcd _y0(size, size);
    _y0 = MatrixXcd::Zero(size, size);
    for (size_t j = 0; j < size; ++j)
    {
        for (size_t i = 0; i < size; ++i) _y0(i, j) = y0[size * j + i];
    }
    return _y0;
}

std::vector<VectorXcd> cast_y0_vector_vectorcd(std::complex<double> y0[], size_t y0_size, size_t size)
{
    std::vector<VectorXcd> _y0 = std::vector<VectorXcd>(y0_size, VectorXcd::Zero(size));
    for (size_t i = 0; i < y0_size; ++i)
    {
        for (size_t j = 0; j < size; ++j) _y0.at(i)(j) = y0[size * i + j];
    }
    return _y0;
}

size_t gen_index(VectorXd p, std::uniform_real_distribution<double>& d, std::mt19937& gen)
{
    double r = d(gen);
    double sum = 0;
    for (size_t i = 0; i < p.size(); ++i)
    {
        sum += p(i);
        if (r < sum) return i;
    }
    return 0;
}

bool check_loop(size_t i, size_t j, size_t m, int pm, std::vector<MatrixXd>& shifts)
{
    VectorXd _shift;
    _shift = shifts.at(i).row(m) - shifts.at(j).row(m);
    _shift(m) += pm;
    for (size_t _m = 0; _m < _shift.size(); ++_m)
    {
        if (_shift(_m) != 0) return true; // printf("m=%zi: %1.1f\n", _m, _shift(_m));
    }
    return false;
}
