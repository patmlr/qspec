
#include "pch.h"
#include "Utility.h"



Vector3d cast_Vector3d(double* x)
{
    Vector3d _x = Vector3d::Zero();
    for (size_t i = 0; i < 3; ++i) _x(i) = x[i];
    return _x;
}

VectorXd cast_VectorXd(double* x, size_t size)
{
    VectorXd _y0(size);
    _y0 = VectorXd::Zero(size);
    for (size_t i = 0; i < size; ++i) _y0(i) = x[i];
    return _y0;
}

VectorXcd cast_VectorXcd(std::complex<double>* x, size_t size)
{
    VectorXcd _y0(size);
    _y0 = VectorXcd::Zero(size);
    for (size_t i = 0; i < size; ++i) _y0(i) = x[i];
    return _y0;
}

MatrixXd cast_MatrixXd(double* x, size_t size)
{
    MatrixXd _y0(size, size);
    _y0 = MatrixXd::Zero(size, size);
    for (size_t j = 0; j < size; ++j)
    {
        for (size_t i = 0; i < size; ++i) _y0(i, j) = x[size * j + i];
    }
    return _y0;
}

MatrixXcd cast_MatrixXcd(std::complex<double>* x, size_t size)
{
    MatrixXcd _y0(size, size);
    _y0 = MatrixXcd::Zero(size, size);
    for (size_t j = 0; j < size; ++j)
    {
        for (size_t i = 0; i < size; ++i) _y0(i, j) = x[size * j + i];
    }
    return _y0;
}

std::vector<double> cast_samples_double(double* x, size_t sample_size)
{
    std::vector<double> _x = std::vector<double>(sample_size);
    for (size_t i = 0; i < sample_size; ++i) _x.at(i) = x[i];
    return _x;
}


std::vector<Vector3d> cast_samples_Vector3d(double* x, size_t sample_size)
{
    std::vector<Vector3d> _x = std::vector<Vector3d>(sample_size, Vector3d::Zero());
    for (size_t i = 0; i < sample_size; ++i)
    {
        for (size_t r = 0; r < 3; ++r) _x.at(i)(r) = x[3 * i + r];
    }
    return _x;
}

std::vector<VectorXd> cast_samples_VectorXd(double* x, size_t sample_size, size_t size)
{
    std::vector<VectorXd> _x = std::vector<VectorXd>(sample_size, VectorXd::Zero(size));
    for (size_t i = 0; i < sample_size; ++i)
    {
        for (size_t m = 0; m < size; ++m) _x.at(i)(m) = x[size * i + m];
    }
    return _x;
}

std::vector<VectorXcd> cast_samples_VectorXcd(std::complex<double>* x, size_t sample_size, size_t size)
{
    std::vector<VectorXcd> _x = std::vector<VectorXcd>(sample_size, VectorXcd::Zero(size));
    for (size_t i = 0; i < sample_size; ++i)
    {
        for (size_t m = 0; m < size; ++m) _x.at(i)(m) = x[size * i + m];
    }
    return _x;
}

std::vector<MatrixXd> cast_samples_MatrixXd(double* x, size_t sample_size, size_t size)
{
    std::vector<MatrixXd> _x = std::vector<MatrixXd>(sample_size, MatrixXd::Zero(size, size));
    for (size_t i = 0; i < sample_size; ++i)
    {
        for (size_t m = 0; m < size; ++m)
        {
            for (size_t n = 0; n < size; ++n) _x.at(i)(n, m) = x[size * size * i + size * m + n];
        }
    }
    return _x;
}

std::vector<MatrixXcd> cast_samples_MatrixXcd(std::complex<double>* x, size_t sample_size, size_t size)
{
    std::vector<MatrixXcd> _x = std::vector<MatrixXcd>(sample_size, MatrixXcd::Zero(size, size));
    for (size_t i = 0; i < sample_size; ++i)
    {
        for (size_t m = 0; m < size; ++m)
        {
            for (size_t n = 0; n < size; ++n) _x.at(i)(n, m) = x[size * size * i + size * m + n];
        }
    }
    return _x;
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
