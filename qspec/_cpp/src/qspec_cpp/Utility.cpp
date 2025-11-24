
#include "pch.h"
#include "Utility.h"

extern double sc::pi = 3.14159265359;

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

std::vector<size_t> cast_samples_size_t(size_t* x, size_t sample_size)
{
    std::vector<size_t> _x = std::vector<size_t>(sample_size);
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

std::vector<Vector3cd> cast_samples_Vector3cd(std::complex<double>* x, size_t sample_size)
{
    std::vector<Vector3cd> _x = std::vector<Vector3cd>(sample_size, Vector3cd::Zero());
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

std::vector<MatrixXcd> cast_samples_VectorXcd_as_MatrixXcd(std::complex<double>* x, size_t sample_size, size_t size)
{
    std::vector<MatrixXcd> _x = std::vector<MatrixXcd>(sample_size, MatrixXcd::Zero(size, 1));
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
            for (size_t n = 0; n < size; ++n)
            {
                _x.at(i)(n, m) = x[size * size * i + size * m + n];
                // if (i == 100 && n == m) printf("rho(100): %s\n", std::format("({}, {}): {:.3e} + {:.3e}i", m, n, _x.at(i)(n, m).real(), _x.at(i)(n, m).imag()).c_str());
                // if (i == 301 && n == m) printf("rho(301): %s\n", std::format("({}, {}): {:.3e} + {:.3e}i", m, n, _x.at(i)(n, m).real(), _x.at(i)(n, m).imag()).c_str());
            }
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

std::vector<size_t> argsort(const std::vector<double>& array)
{
    std::vector<size_t> indexes(array.size());
    std::iota(indexes.begin(), indexes.end(), 0);
    std::sort(indexes.begin(), indexes.end(),
        [&](size_t left, size_t right) -> bool {
            return array[left] < array[right];
        });
    return indexes;
}

std::vector<size_t> invert_order(const std::vector<size_t>& indexes)
{
    std::vector<size_t> inverted;
    for (size_t k = 0; k < indexes.size(); ++k)
    {
        auto it = std::find(indexes.begin(), indexes.end(), k);
        size_t index = std::distance(indexes.begin(), it);
        inverted.push_back(index);
    }
    return inverted;
}

VectorXd gen_unit_vector(size_t dim, size_t index)
{
    VectorXd u = VectorXd::Zero(dim);
    u(index) = 1.;
    return u;
}

double rotation_theta(Vector3d vec)
{
    if (vec.norm() == 0.) return 0.;
    return acos(vec(2) / vec.norm());
}

double rotation_phi(Vector3d vec)
{
    if (vec.norm() == 0.) return 0.;

    Vector3d _vec = vec / vec.norm();
    // printf("vec: %.3f, %.3f, %.3f\n", _vec(0), _vec(1), _vec(2));
    double vec2 = pow(_vec(2), 2);
    double phi = 0.;
    // printf("arg: %.3f\n", _vec(0) / sqrt(1. - vec2));
    if (vec2 < 1.)
    {
        if (abs(_vec(0)) > abs(_vec(1)))
        {
            if (_vec(0) < 0.) phi = sc::pi - asin(_vec(1) / sqrt(1. - vec2));
            else phi = asin(_vec(1) / sqrt(1. - vec2));
        }
        else
        {
            if (_vec(1) < 0.) phi = 2 * sc::pi - acos(_vec(0) / sqrt(1. - vec2));
            else phi = acos(_vec(0) / sqrt(1. - vec2));
        }
    }
    return phi;
}

Matrix3d rotation_matrix(Vector3d vec)
{
    double theta = rotation_theta(vec);
    double phi = rotation_phi(vec);

    // printf("theta, phi: %.3f pi, %.3f pi\n", theta / sc::pi, phi / sc::pi);

    Matrix3d R = Matrix3d::Identity();

    R(0, 0) = cos(theta) * cos(phi);
    R(1, 0) = cos(theta) * sin(phi);
    R(2, 0) = -sin(theta);

    R(0, 1) = -sin(phi);
    R(1, 1) = cos(phi);

    R(0, 2) = sin(theta) * cos(phi);
    R(1, 2) = sin(theta) * sin(phi);
    R(2, 2) = cos(theta);

    return R;
}

Vector3d cast_theta_phi_er(double theta, double phi)
{
    Vector3d r = Vector3d::Zero();
    r(0) = sin(theta) * cos(phi);
    r(1) = sin(theta) * sin(phi);
    r(2) = cos(theta);
    return r / r.norm();
}

std::vector<Vector3d> cast_samples_theta_phi_er(double* theta, double* phi, size_t sample_size)
{
    // printf("theta, phi: %.3f, %.3f\n", theta[0], phi[0]);
    std::vector<Vector3d> r(sample_size, Vector3d::Zero());
    for (size_t i = 0; i < sample_size; ++i)
    {
        r.at(i) = cast_theta_phi_er(theta[i], phi[i]);
    }
    // printf("r: %.3f, %.3f, %.3f\n", r.at(0)(0), r.at(0)(1), r.at(0)(2));
    return r;
}

Vector3d cast_theta_phi_et(double theta, double phi)
{
    Vector3d r = Vector3d::Zero();
    r(0) = cos(theta) * cos(phi);
    r(1) = cos(theta) * sin(phi);
    r(2) = -sin(theta);
    return r / r.norm();
}

std::vector<Vector3d> cast_samples_theta_phi_et(double* theta, double* phi, size_t sample_size)
{
    // printf("theta, phi: %.3f, %.3f\n", theta[0], phi[0]);
    std::vector<Vector3d> r(sample_size, Vector3d::Zero());
    for (size_t i = 0; i < sample_size; ++i)
    {
        r.at(i) = cast_theta_phi_et(theta[i], phi[i]);
    }
    // printf("r: %.3f, %.3f, %.3f\n", r.at(0)(0), r.at(0)(1), r.at(0)(2));
    return r;
}

Vector3d cast_theta_phi_ep(double theta, double phi)
{
    Vector3d r = Vector3d::Zero();
    r(0) = -sin(phi);
    r(1) = cos(phi);
    return r / r.norm();
}

std::vector<Vector3d> cast_samples_theta_phi_ep(double* theta, double* phi, size_t sample_size)
{
    // printf("theta, phi: %.3f, %.3f\n", theta[0], phi[0]);
    std::vector<Vector3d> r(sample_size, Vector3d::Zero());
    for (size_t i = 0; i < sample_size; ++i)
    {
        r.at(i) = cast_theta_phi_ep(theta[i], phi[i]);
    }
    // printf("r: %.3f, %.3f, %.3f\n", r.at(0)(0), r.at(0)(1), r.at(0)(2));
    return r;
}

std::string half_int_to_str(double value)
{
    std::string ret = std::to_string(value);
    std::string suf = "";

    if (ret.find(".5") != std::string::npos)
    {
        ret = std::to_string(2 * value);
        suf = "/2";
    }
    size_t index = ret.find(".");
    return ret.substr(0, index) + suf;
}
