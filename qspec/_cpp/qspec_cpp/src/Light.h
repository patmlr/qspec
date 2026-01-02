#pragma once

#include <stdexcept>
#include <complex>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Geometry>

using namespace Eigen;
typedef std::complex<double> dcomp;

class polarization_error : public std::runtime_error {
public:
	polarization_error(const std::string& message) : std::runtime_error(message) {}
};

class Polarization
{
protected:
	Matrix3cd T;
	Quaterniond R;
	Quaterniond Rq;

	Vector3d q_axis;
	Vector3cd x;
	Vector3cd q;
	Vector3d Z;

public:
	Polarization();
	void init(Vector3cd vec, Vector3d _q_axis, bool vec_as_q);

	void calc_R(Vector3d _q_axis);
	void infer_x();
	void infer_q();
	void def_q_axis(Vector3d _q_axis, bool q_fixed);

	Vector3cd* get_x();
	Vector3cd* get_q();
	Vector3d* get_q_axis();
};

class Polarizationk
{
protected:
	Vector3d Z;
	Matrix3cd T;
	Matrix3d Rz;

	double theta_k;
	double phi_k;
	Matrix3d Rk;

	Vector3d k;
	Vector3d q_axis;

	Vector3cd x;
	Vector3cd qk;

public:
	Polarizationk();
	Polarizationk(Vector3d _q_axis);
	void init(Vector3cd _x, Vector3d _k, Vector3d _q_axis);
	void init_qk(Vector3cd _x, Vector3d _k);

	void infer_qk();
	void def_q_axis(Vector3d _q_axis);


	double get_theta_k();
	double get_phi_k();
	Vector3cd* get_x();
	Vector3cd* get_qk();
	Vector3d* get_q_axis();
};


class Laser
{
protected:
	double freq;
	double intensity;
	Polarization* polarization;
	int index = 0;
	Vector3d k;

public:
	Laser();
	void init(double _freq, double _intensity, Polarization* _polarization, Vector3d _k);
	double get_detuned(const Vector3d& v);
	double get_detuned(double delta, const Vector3d& v);

	double get_intensity();
	void set_intensity(double _intensity);

	Polarization* get_polarization();
	void set_polarization(Polarization* _polarization);

	double get_freq();
	void set_freq(double _freq_0);

	Vector3d* get_k();
	void set_k(Vector3d _k);

	Vector3d get_k_si();
	VectorXcd get_kpol(bool electric, size_t k, Vector3d q_axis);
};
