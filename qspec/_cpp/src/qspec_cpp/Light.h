#pragma once

#include <complex>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Geometry>

using namespace Eigen;
typedef std::complex<double> dcomp;

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


class Laser
{
protected:
	double freq;
	double intensity;
	double gamma;
	Polarization* polarization;
	int index = 0;
	Vector3d k;

public:
	Laser();
	void init(double _freq, double _intensity, double _gamma, Polarization* _polarization, Vector3d _k);
	double get_detuned(const Vector3d& v);
	double get_detuned(double delta, const Vector3d& v);

	double get_freq();
	void set_freq(double _freq_0);

	double get_intensity();
	void set_intensity(double _intensity);

	double get_gamma();
	void set_gamma(double _gamma);

	Polarization* get_polarization();
	void set_polarization(Polarization* _polarization);

	Vector3d get_k();
	void set_k(Vector3d _k);
	Vector3d get_kn();

};
