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

	Vector3d q_axis;
	Vector3cd x;
	Vector3cd q;

public:
	Polarization();
	void init(Vector3cd vec, Vector3d _q_axis, bool vec_as_q);

	void calc_R();
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
	Polarization* polarization;
	int index = 0;
	Vector3d k;

public:
	Laser();
	void init(double _freq, double _intensity, Polarization* _polarization);
	double get_detuned(const Vector3d& v);
	double get_detuned(double delta, const Vector3d& v);

	double get_intensity();
	void set_intensity(double _intensity);

	Polarization* get_polarization();
	void set_polarization(Polarization* _polarization);

	double get_freq();
	void set_freq(double _freq_0);

	Vector3d* get_k();

};
