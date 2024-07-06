
#include "pch.h"
#include "Physics.h"
#include "Light.h"


Polarization::Polarization()
{
	Z << 0, 0, 1;
	T = Matrix3cd{ {1, -sc::i, 0}, {0, 0, sqrt(2)}, {-1, -sc::i, 0} };
	T /= sqrt(2);
	R = AngleAxisd(0, Vector3d(0, 0, 1));
	Rq = AngleAxisd(0, Vector3d(0, 0, 1));

	q_axis << 0, 0, 1;

	x << 0, 0, 1;
	q << 0, 1, 0;
}

void Polarization::init(Vector3cd vec, Vector3d _q_axis, bool vec_as_q)
{
	if (vec_as_q) q = vec / vec.norm();
	else x = vec / vec.norm();
	def_q_axis(_q_axis, vec_as_q);
}

void Polarization::calc_R(Vector3d _q_axis)
{
	// double angle = acos(q_axis.dot(_q_axis) / sqrt(q_axis.dot(q_axis) * _q_axis.dot(_q_axis)));
	// Vector3d rot_axis = q_axis.cross(_q_axis);
	double angle = acos(Z.dot(_q_axis) / sqrt(_q_axis.dot(_q_axis)));
	Vector3d rot_axis = Z.cross(_q_axis);
	if (rot_axis.sum() == 0) rot_axis(2) = 1;
	rot_axis /= rot_axis.norm();
	Rq = AngleAxisd(angle, rot_axis);
	R = R * Rq;
}

void Polarization::def_q_axis(Vector3d _q_axis, bool q_fixed)
{	
	calc_R(_q_axis);
	q_axis = _q_axis / _q_axis.norm();
	if (q_fixed) infer_x();
	else infer_q();
}

void Polarization::infer_x()
{
	x = Rq.matrix() * (T.adjoint() * q);
	for (size_t i = 0; i < 3; ++i)
	{
		if (abs(x.array()[i]) < 1e-9) x(i) = 0;
	}
	x /= x.norm();
}

void Polarization::infer_q()
{
	q = T * (Rq.matrix().transpose() * x);
	for (size_t i = 0; i < 3; ++i)
	{
		if (abs(q.array()[i]) < 1e-9) q(i) = 0;
	}
	q /= q.norm();
}

Vector3cd* Polarization::get_x()
{
	return &x;
}

Vector3cd* Polarization::get_q()
{
	return &q;
}

Vector3d* Polarization::get_q_axis()
{
	return &q_axis;
}


Laser::Laser()
{
	freq = 0.;
	intensity = 1.;
	polarization = new Polarization();
	k << 1, 0, 0;
}

void Laser::init(double _freq, double _intensity, Polarization* _polarization, Vector3d _k)
{
	set_freq(_freq);
	set_intensity(_intensity);
	set_polarization(_polarization);
	set_k(_k);
}

double Laser::get_detuned(const Vector3d& v)
{
	double angle = 0;
	if (v.norm() != 0) angle = acos(v.dot(k) / (v.norm() * k.norm()));
	return doppler(freq, v.norm(), angle);
}

double Laser::get_detuned(double delta, const Vector3d& v)
{
	double angle = 0;
	if (v.norm() != 0) angle = acos(v.dot(k) / (v.norm() * k.norm()));
	return doppler(freq + delta, v.norm(), angle);
}

double Laser::get_intensity()
{
	return intensity;
}

void Laser::set_intensity(double _intensity)
{
	intensity = _intensity;
}

Polarization* Laser::get_polarization()
{
	return polarization;
}

void Laser::set_polarization(Polarization* _polarization)
{
	polarization = _polarization;
}

double Laser::get_freq()
{
	return freq;
}

void Laser::set_freq(double _freq)
{
	freq = _freq;
}

Vector3d Laser::get_k()
{
	return k;
}

void Laser::set_k(Vector3d _k)
{
	k = _k / _k.norm() * freq / sc::c;
}

Vector3d Laser::get_kn()
{
	return k / k.norm();
}
