
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
		if (abs(x.array()[i]) < 1e-15) x(i) = 0;
	}
	x /= x.norm();
}

void Polarization::infer_q()
{
	q = T * (Rq.matrix().transpose() * x);
	for (size_t i = 0; i < 3; ++i)
	{
		if (abs(q.array()[i]) < 1e-15) q(i) = 0;
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

Polarizationk::Polarizationk()
{
	Z << 0, 0, 1;
	T = Matrix3cd{ {1, -sc::i, 0}, {0, 0, sqrt(2)}, {-1, -sc::i, 0} };
	T /= sqrt(2);

	q_axis << 0, 0, 1;
	Rz = Matrix3d::Identity();

	theta_k = 0.;
	phi_k = 0.;
	Rk = Matrix3d::Identity();

	x << 0, 0, 1;
	qk << 1, 0, -1;
	qk /= sqrt(2);
}

Polarizationk::Polarizationk(Vector3d _q_axis)
{
	Z << 0, 0, 1;
	T = Matrix3cd{ {1, -sc::i, 0}, {0, 0, sqrt(2)}, {-1, -sc::i, 0} };
	T /= sqrt(2);

	q_axis = _q_axis / _q_axis.norm();
	Rz = rotation_matrix(q_axis);

	theta_k = 0.;
	phi_k = 0.;
	Rk = Matrix3d::Identity();

	x << 0, 0, 1;
	qk << 1, 0, -1;
	qk /= sqrt(2);
}

void Polarizationk::init(Vector3cd _x, Vector3d _k, Vector3d _q_axis)
{
	x = _x / _x.norm();
	k = _k / _k.norm();
	def_q_axis(_q_axis);
}

void Polarizationk::init_qk(Vector3cd _x, Vector3d _k)
{
	x = _x / _x.norm();
	k = _k / _k.norm();
	infer_qk();
}

void Polarizationk::def_q_axis(Vector3d _q_axis)
{
	q_axis = _q_axis / _q_axis.norm();
	Rz = rotation_matrix(q_axis).transpose();
	infer_qk();
}

void Polarizationk::infer_qk()
{
	Vector3d kz = Rz * k;
	Vector3cd xz = Rz * x;

	theta_k = rotation_theta(kz);
	phi_k = rotation_phi(kz);
	Rk = rotation_matrix(kz);
	
	/*printf("theta_k, phi_k: %.3f, %.3f\n", theta_k, phi_k);
	printf("x: %.3f, %.3f, %.3f\n", std::abs(x(0)), std::abs(x(1)), std::abs(x(2)));
	printf("xz: %.3f, %.3f, %.3f\n", std::abs(xz(0)), std::abs(xz(1)), std::abs(xz(2)));
	printf("Rk * xz: %.3f, %.3f, %.3f\n", std::abs((Rk.transpose() * xz)(0)), std::abs((Rk.transpose() * xz)(1)), std::abs((Rk.transpose() * xz)(2)));

	printf("k: %.3f, %.3f, %.3f\n", k(0), k(1), k(2));
	printf("kz: %.3f, %.3f, %.3f\n", kz(0), kz(1), kz(2));*/

	qk = T * (Rk.transpose() * xz);
	for (size_t i = 0; i < 3; ++i)
	{
		if (abs(qk.array()[i]) < 1e-15) qk(i) = 0;
	}
	// printf("qk: %.3f, %.3f, %.3f\n", std::abs(qk(0)), std::abs(qk(1)), std::abs(qk(2)));

	if (pow(std::abs(qk(1)), 2) > 1e-2) printf("\033[93mWarning: %.3f %% of the field amplitude oscillates along the k-vector (%.3f, %.3f, %.3f). Setting field component to 0.\033[0m\n",
		100. * std::abs(qk(1)) / (std::abs(qk(0)) + std::abs(qk(1)) + std::abs(qk(2))), k(0), k(1), k(2));
	qk(1) = 0.;
	if (qk.norm() == 0.) throw polarization_error("An electro-magnetic wave cannot oscillate along its k-vector.");
	qk /= qk.norm();

}

double Polarizationk::get_theta_k()
{
	return theta_k;
}

double Polarizationk::get_phi_k()
{
	return phi_k;
}

Vector3cd* Polarizationk::get_x()
{
	return &x;
}

Vector3cd* Polarizationk::get_qk()
{
	return &qk;
}

Vector3d* Polarizationk::get_q_axis()
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

Vector3d* Laser::get_k()
{
	return &k;
}

void Laser::set_k(Vector3d _k)
{
	k = _k / _k.norm() * freq / sc::c;
}

Vector3d Laser::get_k_si()
{
	return k * freq / sc::c;
}

VectorXcd Laser::get_kpol(bool electric, size_t _k, Vector3d q_axis)
{
	Polarizationk polarization_k = Polarizationk();
	polarization_k.init(*polarization->get_x(), k, q_axis);

	return spherical_tensor(electric, _k, *polarization_k.get_qk(), polarization_k.get_theta_k(), polarization_k.get_phi_k());
}
