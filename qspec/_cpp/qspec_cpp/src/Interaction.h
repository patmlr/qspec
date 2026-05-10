#pragma once

// order is important
#include "odeint_eigen_complex_norm.hpp"
#include <boost/numeric/odeint.hpp>
#include <boost/numeric/odeint/external/eigen/eigen.hpp>
#include <unsupported/Eigen/MatrixFunctions>

#include <Eigen/Core>

#if defined(EIGEN_VECTORIZE_AVX512)
#pragma message("Eigen: AVX-512 enabled")
#elif defined(EIGEN_VECTORIZE_AVX2)
#pragma message("Eigen: AVX2 enabled")
#elif defined(EIGEN_VECTORIZE_AVX)
#pragma message("Eigen: AVX enabled")
#elif defined(EIGEN_VECTORIZE_SSE4_2)
#pragma message("Eigen: SSE4.2 enabled")
#elif defined(EIGEN_VECTORIZE_SSE2)
#pragma message("Eigen: SSE2 enabled")
#elif defined(EIGEN_VECTORIZE_NEON)
#pragma message("Eigen: NEON enabled")
#else
#pragma message("Eigen: NO SIMD")
#endif

#include "Physics.h"
#include "Matter.h"
#include "Light.h"
#include "Utility.h"
#include <atomic>
#include <thread>
#include <set>
#include <queue>
#include <random>
#include <iostream>

#define FMT_HEADER_ONLY
#include <fmt/core.h>

using namespace boost::numeric::odeint;

using adams_vd_type = adams_bashforth<4, VectorXd, double, VectorXd, double, vector_space_algebra>;
using adams_vcd_type = adams_bashforth<4, VectorXcd, double, VectorXcd, double, vector_space_algebra>;
using adams_mcd_type = adams_bashforth<4, MatrixXcd, double, MatrixXcd, double, vector_space_algebra>;

using rk4_vd_type = runge_kutta4<VectorXd, double, VectorXd, double, vector_space_algebra>;
using rk4_vcd_type = runge_kutta4<VectorXcd, double, VectorXcd, double, vector_space_algebra>;
using rk4_mcd_type = runge_kutta4<MatrixXcd, double, MatrixXcd, double, vector_space_algebra>;

using dopri5_vd_type = runge_kutta_dopri5< VectorXd, double, VectorXd, double, vector_space_algebra >;
using c_dopri5_vd_type = controlled_runge_kutta< dopri5_vd_type >;
using d_dopri5_vd_type = dense_output_runge_kutta< c_dopri5_vd_type >;

using dopri5_vcd_type = runge_kutta_dopri5< VectorXcd, double, VectorXcd, double, vector_space_algebra >;
using c_dopri5_vcd_type = controlled_runge_kutta< dopri5_vcd_type >;
using d_dopri5_vcd_type = dense_output_runge_kutta< c_dopri5_vcd_type >;

using dopri5_mcd_type = runge_kutta_dopri5< MatrixXcd, double, MatrixXcd, double, vector_space_algebra >;
using c_dopri5_mcd_type = controlled_runge_kutta< dopri5_mcd_type >;
using d_dopri5_mcd_type = dense_output_runge_kutta< c_dopri5_mcd_type >;

using ArrayXb = Array<bool, Dynamic, Dynamic>;
using Matrix3Xd = Matrix<double, 3, Dynamic>;

VectorXd rate_exponential(double t, VectorXd x0, MatrixXd R);


class Interaction
{
protected:
	Atom* atom;
	std::vector<Laser*> lasers;
	Environment* env;

	double dt = 1e-3;
	double dt_max = 1e-3;
	double atol = 1e-6;
	double rtol = 1e-6;
	double delta_max = 1e3;
	bool loop = false;
	bool time_dependent = false;
	bool controlled = true;
	bool dense = true;

	std::vector<std::vector<MatrixXi>> lasermap;
	MatrixXi summap;
	std::vector<MatrixXcd> rabimap;
	std::vector<std::vector<size_t>> trees;
	std::vector<std::vector<size_t>> con_list;
	MatrixXd deltamap;
	MatrixXd atommap;
	std::vector<MatrixXi> tmap;

public:
	int n_history;
	std::vector<size_t> history;
	int info = 0;

	Interaction();

	void init(Atom* _atom, std::vector<Laser*> _lasers, Environment* _env);

	Atom* get_atom();
	void set_atom(Atom* _atom);

	void clear_lasers();
	std::vector<Laser*>* get_lasers();
	void add_laser(Laser* laser);

	Environment* get_env();
	void set_env(Environment* _env);

	double get_delta_max();
	void set_delta_max(double _delta_max);

	bool get_controlled();
	void set_controlled(bool _controlled);

	bool get_dense();
	void set_dense(bool _dense);

	bool get_time_dependent();
	void set_time_dependent(bool _time_dependent);

	double get_dt();
	void set_dt(double _dt);

	double get_dt_max();
	void set_dt_max(double _dt_max);

	double get_atol();
	void set_atol(double _atol);

	double get_rtol();
	void set_rtol(double _rtol);

	bool get_loop();
	MatrixXi* get_summap();
	std::vector<MatrixXcd>* get_rabimap();
	MatrixXd* get_atommap();
	MatrixXd* get_deltamap();
	MatrixXcd get_hamiltonian(const double t, const VectorXd& delta, const Vector3d& v);

	void resonance_info();

	int update();
	void gen_coordinates();
	void gen_rabi();
	void gen_trees();
	void gen_conlist();
	void gen_deltamap();
	void propagate(size_t i, size_t i0, std::set<size_t>& visited, const std::vector<size_t>& tree,
		std::array<std::vector<size_t>, 2>& path, std::vector<MatrixXd>& shifts);

	VectorXd gen_w(const bool dynamics = false);
	VectorXd gen_w(const VectorXd& delta, const bool dynamics = false);
	VectorXd gen_w(const Vector3d& v, const bool dynamics = false);
	VectorXd gen_w(const VectorXd& delta, const Vector3d& v, const bool dynamics = false);
	void update_w(VectorXd& w, const VectorXd& delta, const Vector3d& v, const bool dynamics = false);
	// VectorXd gen_delta(VectorXd& w0, VectorXd& w);

	VectorXd get_delta(const VectorXd& w0, const VectorXd& w);
	void update_delta(VectorXd& delta, const VectorXd& w0, const VectorXd& w);

	std::vector<MatrixXd> gen_R_k(VectorXd& w0, VectorXd& w);
	Vector3d gen_k_up(std::mt19937& gen, VectorXd& w0, VectorXd& w, size_t i, size_t j);
	Vector3d gen_velocity_change(std::mt19937& gen, VectorXd& w0, VectorXd& w, size_t i, size_t j, size_t f);

	MatrixXd gen_rates(const VectorXd& w0, const VectorXd& w);
	VectorXd gen_rates_sum(const MatrixXd& R);
	void update_rates(MatrixXd& R, const VectorXd& w0, const VectorXd& w);
	void update_rates_sum(VectorXd& R_sum, const MatrixXd& R);

	MatrixXcd gen_hamiltonian(const VectorXd& w0, const VectorXd& w);

	void update_hamiltonian(MatrixXcd& H, const VectorXd& w0, const VectorXd& w);
	void update_hamiltonian_diag(MatrixXcd& H, const VectorXd& w0, const VectorXd& w);
	void update_hamiltonian_off(MatrixXcd& H);

	void update_hamiltonian(MatrixXcd& H, const VectorXd& w0, const VectorXd& w, double t);
	void update_hamiltonian_off(MatrixXcd& H, const VectorXd& w, double t);

	MatrixXcd gen_hamiltonian_leaky(const VectorXd& w0, const VectorXd& w);

	void update_hamiltonian_leaky(MatrixXcd& H, const VectorXd& w0, const VectorXd& w, double t);
	void update_hamiltonian_leaky_diag(MatrixXcd& H, const VectorXd& w0, const VectorXd& w);

	std::vector<std::vector<VectorXd>> rates(
		const std::vector<double>& t,
		const std::vector<VectorXd>& delta,
		const std::vector<Vector3d>& v,
		std::vector<VectorXd>& x0,
		const bool analytic
	);

	std::vector<std::vector<VectorXcd>> schroedinger(
		const std::vector<double>& t,
		const std::vector<VectorXd>& delta,
		const std::vector<Vector3d>& v,
		std::vector<VectorXcd>& x0
	);

	std::vector<std::vector<MatrixXcd>> master(
		const std::vector<double>& t,
		const std::vector<VectorXd>& delta,
		const std::vector<Vector3d>& v,
		std::vector<MatrixXcd>& x0
	);

	std::vector<std::vector<VectorXcd>> mc_master(
		const std::vector<double>& t,
		const std::vector<VectorXd>& delta,
		std::vector<Vector3d>& v,
		std::vector<VectorXcd>& x0,
		const bool dynamics = false
	);
};
