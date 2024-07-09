#pragma once

#include "Physics.h"
#include "Matter.h"
#include "Light.h"
#include "Utility.h"
#include <set>
#include <queue>
#include <random>
#include <execution>
#include <iostream>
#include <boost/numeric/odeint.hpp>
#include <boost/numeric/odeint/external/eigen/eigen.hpp>

using namespace boost::numeric::odeint;

typedef adams_bashforth<4, VectorXd, double, VectorXd, double, vector_space_algebra> adams_vd_type;
typedef adams_bashforth<4, VectorXcd, double, VectorXcd, double, vector_space_algebra> adams_vcd_type;
typedef adams_bashforth<4, MatrixXcd, double, MatrixXcd, double, vector_space_algebra> adams_mcd_type;

typedef runge_kutta4<VectorXd, double, VectorXd, double, vector_space_algebra> rk4_vd_type;
typedef runge_kutta4<VectorXcd, double, VectorXcd, double, vector_space_algebra> rk4_vcd_type;
typedef runge_kutta4<MatrixXcd, double, MatrixXcd, double, vector_space_algebra> rk4_mcd_type;


typedef runge_kutta_dopri5< VectorXd, double, VectorXd, double, vector_space_algebra > dopri5_vd_type;
typedef controlled_runge_kutta< dopri5_vd_type > c_dopri5_vd_type;
typedef dense_output_runge_kutta< c_dopri5_vd_type > d_dopri5_vd_type;

typedef runge_kutta_dopri5< VectorXcd, double, VectorXcd, double, vector_space_algebra > dopri5_vcd_type;
typedef controlled_runge_kutta< dopri5_vcd_type > c_dopri5_vcd_type;
typedef dense_output_runge_kutta< c_dopri5_vcd_type > d_dopri5_vcd_type;

typedef runge_kutta_dopri5< MatrixXcd, double, MatrixXcd, double, vector_space_algebra > dopri5_mcd_type;
typedef controlled_runge_kutta< dopri5_mcd_type > c_dopri5_mcd_type;
typedef dense_output_runge_kutta< c_dopri5_mcd_type > d_dopri5_mcd_type;

typedef Array<bool, Dynamic, Dynamic> ArrayXb;
typedef Matrix<double, 3, Dynamic> Matrix3Xd;


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

	std::array<std::vector<MatrixXi>, 3> lasermap;
	MatrixXi summap;
	std::vector<MatrixXcd> rabimap;
	MatrixXd laser_gamma_map;
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

	void update();
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
	VectorXd gen_delta(const VectorXd& w0, const VectorXd& w);

	std::vector<MatrixXd> gen_R_k(VectorXd& w0, VectorXd& w);
	Vector3d gen_k_up(std::mt19937& gen, VectorXd& w0, VectorXd& w, size_t i, size_t j);
	Vector3d gen_velocity_change(std::mt19937& gen, VectorXd& w0, VectorXd& w, size_t i, size_t j, size_t f);

	MatrixXd gen_rates(VectorXd& w0, VectorXd& w);
	VectorXd gen_rates_sum(MatrixXd& R);
	void update_rates(MatrixXd& R, VectorXd& w0, VectorXd& w);
	void update_rates_sum(VectorXd& R_sum, MatrixXd& R);

	MatrixXcd gen_hamiltonian(VectorXd& w0, VectorXd& w);
	void update_hamiltonian_off(MatrixXcd& H);

	void update_hamiltonian(MatrixXcd& H, VectorXd& w0, VectorXd& w, double t);
	void update_hamiltonian_off(MatrixXcd& H, VectorXd& w, double t);

	MatrixXcd gen_hamiltonian_leaky(VectorXd& w0, VectorXd& w);
	void update_hamiltonian_leaky(MatrixXcd& H, VectorXd& w0, VectorXd& w, double t);

	void update_hamiltonian_diag(MatrixXcd& H, VectorXd& w0, VectorXd& w);
	void update_hamiltonian_leaky_diag(MatrixXcd& H, VectorXd& w0, VectorXd& w);

	std::vector<std::vector<VectorXd>> rates(
		const std::vector<double>& t, const std::vector<VectorXd>& delta, const std::vector<Vector3d>& v, std::vector<VectorXd>& x0);
	std::vector<std::vector<VectorXcd>> schroedinger(
		const std::vector<double>& t, const std::vector<VectorXd>& delta, const std::vector<Vector3d>& v, std::vector<VectorXcd>& x0);
	std::vector<std::vector<MatrixXcd>> master(
		const std::vector<double>& t, const std::vector<VectorXd>& delta, const std::vector<Vector3d>& v, std::vector<MatrixXcd>& x0);
	std::vector<std::vector<VectorXcd>> mc_master(
		const std::vector<double>& t, const std::vector<VectorXd>& delta, std::vector<Vector3d>& v, std::vector<VectorXcd>& x0, const bool dynamics = false);
};
