#pragma once

#include "Physics.h"
#include "Matter.h"
#include "Light.h"
#include "Results.h"
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


typedef runge_kutta_dopri5< MatrixXcd, double, MatrixXcd, double, vector_space_algebra > dopri5_vd_type;
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
	double dt_var = 1e-3;
	double delta_max = 1e3;
	bool loop = false;
	bool time_dependent = false;
	bool controlled = false;

	std::array<std::vector<MatrixXi>, 3> lasermap;
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
	double get_dt();
	void set_dt(double _dt);

	void update();
	void gen_coordinates();
	void gen_rabi();
	void gen_trees();
	void gen_conlist();
	void gen_deltamap();
	void propagate(size_t i, size_t i0, std::set<size_t>& visited, const std::vector<size_t>& tree,
		std::array<std::vector<size_t>, 2>& path, std::vector<MatrixXd>& shifts);

	VectorXd* gen_w0();
	VectorXd* gen_w0(Environment& env);
	VectorXd* gen_w(const bool dynamics = false);
	VectorXd* gen_w(const VectorXd& delta, const bool dynamics = false);
	VectorXd* gen_w(const Vector3d& v, const bool dynamics = false);
	VectorXd* gen_w(const VectorXd& delta, const Vector3d& v, const bool dynamics = false);
	void update_w(VectorXd& w, const VectorXd& delta, const Vector3d& v, const bool dynamics = false);
	// VectorXd gen_delta(VectorXd& w0, VectorXd& w);
	VectorXd gen_delta(const VectorXd& w0, const VectorXd& w);

	MatrixXd* gen_rates(VectorXd& w0, VectorXd& w);
	VectorXd* gen_rates_sum(MatrixXd& R);
	void update_rates();

	MatrixXcd* gen_hamiltonian(VectorXd& w0, VectorXd& w);
	void update_hamiltonian_off(MatrixXcd& H);

	void update_hamiltonian(MatrixXcd& H, VectorXd& w0, VectorXd& w, double t);
	void update_hamiltonian_off(MatrixXcd& H, VectorXd& w, double t);

	MatrixXcd* gen_hamiltonian_leaky(VectorXd& w0, VectorXd& w);
	void update_hamiltonian_leaky(MatrixXcd& H, VectorXd& w0, VectorXd& w, double t);

	void update_hamiltonian_diag(MatrixXcd& H, VectorXd& w0, VectorXd& w);
	void update_hamiltonian_leaky_diag(MatrixXcd& H, VectorXd& w0, VectorXd& w);

	size_t arange_t(double t);

	Result* rate_equations(size_t n, VectorXd& x0, MatrixXd& R, VectorXd& R_sum);
	Result* rate_equations(size_t n, VectorXd& x0, const VectorXd& delta, const Vector3d& v);
	Result* rate_equations(size_t n, VectorXd& x0, const VectorXd& delta);
	Result* rate_equations(size_t n, VectorXd& x0, const Vector3d& v);
	Result* rate_equations(size_t n, VectorXd& x0);
	Result* rate_equations(size_t n);
	Result* rate_equations(double t);

	Result* schroedinger(size_t n, VectorXcd& x0, VectorXd& w0, VectorXd& w, MatrixXcd& H);
	Result* schroedinger(size_t n, VectorXcd& x0, MatrixXcd& H);
	Result* schroedinger(size_t n, VectorXcd& x0, const VectorXd& delta, const Vector3d& v);
	Result* schroedinger(size_t n, VectorXcd& x0, const VectorXd& delta);
	Result* schroedinger(size_t n, VectorXcd& x0, const Vector3d& v);
	Result* schroedinger(size_t n, VectorXcd& x0);
	Result* schroedinger(size_t n);
	Result* schroedinger(double t);

	Result* master(size_t n, MatrixXcd& x0, VectorXd& w0, VectorXd& w, MatrixXcd& H);
	Result* master(size_t n, MatrixXcd& x0, MatrixXcd& H);
	Result* master(size_t n, MatrixXcd& x0, const VectorXd& delta, const Vector3d& v);
	Result* master(size_t n, MatrixXcd& x0, const VectorXd& delta);
	Result* master(size_t n, MatrixXcd& x0, const Vector3d& v);
	Result* master(size_t n, MatrixXcd& x0);
	Result* master(size_t n);
	Result* master(double t);

	Result* master_mc(size_t n, std::vector<VectorXcd>& x0, const VectorXd& delta, const std::vector<Vector3d>& v, const bool dynamics = false);
	Result* master_mc(size_t n, std::vector<VectorXcd>& x0, const VectorXd& delta, const Vector3d& v, size_t num);
	Result* master_mc(size_t n, const VectorXd& delta, std::vector<Vector3d>& v, bool dynamics = false);
	Result* master_mc(size_t n, std::vector<VectorXcd>& x0, size_t num, bool dynamics = false);
	Result* master_mc(size_t n, size_t num, bool dynamics = false);
	Result* master_mc(double t, size_t num, bool dynamics = false);

	Result* call_solver_v(size_t n, VectorXd& x0, const VectorXd& delta, const Vector3d& v, int solver);
	Result* call_solver_v(size_t n, VectorXcd& x0, const VectorXd& delta, const Vector3d& v, int solver);
	Result* call_solver_v(size_t n, MatrixXcd& x0, const VectorXd& delta, const Vector3d& v, int solver);

	template<typename T>
	Result* mean_v(const VectorXd& delta, const std::vector<Vector3d>& v, size_t n, T& y0, int solver);
	template<typename T>
	Result* mean_v(const std::vector<Vector3d>& v, size_t n, T& y0, int solver);
	Result* mean_v(const std::vector<Vector3d>& v, size_t n, int solver);
	Result* mean_v(const std::vector<Vector3d>& v, double t, int solver);

	Result* call_solver_d(size_t n, VectorXd& x0, const VectorXd& delta, int solver);
	Result* call_solver_d(size_t n, VectorXcd& x0, const VectorXd& delta, int solver);
	Result* call_solver_d(size_t n, MatrixXcd& x0, const VectorXd& delta, int solver);

	template<typename T>
	Spectrum* spectrum(const std::vector<VectorXd>& delta, size_t n, T& y0, int solver);
	Spectrum* spectrum(const std::vector<VectorXd>& delta, size_t n, int solver);
	Spectrum* spectrum(const std::vector<VectorXd>& delta, double t, int solver);
	template<typename T>
	Spectrum* spectrum(const std::vector<VectorXd>& delta, const std::vector<Vector3d>& v, size_t n, T& y0, int solver);
	Spectrum* spectrum(const std::vector<VectorXd>& delta, const std::vector<Vector3d>& v, size_t n, int solver);
	Spectrum* spectrum(const std::vector<VectorXd>& delta, const std::vector<Vector3d>& v, double t, int solver);
	Spectrum* spectrum_mc(const std::vector<VectorXd>& delta, const std::vector<Vector3d>& v, size_t n, std::vector<VectorXcd>& y0, int solver, bool dynamics = false);

	void set_result(Result* _result);
	Result* get_result();

	bool get_time_dependent();
	void set_time_dependent(bool _time_dependent);
	bool get_loop();
	MatrixXi* get_summap();
	std::vector<MatrixXcd>* get_rabimap();
	MatrixXd* get_atommap();
	MatrixXd* get_deltamap();
};
