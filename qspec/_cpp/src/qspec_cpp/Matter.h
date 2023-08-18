#pragma once

#include <array>
#include <vector>
#include <string>
#include <random>
#include <Eigen/Dense>

using namespace Eigen;


class Environment
{
protected:
	double E;
	double B;
	Vector3d e_E;
	Vector3d e_B;
public:
	Environment();
	double get_E();
	double get_B();
	Vector3d* get_e_E();
	Vector3d* get_e_B();

	void set_E(double _E);
	void set_E(Vector3d _E);

	void set_B(double _B);
	void set_B(Vector3d _B);
};


class State
{
protected:
	const int HYPER_SIZE = 3;
	double freq_j;
	double freq;
	double s;
	double l;
	double j;
	double i;
	double f;
	double m;
	double* hyper_const;
	double g;
	std::string label;

public:

	State();
	~State();
	void init(double _freq_j, double _s, double _l, double _j, double _i, double _f, double _m,
		double* _hyper_const, double _g, std::string _label);
	void update();
	void update(Environment* env);
	double get_shift();

	double get_freq_j();
	void set_freq_j(double _freq_j);

	double get_freq();

	double get_s();
	void set_s(double _s);

	double get_l();
	void set_l(double _l);

	double get_j();
	void set_j(double _j);

	double get_i();
	void set_i(double _i);

	double get_f();
	void set_f(double _f);

	double get_m();
	void set_m(double _m);

	double* get_hyper_const();
	void set_hyper_const(double* _hyper_const);

	double get_g();
	void set_g(double _g);

	std::string get_label();
	void set_label(std::string _label);
};


class DecayMap
{
protected:
	size_t size;
	std::vector<std::string> states_0;
	std::vector<std::string> states_1;
	std::vector<double> a;
public:
	DecayMap();
	~DecayMap();
	DecayMap(std::vector<std::string> _states_0, std::vector<std::string> _states_1, std::vector<double> _a);
	void add_decay(std::string state_0, std::string state_1, double _a);
	size_t get_size();
	std::vector<std::string>* get_states_0();
	std::vector<std::string>* get_states_1();
	std::vector<double>* get_a();
	double get_item(std::string state_0, std::string state_1);
	double get_gamma(std::string state_0, std::string state_1);
};


class Atom
{
protected:
	std::vector<State*> states;
	DecayMap* decays;
	double mass;
	size_t size;

	std::vector<size_t> gs;
	std::array<MatrixXd, 3> m_dipole;  // -1, 0, +1
	VectorXd w0;
	VectorXd Lsum;
	MatrixXd L0;
	MatrixXd L1;

public:

	Atom();
	~Atom();
	void init(std::vector<State*> _states, DecayMap* _decays);
	void update();
	void add_state(State* state);
	void clear_states();
	void gen_w0();
	VectorXd* get_w0();

	std::vector<State*>* get_states();
	DecayMap* get_decay_map();
	void set_decay_map(DecayMap* _decays);
	double get_mass();
	void set_mass(double _mass);
	size_t get_size();

	State* get(size_t index);
	std::vector<size_t>* get_gs();
	std::array<MatrixXd, 3>* get_m_dipole();
	VectorXd* get_Lsum();
	MatrixXd* get_L0();
	MatrixXd* get_L1();

	Vector3d gen_velocity_change(size_t i, size_t j, size_t f, Vector3d& k, std::mt19937& gen);
};


double* get_f(double i, double j);
