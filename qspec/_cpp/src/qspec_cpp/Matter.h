#pragma once

#include <stdexcept>
#include <array>
#include <set>
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
	~Environment();
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
	double j;
	double i;
	double f;
	double m;
	bool parity;
	double* hyper_const;
	double gj;
	double gi;
	std::string label;

public:

	State();
	~State();
	void init(double _freq_j, double _j, double _i, double _f, double _m, bool _parity,
		double* _hyper_const, double _gj, double _gi, std::string _label);

	void reset();

	double get_shift();
	void set_shift(double _shift);

	double get_freq_j();
	void set_freq_j(double _freq_j);

	double get_freq();
	void set_freq(double _freq);

	double get_j();
	void set_j(double _j);

	double get_i();
	void set_i(double _i);

	double get_f();
	void set_f(double _f);

	double get_m();
	void set_m(double _m);

	bool get_parity();
	void set_parity(bool _parity);

	double* get_hyper_const();
	void set_hyper_const(double* _hyper_const);

	double get_gj();
	void set_gj(double _gj);

	double get_gi();
	void set_gi(double _gi);

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
	double mass = 0;
	size_t k_em_max = 1;
	size_t size = 0;

	std::vector<size_t> gs;
	std::array<MatrixXd, 3> m_e1;  // -1, 0, +1
	std::array<MatrixXd, 3> m_m1;  // -1, 0, +1

	std::vector<MatrixXi> ek;
	std::vector<MatrixXi> mk;
	std::vector<MatrixXd> d_em;

	VectorXd w0;
	VectorXd Lsum;
	MatrixXd L0;
	MatrixXd L1;

public:

	Atom();
	~Atom();
	void init(std::vector<State*> _states, DecayMap* _decays);
	void add_state(State* state);
	void clear_states();
	void gen_w0();
	VectorXd* get_w0();

	void gen_frequencies(Environment* env);
	void gen_multipole();
	void gen_dipole();
	void update();
	void update(Environment* env);

	std::vector<State*>* get_states();

	DecayMap* get_decay_map();
	void set_decay_map(DecayMap* _decays);

	double get_mass();
	void set_mass(double _mass);

	size_t get_k_em_max();
	size_t get_min_k(size_t i, size_t j);
	bool get_parity_equal(size_t i, size_t j);

	size_t get_size();

	State* get(size_t index);

	std::vector<size_t>* get_gs();
	std::array<MatrixXd, 3>* get_m_e1();
	std::array<MatrixXd, 3>* get_m_m1();

	std::vector<MatrixXi> get_ek();
	MatrixXi get_ek(size_t k);
	size_t get_ek(size_t k, size_t i, size_t j);

	std::vector<MatrixXi> get_mk();
	MatrixXi get_mk(size_t k);
	size_t get_mk(size_t k, size_t i, size_t j);

	std::vector<MatrixXi> get_emk();
	MatrixXi get_emk(size_t k);
	size_t get_emk(size_t k, size_t i, size_t j);

	std::vector<MatrixXd> get_d_em();
	MatrixXd get_d_em(size_t k);
	double get_d_em(size_t k, size_t i, size_t j);

	VectorXd* get_Lsum();
	MatrixXd* get_L0();
	MatrixXd* get_L1();

};


double* get_f(double i, double j);
