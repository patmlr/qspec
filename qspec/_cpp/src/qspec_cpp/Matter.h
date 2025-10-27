#pragma once

#include <stdexcept>
#include <array>
#include <set>
#include <vector>
#include <string>
#include <random>
#include <execution>
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
	std::vector<double> s;
	std::vector<double> l;
	std::vector<double> jj;
	bool parity;
	double* hyper_const;
	double gj;
	double gi;
	std::string label;

public:

	State();
	~State();
	void init(double _freq_j, double _j, double _i, double _f, double _m, bool _parity,
			  std::vector<double> _s, std::vector<double> _l, std::vector<double> _jj,
			  double* _hyper_const, double _gj, double _gi, std::string _label);

	std::string repr();
	void reset();

	double get_shift();
	void set_shift(double _shift);

	double get_freq_j();
	void set_freq_j(double _freq_j);

	double get_freq();
	void set_freq(double _freq);

	std::vector<double> get_s();
	void set_s(std::vector<double> _s);

	std::vector<double> get_l();
	void set_l(std::vector<double> _l);

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

	std::vector<double> get_jj();
	void set_jj(std::vector<double> _jj);

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
	size_t k_em_max = 1;
	std::vector<std::string> states_0;
	std::vector<std::string> states_1;

	std::vector<bool> single_leading_order;
	std::vector<std::vector<double>> ae;
	std::vector<std::vector<double>> am;
public:
	DecayMap();
	~DecayMap();
	DecayMap(size_t _k_em_max);
	void add_decay(std::string state_0, std::string state_1, std::vector<double> _ae, std::vector<double> _am, bool _single_leading_order);

	size_t get_size();

	size_t get_k_em_max();
	void set_k_em_max(size_t _k_em_max);

	std::vector<std::string>* get_states_0();
	std::vector<std::string>* get_states_1();

	size_t get_index(std::string state_0, std::string state_1);

	double get_a(size_t i, bool parity_equal);
	double get_a(std::string state_0, std::string state_1, bool parity_equal);

	bool get_single_leading_order(size_t i);
	bool get_single_leading_order(std::string state_0, std::string state_1);

	std::vector<std::vector<double>>* get_ae();
	std::vector<double> get_ae(size_t i);
	double get_ae(size_t i, size_t k);
	std::vector<double> get_ae(std::string state_0, std::string state_1);
	double get_ae(std::string state_0, std::string state_1, size_t k);

	std::vector<std::vector<double>>* get_am();
	std::vector<double> get_am(size_t i);
	double get_am(size_t i, size_t k);
	std::vector<double> get_am(std::string state_0, std::string state_1);
	double get_am(std::string state_0, std::string state_1, size_t k);
};


class Atom
{
protected:
	std::vector<State*> states;
	DecayMap* decays;
	double mass = 0;
	size_t size = 0;

	Environment* env;

	std::vector<size_t> gs;

	std::vector<MatrixXi> ek;
	std::vector<MatrixXi> mk;
	std::vector<MatrixXd> a_em;
	std::vector<MatrixXd> d_em;

	VectorXd w0;
	VectorXd Lsum;
	// std::vector<MatrixXd> L0_k;
	// std::vector<MatrixXd> L1_k;
	std::vector<MatrixXd> A_einst;
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

	bool is_electric(size_t k, size_t i, size_t j);
	void gen_multipole();
	void gen_frequencies(Environment* _env);
	void update();

	std::vector<State*>* get_states();

	DecayMap* get_decay_map();
	void set_decay_map(DecayMap* _decays);
	double get_gamma(size_t i);

	double get_mass();
	void set_mass(double _mass);

	Environment* get_env();
	void set_env(Environment* _env);

	size_t get_min_k(size_t i, size_t j);
	bool get_parity_equal(size_t i, size_t j);

	size_t get_size();

	State* get(size_t index);

	std::vector<size_t>* get_gs();

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

	void scattering_rate(double* results, std::vector<size_t>& k, std::vector<MatrixXcd>& rho, std::vector<Vector3d>& k_vec, std::vector<Vector3cd>& x_vec, std::vector<size_t>& i, std::vector<size_t>& f);
	void scattering_rate(double* results, std::vector<size_t>& k, std::vector<MatrixXcd>& rho, std::vector<Vector3d>& k_vec, std::vector<size_t>& i, std::vector<size_t>& f);

	void scattering_rate(double* results, std::vector<size_t>& k, std::vector<MatrixXcd>& rho, std::vector<std::vector<MatrixXcd>>& qk, std::vector<size_t>& i, std::vector<size_t>& f);
	void scattering_rate(double* results, std::vector<size_t>& k, std::vector<MatrixXcd>& rho, std::vector<size_t>& i, std::vector<size_t>& f);

};


double* get_f(double i, double j);
