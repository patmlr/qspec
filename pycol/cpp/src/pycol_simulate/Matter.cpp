
#include "pch.h"
#include "Matter.h"
#include "Physics.h"


State::State()
{
	freq_j = 0.;
	freq = 0.;

	s = 0.;
	l = 0.;
	j = 0.;
	i = 0.;
	f = 0.;
	m = 0.;

	hyper_const = new double[HYPER_SIZE]{0., 0., 0.};
	g = 0.;

	label = std::string ("<State>");
}

State::~State()
{
	delete[] hyper_const;
}

void State::init(double _freq_j, double _s, double _l, double _j, double _i, double _f, double _m,
	double* _hyper_const, double _g, std::string _label)
{
	freq_j = _freq_j;
	freq = _freq_j;

	s = _s;
	l = _l;
	j = _j;
	i = _i;
	f = _f;
	m = _m;

	for (int i = 0; i < HYPER_SIZE; ++i)
	{
		hyper_const[i] = _hyper_const[i];
	}
	g = _g;

	label = _label;

	update();
}

void State::update()
{
	freq = freq_j + hyper_zeeman(i, s, l, j, f, m, g, hyper_const, 0, false);
}

void State::update(Environment* env)
{
	double B = 0;
	double E = 0;
	if (env)
	{
		B = env->B;
		E = env->E;
	}
	freq = freq_j + hyper_zeeman(i, s, l, j, f, m, g, hyper_const, B, false);
}

double State::get_shift()
{
	return freq - freq_j;
}

double State::get_freq_j()
{
	return freq_j;
}

void State::set_freq_j(double _freq_j)
{
	freq_j = _freq_j;
}

double State::get_freq()
{
	return freq;
}

double State::get_s()
{
	return s;
}

void State::set_s(double _s)
{
	s = _s;
}

double State::get_l()
{
	return l;
}

void State::set_l(double _l)
{
	l = _l;
}

double State::get_j()
{
	return j;
}

void State::set_j(double _j)
{
	j = _j;
}

double State::get_i()
{
	return i;
}

void State::set_i(double _i)
{
	i = _i;
}

double State::get_f()
{
	return f;
}

void State::set_f(double _f)
{
	f = _f;
}

double State::get_m()
{
	return m;
}

void State::set_m(double _m)
{
	m = _m;
}

double* State::get_hyper_const()
{
	return hyper_const;
}

void State::set_hyper_const(double* _hyper_const)
{
	for (size_t i = 0; i < HYPER_SIZE; ++i)
	{
		hyper_const[i] = _hyper_const[i];
	}
}

double State::get_g()
{
	return g;
}

void State::set_g(double _g)
{
	g = _g;
}

std::string State::get_label()
{
	return label;
}

void State::set_label(std::string _label)
{
	label = _label;
}


DecayMap::DecayMap()
{
	size = 0;
}

DecayMap::DecayMap(std::vector<std::string> _states_0, std::vector<std::string> _states_1, std::vector<double> _a)
{
	size = _a.size();
	states_0 = _states_0;
	states_1 = _states_1;
	a = _a;
}

DecayMap::~DecayMap()
{
	std::vector<std::string>().swap(states_0);
	std::vector<std::string>().swap(states_1);
	std::vector<double>().swap(a);
}

void DecayMap::add_decay(std::string state_0, std::string state_1, double _a)
{
	size += 1;
	states_0.push_back(state_0);
	states_1.push_back(state_1);
	a.push_back(_a);
}

size_t DecayMap::get_size()
{
	return size;
}

std::vector<std::string>* DecayMap::get_states_0()
{
	return &states_0;
}

std::vector<std::string>* DecayMap::get_states_1()
{
	return &states_1;
}

std::vector<double>* DecayMap::get_a()
{
	return &a;
}

double DecayMap::get_item(std::string state_0, std::string state_1)
{
	for (size_t i = 0; i < size; ++i)
	{
		if ((state_0 == states_0[i] && state_1 == states_1[i]) 
			|| (state_0 == states_1[i] && state_1 == states_0[i])) return a[i];
	}
	return 0.;
}


Atom::Atom()
{
	size = 0;
	decays = new DecayMap();
	mass = 0;
}

Atom::~Atom()
{
	std::vector<State*>().swap(states);
	for (size_t q = 0; q < 3; ++q)
	{
		m_dipole.at(q).resize(0, 0);
	}
}

void Atom::init(std::vector<State*> _states, DecayMap* _decays)
{
	size = _states.size();
	states = _states;
	decays = _decays;
	update();
}

void Atom::update()
{
	L0 = MatrixXd::Zero(size, size);
	L1 = MatrixXd::Zero(size, size);
	Lsum = VectorXd::Zero(size);
	for (size_t q = 0; q < 3; ++q)
	{
		m_dipole.at(q) = MatrixXd::Zero(size, size);
	}

	size_t _i = 0;
	size_t _j = 0;
	for (size_t i = 1; i < size; ++i)
	{
		for (size_t j = 0; j < i; ++j)
		{
			_i = i;
			_j = j;
			if (states[i]->get_freq() > states[j]->get_freq())
			{
				_i = j;
				_j = i;
			}

			double a = decays->get_item(states[i]->get_label(), states[j]->get_label());
			L0(_i, _j) = a * pow(a_dipole(states[_i]->get_i(), states[_i]->get_j(), states[_i]->get_f(), states[_i]->get_m(),
				states[_j]->get_j(), states[_j]->get_f(), states[_j]->get_m(), states[_j]->get_m() - states[_i]->get_m()), 2);
			L0(_j, _i) = 0.;
			if (states[i]->get_freq() == states[j]->get_freq()) continue;

			double norm = j_dipole(a, states[i]->get_freq(), states[j]->get_freq());
			for (size_t q = 0; q < 3; ++q)
			{
				double q_val = static_cast<double>(q) - 1;
				m_dipole.at(q)(i, j) = norm * a_dipole(states[_i]->get_i(), states[_i]->get_j(), states[_i]->get_f(), states[_i]->get_m(),
					states[_j]->get_j(), states[_j]->get_f(), states[_j]->get_m(), q_val);
				m_dipole.at(q)(j, i) = m_dipole.at(q)(i, j);
			}
		}
	}
	Lsum = L0.colwise().sum();
	for (size_t i = 0; i < size; ++i)
	{
		L1.row(i) += Lsum;
		L1.col(i) += Lsum;
	}
	L1 *= -0.5;
	gen_w0();
}

size_t Atom::get_size()
{
	return size;
}

void Atom::add_state(State* state)
{
	size += 1;
	states.push_back(state);
	if (state->get_label() == states.at(0)->get_label()) gs.push_back(size - 1);
}

void Atom::clear_states()
{
	states.clear();
	gs.clear();
	size = 0;
}

void Atom::gen_w0()
{
	w0.resize(size);
	for (size_t i = 0; i < size; ++i) w0(i) = 2 * sc::pi * get(i)->get_freq();
}

DecayMap* Atom::get_decay_map()
{
	return decays;
}

void Atom::set_decay_map(DecayMap* _decays)
{
	decays = _decays;
}

double Atom::get_mass()
{
	return mass;
}

void Atom::set_mass(double _mass)
{
	mass = _mass;
}

std::vector<size_t>* Atom::get_gs()
{
	return &gs;
}

std::array<MatrixXd, 3>* Atom::get_m_dipole()
{
	return &m_dipole;
}

VectorXd* Atom::get_w0()
{
	return &w0;
}

VectorXd* Atom::get_Lsum()
{
	return &Lsum;
}

MatrixXd* Atom::get_L0()
{
	return &L0;
}

MatrixXd* Atom::get_L1()
{
	return &L1;
}

State* Atom::get(size_t index)
{
	return states[index];
}

Vector3d Atom::gen_velocity_change(size_t i, size_t j, size_t f, Vector3d& k, std::mt19937& gen)
{
	std::normal_distribution<double> d(0, 1);
	Vector3d e_r;
	e_r << d(gen), d(gen), d(gen);
	e_r /= e_r.norm();

	double freq_up = states.at(j)->get_freq() - states.at(i)->get_freq();
	double freq_down = states.at(j)->get_freq() - states.at(f)->get_freq();

	return sc::h / (mass * sc::amu * sc::c) * (freq_up * k - freq_down * e_r) * 1e6;
}


double* get_f(double i, double j)
{
	double f_min = abs(i - j);
	int size = static_cast<int>(i + j - f_min + 1);
	double* f = new double[size];
	for (int k = 0; k < size; ++k) {
		f[k] = f_min + k;
	}
	return f;
}
