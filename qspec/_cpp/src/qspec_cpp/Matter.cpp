
#include "pch.h"
#include "Physics.h"
#include "Matter.h"


Environment::Environment()
{
	E = 0.;
	B = 0.;
	e_E << 1, 0, 0;
	e_B << 0, 0, 1;
}

Environment::~Environment()
{

}

double Environment::get_E()
{
	return E;
}

double Environment::get_B()
{
	return B;
}

Vector3d* Environment::get_e_E()
{
	return &e_E;
}

Vector3d* Environment::get_e_B()
{
	return &e_B;
}

void Environment::set_E(double _E)
{
	E = _E;
}

void Environment::set_E(Vector3d _E)
{
	E = _E.norm();
	if (E == 0)
	{
		e_E << 1, 0, 0;
	}
	else
	{
		e_E = _E / E;
	}
}

void Environment::set_B(double _B)
{
	B = _B;
}

void Environment::set_B(Vector3d _B)
{
	B = _B.norm();
	if (B == 0)
	{
		e_B << 0, 0, 1;
	}
	else
	{
		e_B = _B / B;
	}
}


State::State()
{
	freq_j = 0.;
	freq = 0.;

	j = 0.;
	i = 0.;
	f = 0.;
	m = 0.;

	hyper_const = new double[HYPER_SIZE]{0., 0., 0.};

	gj = 0.;
	gi = 0.;

	label = std::string("<State>");
}

State::~State()
{
	delete[] hyper_const;
}

void State::init(double _freq_j, double _j, double _i, double _f, double _m, bool _parity,
				 std::vector<double> _s, std::vector<double> _l, std::vector<double> _jj,
				 double* _hyper_const, double _gj, double _gi, std::string _label)
{
	freq_j = _freq_j;
	freq = _freq_j;

	j = _j;
	i = _i;
	f = _f;
	m = _m;
	parity = _parity;
	s = _s;
	l = _l;
	jj = _jj;

	for (int i = 0; i < HYPER_SIZE; ++i)
	{
		hyper_const[i] = _hyper_const[i];
	}
	gj = _gj;
	gi = _gi;

	label = _label;

	reset();
}

void State::reset()
{
	freq = freq_j + hyperfine(i, j, f, hyper_const);
}

double State::get_shift()
{
	return freq - freq_j;
}

void State::set_shift(double _shift)
{
	freq = freq_j + _shift;
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

void State::set_freq(double _freq)
{
	freq = _freq;
}

std::vector<double> State::get_s()
{
	return s;
}

void State::set_s(std::vector<double> _s)
{
	s = _s;
}

std::vector<double> State::get_l()
{
	return l;
}

void State::set_l(std::vector<double> _l)
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

bool State::get_parity()
{
	return parity;
}

void State::set_parity(bool _parity)
{
	parity = _parity;
}

std::vector<double> State::get_jj()
{
	return jj;
}

void State::set_jj(std::vector<double> _jj)
{
	jj = _jj;
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

double State::get_gj()
{
	return gj;
}

void State::set_gj(double _gj)
{
	gj = _gj;
}

double State::get_gi()
{
	return gi;
}

void State::set_gi(double _gi)
{
	gi = _gi;
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

DecayMap::DecayMap(size_t _k_em_max)
{
	size = 0;
	k_em_max = _k_em_max;
}

DecayMap::DecayMap(std::vector<std::string> _states_0, std::vector<std::string> _states_1, std::vector<double> _a, size_t _k_em_max)
{
	size = _a.size();
	states_0 = _states_0;
	states_1 = _states_1;
	a = _a;
	k_em_max = _k_em_max;
}

DecayMap::~DecayMap()
{
	std::vector<std::string>().swap(states_0);
	std::vector<std::string>().swap(states_1);
	std::vector<double>().swap(a);
}

void DecayMap::add_decay(std::string state_0, std::string state_1, std::vector<double> _ae, std::vector<double> _am)
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
			|| (state_0 == states_1[i] && state_1 == states_0[i])) return a.at(i);
	}

	return 0.;
}

double DecayMap::get_gamma(std::string state_0, std::string state_1)
{
	double gamma = 0;
	for (size_t i = 0; i < size; ++i)
	{
		if (states_0.at(i) == state_0 || states_0.at(i) == state_1 || states_1.at(i) == state_0 || states_1.at(i) == state_1)
		{
			gamma += a.at(i);
			continue;
		}
	}
	return gamma;
}

size_t DecayMap::get_k_em_max()
{
	return k_em_max;
}

void DecayMap::set_k_em_max(size_t _k_em_max)
{
	k_em_max = _k_em_max;
}


Atom::Atom()
{
	decays = new DecayMap();
	env = new Environment();
}

Atom::~Atom()
{
	std::vector<State*>().swap(states);
	std::vector<MatrixXi>().swap(ek);
	std::vector<MatrixXi>().swap(mk);
	std::vector<MatrixXd>().swap(d_em);
	for (size_t q = 0; q < 3; ++q)
	{
		m_e1.at(q).resize(0, 0);
		m_m1.at(q).resize(0, 0);
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
	// gen_dipole();
	gen_frequencies(env);
	gen_w0();
	gen_multipole();
}


Environment* Atom::get_env()
{
	return env;
}

void Atom::set_env(Environment* _env)
{
	env = _env;
}

void Atom::gen_multipole()
{
	ek.clear();
	mk.clear();
	d_em.clear();
	for (size_t ik = 0; ik < decays->get_k_em_max(); ++ik)
	{
		ek.push_back(MatrixXi::Zero(size, size));
		mk.push_back(MatrixXi::Zero(size, size));
		d_em.push_back(MatrixXd::Zero(size, size));
	}

	for (size_t q = 0; q < 3; ++q)  // Deprecated.
	{
		m_e1.at(q) = MatrixXd::Zero(size, size);
		m_m1.at(q) = MatrixXd::Zero(size, size);
	}

	L0 = MatrixXd::Zero(size, size);
	L1 = MatrixXd::Zero(size, size);
	Lsum = VectorXd::Zero(size);

	for (size_t i = 1; i < size; ++i)
	{
		for (size_t j = 0; j < i; ++j)
		{
			double a = decays->get_item(states[i]->get_label(), states[j]->get_label());
			if (a == 0) continue;

			size_t _i = i;
			size_t _j = j;
			if (states[i]->get_freq() > states[j]->get_freq())
			{
				_i = j;
				_j = i;
			}

			size_t k = 1;  // get_min_k(_i, _j);
			// printf("k(%zi, %zi): %zi\n", _i, _j, k);
			if (k == 0) continue;
			bool parity_equal = get_parity_equal(_i, _j);

			while (k <= decays->get_k_em_max())
			{
				double q_val = states[_j]->get_m() - states[_i]->get_m();
				double ak = a_multipole(static_cast<double>(k), states[_i]->get_i(), states[_i]->get_j(), states[_i]->get_f(), states[_i]->get_m(),
																states[_j]->get_j(), states[_j]->get_f(), states[_j]->get_m(), q_val);  // This takes the time.
				if (ak == 0.)
				{
					k += 1;
					continue;
				}

				if (parity_equal)
				{
					if (k % 2 != 0) mk.at(k - 1)(_i, _j) = static_cast<int>(k);
					else ek.at(k - 1)(_i, _j) = static_cast<int>(k);
				}
				else
				{
					if (k % 2 != 0) ek.at(k - 1)(_i, _j) = static_cast<int>(k);
					else mk.at(k - 1)(_i, _j) = static_cast<int>(k);
				}
				mk.at(k - 1)(_j, _i) = mk.at(k - 1)(_i, _j);
				ek.at(k - 1)(_j, _i) = ek.at(k - 1)(_i, _j);

				// if (k == 1) printf("%s\n", std::format("ak({}, {}): {:.0e}", _i, _j, ak).c_str());
				L0(_i, _j) = a * ak * ak;
				L0(_j, _i) = 0.;

				// size_t q = (4 * abs(q_val) + (sgn<double>(q_val) - abs(sgn<double>(q_val)))) / 2;
				// double q_val = 0.5 * (1 - 2 * (q % 2)) * (q + (q % 2));
				double norm = d_emk(k, parity_equal, a, states[i]->get_freq(), states[j]->get_freq());
				d_em.at(k - 1)(i, j) = ak * norm;
				d_em.at(k - 1)(j, i) = d_em.at(k - 1)(i, j);

				// k += 1;  // Use only leading order for now.
				k = decays->get_k_em_max() + 1;
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
}

void Atom::gen_dipole()
{
	L0 = MatrixXd::Zero(size, size);
	L1 = MatrixXd::Zero(size, size);
	Lsum = VectorXd::Zero(size);
	for (size_t q = 0; q < 3; ++q)
	{
		m_e1.at(q) = MatrixXd::Zero(size, size);
		m_m1.at(q) = MatrixXd::Zero(size, size);
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

			// Calculate (F, m) reduction of dipole (e1 and m1) moments.
			double a = decays->get_item(states[i]->get_label(), states[j]->get_label());
			double a1 = 0.;
			if (abs(states[_i]->get_j() - states[_j]->get_j()) < 1.1 
				&& abs(states[_i]->get_f() - states[_j]->get_f()) < 1.1
				&& abs(states[_i]->get_m() - states[_j]->get_m()) < 1.1)  // Check dipole condition before calling a_dipole.
			{
				a1 = a_dipole(states[_i]->get_i(), states[_i]->get_j(), states[_i]->get_f(), states[_i]->get_m(),
					states[_j]->get_j(), states[_j]->get_f(), states[_j]->get_m(), states[_j]->get_m() - states[_i]->get_m());  // This takes the time.
			}

			L0(_i, _j) = a * a1 * a1;
			L0(_j, _i) = 0.;

			if (states[i]->get_freq() == states[j]->get_freq()) continue;

			size_t q = 1;
			if (states[_j]->get_m() - states[_i]->get_m() < 0) q = 0;
			else if (states[_j]->get_m() - states[_i]->get_m() > 0) q = 2;
			
			if (states[i]->get_parity() != states[j]->get_parity())
			{
				double norm = d_e1(a, states[i]->get_freq(), states[j]->get_freq());
				m_e1.at(q)(i, j) = norm * a1;
				m_e1.at(q)(j, i) = m_e1.at(q)(i, j);
			}
			else
			{
				double norm = d_m1(a, states[i]->get_freq(), states[j]->get_freq());
				m_m1.at(q)(i, j) = norm * a1;
				m_m1.at(q)(j, i) = m_m1.at(q)(i, j);
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
}

void Atom::gen_frequencies(Environment* _env)
{
	std::set<size_t> done;
	for (size_t k = 0; k < size; ++k)
	{
		if (done.count(k)) continue;

		State& s = *states.at(k);
		if (_env->get_B() == 0)
		{
			s.reset();
			done.insert(k);
			continue;
		}

		std::vector<double> freqs = hyper_zeeman_num(s.get_i(), s.get_j(), s.get_m(), s.get_gj(), s.get_gi(), s.get_hyper_const(), _env->get_B());

		double f_min = max(abs(s.get_m()), abs(s.get_i() - s.get_j()));
		size_t i = static_cast<size_t>(s.get_f() - f_min);
		s.set_shift(freqs.at(i));
		done.insert(k);

		// The below section can be omitted. But it should increase the execution speed by using the already found eigenvalues for the other states.
		for (size_t l = 0; l < size; ++l)
		{
			if (done.count(l)) continue;

			State& s_mix = *states.at(l);
			if (s_mix.get_i() == s.get_i() && s_mix.get_j() == s.get_j() && s_mix.get_m() == s.get_m() && s_mix.get_freq_j() == s.get_freq_j())
			{
				size_t i = static_cast<size_t>(s_mix.get_f() - f_min);
				s_mix.set_shift(freqs.at(i));
				done.insert(l);
			}
		}
	}
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

std::vector<State*>* Atom::get_states()
{
	return &states;
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

size_t Atom::get_min_k(size_t i, size_t j)
{
	State& s0 = *states.at(i);
	State& s1 = *states.at(j);
	bool parity_equal = get_parity_equal(i, j);
	size_t ne = s0.get_l().size();

	size_t dj = s1.get_j() - s0.get_j();
	size_t ds = 0;
	size_t dl = 0;
	size_t n = 0;
	for (size_t ie = 0; ie < ne; ++ie)
	{
		size_t _ds = static_cast<size_t>(abs(s1.get_s().at(ie) - s0.get_s().at(ie)));
		size_t _dl = static_cast<size_t>(abs(s1.get_l().at(ie) - s0.get_l().at(ie)));
		if (_ds > 0 || _dl > 0) ++n;
		ds += _ds;
		dl += _dl;
	}

	// printf("%d %zi [dl, ds](%zi, %zi): [%zi, %zi]\n", parity_equal, n, i, j, dl, ds);
	if (n > 1 || ds > 1) return 0;
	else
	{
		if (ne == 1 && s0.get_s().at(0) == 0.5)
		{
			if (parity_equal)
			{
				if (dl % 2 != 0) return 0;
			}
			else
			{
				if (dl % 2 == 0) return 0;
			}
			return max(1, dl);
		}
		else
		{
			if (ds == 0)
			{
				if (parity_equal && dl == 1) return 2;
				// printf("%zi, %zi", dl, max(1, dl));
				return max(1, dl);
			}
			else
			{
				if (dl < 3) return 1;
				else
				{
					if (parity_equal)
					{
						if (dl % 2 != 0) return dl - 1;
					}
					else
					{
						if (dl % 2 == 0) return dl - 1;
					}
					return dl;
				}
			}
		}
	}
	return 0;
}

bool Atom::get_parity_equal(size_t i, size_t j)
{
	return states.at(i)->get_parity() == states.at(j)->get_parity();
}

std::vector<size_t>* Atom::get_gs()
{
	return &gs;
}

std::array<MatrixXd, 3>* Atom::get_m_e1()
{
	return &m_e1;
}

std::array<MatrixXd, 3>* Atom::get_m_m1()
{
	return &m_m1;
}

std::vector<MatrixXi> Atom::get_ek()
{
	return ek;
}

MatrixXi Atom::get_ek(size_t k)
{
	return ek.at(k - 1);
}

size_t Atom::get_ek(size_t k, size_t i, size_t j)
{
	return ek.at(k - 1)(i, j);
}

std::vector<MatrixXi> Atom::get_mk()
{
	return mk;
}

MatrixXi Atom::get_mk(size_t k)
{
	return mk.at(k - 1);
}

size_t Atom::get_mk(size_t k, size_t i, size_t j)
{
	return mk.at(k - 1)(i, j);
}

std::vector<MatrixXi> Atom::get_emk()
{
	std::vector<MatrixXi> emk;
	for (size_t ik = 0; ik < decays->get_k_em_max(); ++ik)
	{
		emk.push_back(ek.at(ik) + mk.at(ik));
	}
	return emk;
}

MatrixXi Atom::get_emk(size_t k)
{
	return ek.at(k - 1) + mk.at(k - 1);
}

size_t Atom::get_emk(size_t k, size_t i, size_t j)
{
	return ek.at(k - 1)(i, j) + mk.at(k - 1)(i, j);
}

std::vector<MatrixXd> Atom::get_d_em()
{
	return d_em;
}

MatrixXd Atom::get_d_em(size_t k)
{
	return d_em.at(k - 1);
}

double Atom::get_d_em(size_t k, size_t i, size_t j)
{
	return d_em.at(k - 1)(i, j);
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
