
#include "pch.h"
#include "Physics.h"
#include "Light.h"
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

DecayMap::~DecayMap()
{
	std::vector<std::string>().swap(states_0);
	std::vector<std::string>().swap(states_1);
	std::vector<double>().swap(a);
	std::vector<std::vector<double>>().swap(ae);
	std::vector<std::vector<double>>().swap(am);
	std::vector<bool>().swap(single_leading_order);
}

void DecayMap::add_decay(std::string state_0, std::string state_1, std::vector<double> _ae, std::vector<double> _am, bool _single_leading_order)
{
	size += 1;
	states_0.push_back(state_0);
	states_1.push_back(state_1);
	ae.push_back(_ae);
	am.push_back(_am);
	single_leading_order.push_back(_single_leading_order);
}

size_t DecayMap::get_size()
{
	return size;
}

size_t DecayMap::get_k_em_max()
{
	return k_em_max;
}

void DecayMap::set_k_em_max(size_t _k_em_max)
{
	k_em_max = _k_em_max;
}

std::vector<std::string>* DecayMap::get_states_0()
{
	return &states_0;
}

std::vector<std::string>* DecayMap::get_states_1()
{
	return &states_1;
}

size_t DecayMap::get_index(std::string state_0, std::string state_1)
{
	for (size_t i = 0; i < size; ++i)
	{
		if ((state_0 == states_0[i] && state_1 == states_1[i])
			|| (state_0 == states_1[i] && state_1 == states_0[i])) return i;
	}
	return size;
}

std::vector<double>* DecayMap::get_a()
{
	return &a;
}

double DecayMap::get_a(std::string state_0, std::string state_1)
{
	for (size_t i = 0; i < size; ++i)
	{
		if ((state_0 == states_0[i] && state_1 == states_1[i])
			|| (state_0 == states_1[i] && state_1 == states_0[i])) return a.at(i);
	}

	return 0.;
}

bool DecayMap::get_single_leading_order(size_t i)
{
	if (i < size) return single_leading_order.at(i);
	return true;
}

bool DecayMap::get_single_leading_order(std::string state_0, std::string state_1)
{
	size_t i = get_index(state_0, state_1);
	return get_single_leading_order(i);
}


std::vector<std::vector<double>>* DecayMap::get_ae()
{
	return &ae;
}

std::vector<double> DecayMap::get_ae(size_t i)
{
	if (i < size) return ae.at(i);
	return std::vector<double>(1);
}

double DecayMap::get_ae(size_t i, size_t k)
{
	if (i < size && k > 0)
	{
		if (ae.at(i).size() > k - 1) return ae.at(i).at(k - 1);
		else return ae.at(i).at(0);
	}
	return 0.;
}

std::vector<double> DecayMap::get_ae(std::string state_0, std::string state_1)
{
	size_t i = get_index(state_0, state_1);
	return get_ae(i);
}

double DecayMap::get_ae(std::string state_0, std::string state_1, size_t k)
{
	size_t i = get_index(state_0, state_1);
	return get_ae(i, k);
}

std::vector<std::vector<double>>* DecayMap::get_am()
{
	return &am;
}

std::vector<double> DecayMap::get_am(size_t i)
{
	if (i < size) return am.at(i);
	return std::vector<double>(1);
}

double DecayMap::get_am(size_t i, size_t k)
{
	if (i < size && k > 0)
	{
		if (am.at(i).size() > k - 1) return am.at(i).at(k - 1);
		else return am.at(i).at(0);
	}
	return 0.;
}

std::vector<double> DecayMap::get_am(std::string state_0, std::string state_1)
{
	size_t i = get_index(state_0, state_1);
	return get_am(i);
}

double DecayMap::get_am(std::string state_0, std::string state_1, size_t k)
{
	size_t i = get_index(state_0, state_1);
	return get_am(i, k);
}

double DecayMap::get_gamma(std::string state_0, std::string state_1, bool parity_equal)
{
	double gamma = 0.;
	for (size_t i = 0; i < size; ++i)
	{
		if (states_0.at(i) == state_0 || states_0.at(i) == state_1 || states_1.at(i) == state_0 || states_1.at(i) == state_1)
		{
			if (single_leading_order.at(i)) gamma += ae.at(i).at(0);
			else
			{
				if (ae.at(i).size() > 1)
				{
					for (size_t k = 0; k < k_em_max; ++k)
					{
						if (parity_equal)
						{
							if (k % 2 != 0) gamma += ae.at(i).at(k);
						}
						else
						{
							if (k % 2 == 0) gamma += ae.at(i).at(k);
						}
					}
				}
				else gamma += ae.at(i).at(0);

				if (am.at(i).size() > 1)
				{
					for (size_t k = 0; k < k_em_max; ++k)
					{
						if (parity_equal)
						{
							if (k % 2 == 0) gamma += am.at(i).at(k);
						}
						else
						{
							if (k % 2 != 0) gamma += am.at(i).at(k);
						}
					}
				}
				else gamma += am.at(i).at(0);
			}
		}
	}
	return gamma;
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
	std::vector<MatrixXd>().swap(a_em);
	std::vector<MatrixXd>().swap(d_em);
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
	gen_frequencies(env);
	gen_w0();
	gen_multipole();
}

size_t Atom::get_size()
{
	return size;
}

std::vector<State*>* Atom::get_states()
{
	return &states;
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

bool Atom::get_parity_equal(size_t i, size_t j)
{
	return states.at(i)->get_parity() == states.at(j)->get_parity();
}

std::vector<size_t>* Atom::get_gs()
{
	return &gs;
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

Environment* Atom::get_env()
{
	return env;
}

void Atom::set_env(Environment* _env)
{
	env = _env;
}

bool Atom::is_electric(size_t k, size_t i, size_t j)
{
	bool parity_equal = get_parity_equal(i, j);

	bool electric = true;
	if ((parity_equal && k % 2 != 0) || (not parity_equal && k % 2 == 0)) electric = false;
	return electric;
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
		a_em.push_back(MatrixXd::Zero(size, size));
		d_em.push_back(MatrixXd::Zero(size, size));
		A_einst.push_back(MatrixXd::Zero(size, size));
		// L0_k.push_back(MatrixXd::Zero(size, size));
		// L1_k.push_back(MatrixXd::Zero(size, size));
	}

	L0 = MatrixXd::Zero(size, size);
	L1 = MatrixXd::Zero(size, size);
	Lsum = VectorXd::Zero(size);

	bool leading_order_e_set = false;
	bool leading_order_m_set = false;
	std::vector<size_t> leading_order_e(decays->get_size(), 1);
	std::vector<size_t> leading_order_m(decays->get_size(), 1);
	for (size_t k = 1; k <= decays->get_k_em_max(); ++k)
	{
		// printf("%zi\n", k);
		for (size_t i = 1; i < size; ++i)
		{
			for (size_t j = 0; j < i; ++j)
			{
				size_t i_decay = decays->get_index(states[i]->get_label(), states[j]->get_label());
				if (i_decay == decays->get_size()) continue;
				if (not leading_order_e_set) leading_order_e.at(i_decay) = k + 1;
				if (not leading_order_m_set) leading_order_m.at(i_decay) = k + 1;

				bool parity_equal = get_parity_equal(i, j);
				double a = 0.;

				bool electric = true;
				if ((parity_equal && k % 2 != 0) || (not parity_equal && k % 2 == 0)) electric = false;

				if (electric)
				{
					if (k > leading_order_e.at(i_decay)) continue;
					a = decays->get_ae(i_decay, k);
				}
				else
				{
					if (k > leading_order_m.at(i_decay)) continue;
					a = decays->get_am(i_decay, k);
				}

				// printf("a: %.9f\n", a);
				if (a == 0.) continue;

				size_t _i = i;
				size_t _j = j;
				if (states[i]->get_freq() > states[j]->get_freq())
				{
					_i = j;
					_j = i;
				}

				double q_val = states[_j]->get_m() - states[_i]->get_m();
				double ak = a_multipole(static_cast<double>(k), states[_i]->get_i(), states[_i]->get_j(), states[_i]->get_f(), states[_i]->get_m(),
																states[_j]->get_j(), states[_j]->get_f(), states[_j]->get_m(), q_val);  // This takes the time.
				if (ak == 0.) continue;

				if (electric) ek.at(k - 1)(_i, _j) = static_cast<int>(k);
				else mk.at(k - 1)(_i, _j) = static_cast<int>(k);
				mk.at(k - 1)(_j, _i) = mk.at(k - 1)(_i, _j);
				ek.at(k - 1)(_j, _i) = ek.at(k - 1)(_i, _j);

				// printf("%s\n", std::format("ak({}, {}): {:.0e}", _i, _j, ak).c_str());
				// L0_k.at(k - 1)(_i, _j) = a * ak * ak;
				// L0_k.at(k - 1)(_j, _i) = 0.;

				A_einst.at(k - 1)(_i, _j) = a;
				L0(_i, _j) += a * ak * ak;
				L0(_j, _i) = 0.;


				// size_t q = (4 * abs(q_val) + (sgn<double>(q_val) - abs(sgn<double>(q_val)))) / 2;
				// double q_val = 0.5 * (1 - 2 * (q % 2)) * (q + (q % 2));
				double norm = d_emk(k, parity_equal, a, states[i]->get_freq(), states[j]->get_freq());
				a_em.at(k - 1)(_i, _j) = ak;
				a_em.at(k - 1)(_j, _i) = a_em.at(k - 1)(_i, _j);
				d_em.at(k - 1)(_i, _j) = ak * norm;
				d_em.at(k - 1)(_j, _i) = d_em.at(k - 1)(_i, _j);

				if (decays->get_single_leading_order(i_decay))
				{
					leading_order_e.at(i_decay) = k;
					leading_order_m.at(i_decay) = k;
					leading_order_e_set = true;
					leading_order_m_set = true;
				}
				else
				{
					if (electric && decays->get_ae(i_decay).size() == 1)
					{
						leading_order_e.at(i_decay) = k;
						leading_order_e_set = true;
					}
					if (not electric && decays->get_am(i_decay).size() == 1)
					{
						leading_order_m.at(i_decay) = k;
						leading_order_m_set = true;
					}
				}
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

void Atom::gen_w0()
{
	w0.resize(size);
	for (size_t i = 0; i < size; ++i) w0(i) = 2 * sc::pi * get(i)->get_freq();
}

void Atom::scattering_rate(double* results, std::vector<size_t>& k, std::vector<MatrixXcd>& rho,
	std::vector<Vector3d>& k_vec, std::vector<Vector3cd>& x_vec, std::vector<size_t>& i, std::vector<size_t>& f)
{
	std::vector<std::vector<MatrixXcd>> qk(k_vec.size(), std::vector<MatrixXcd>(k.size()));

	Polarizationk kpol = Polarizationk(*env->get_e_B());
	for (size_t index_qk = 0; index_qk < k_vec.size(); ++index_qk)
	{
		for (size_t index_k = 0; index_k < k.size(); ++index_k)
		{
			size_t _k = k.at(index_k);

			kpol.init_qk(x_vec.at(index_qk), k_vec.at(index_qk));
			qk.at(index_qk).at(index_k) = MatrixXcd::Zero(2 * _k + 1, 2),
			qk.at(index_qk).at(index_k).col(0) = spherical_tensor(true, _k, *kpol.get_qk(), kpol.get_theta_k(), kpol.get_phi_k());
			qk.at(index_qk).at(index_k).col(1) = spherical_tensor(false, _k, *kpol.get_qk(), kpol.get_theta_k(), kpol.get_phi_k());
		}
	}

	scattering_rate(results, k, rho, qk, i, f);
}

void Atom::scattering_rate(double* results, std::vector<size_t>& k, std::vector<MatrixXcd>& rho,
	std::vector<Vector3d>& k_vec, std::vector<size_t>& i, std::vector<size_t>& f)
{
	std::vector<std::vector<MatrixXcd>> qk_0(k_vec.size(), std::vector<MatrixXcd>(k.size()));
	std::vector<std::vector<MatrixXcd>> qk_1(k_vec.size(), std::vector<MatrixXcd>(k.size()));

	Polarizationk kpol = Polarizationk(*env->get_e_B());
	for (size_t index_qk = 0; index_qk < k_vec.size(); ++index_qk)
	{
		Vector3d _k_vec = k_vec.at(index_qk);
		// printf("k_vec: %.3f, %.3f, %.3f\n", _k_vec(0), _k_vec(1), _k_vec(2));

		Vector3d x = Vector3d::Zero();
		x(0) = _k_vec(1) - _k_vec(2);
		x(1) = _k_vec(2) - _k_vec(0);
		x(2) = _k_vec(0) - _k_vec(1);
		x /= x.norm();

		Vector3d y = _k_vec.cross(x);
		y /= y.norm();

		for (size_t index_k = 0; index_k < k.size(); ++index_k)
		{
			size_t _k = k.at(index_k);

			kpol.init_qk(x, _k_vec);
			qk_0.at(index_qk).at(index_k) = MatrixXcd::Zero(2 * _k + 1, 2),
			qk_0.at(index_qk).at(index_k).col(0) = spherical_tensor(true, _k, *kpol.get_qk(), kpol.get_theta_k(), kpol.get_phi_k());
			qk_0.at(index_qk).at(index_k).col(1) = spherical_tensor(false, _k, *kpol.get_qk(), kpol.get_theta_k(), kpol.get_phi_k());

			kpol.init_qk(y, _k_vec);
			qk_1.at(index_qk).at(index_k) = MatrixXcd::Zero(2 * _k + 1, 2),
			qk_1.at(index_qk).at(index_k).col(0) = spherical_tensor(true, _k, *kpol.get_qk(), kpol.get_theta_k(), kpol.get_phi_k());
			qk_1.at(index_qk).at(index_k).col(1) = spherical_tensor(false, _k, *kpol.get_qk(), kpol.get_theta_k(), kpol.get_phi_k());
		}
	}

	scattering_rate(results, k, rho, qk_0, i, f);
	// printf("--------------------------------\n");
	scattering_rate(results, k, rho, qk_1, i, f);
}

void Atom::scattering_rate(double* results, std::vector<size_t>& k, std::vector<MatrixXcd>& rho,
	std::vector<std::vector<MatrixXcd>>& qk, std::vector<size_t>& i, std::vector<size_t>& f)
{
	std::vector<size_t> indexes(rho.size() * qk.size());
	std::iota(indexes.begin(), indexes.end(), 0);

	// printf("qk: %.3f + %.3fi, %.3f + %.3fi, %.3f + %.3fi\n", qk.at(0).at(0)(0, 1).real(), qk.at(0).at(0)(0, 1).imag(), qk.at(0).at(0)(1, 1).real(), qk.at(0).at(0)(1, 1).imag(), qk.at(0).at(0)(2, 1).real(), qk.at(0).at(0)(2, 1).imag());
	std::for_each(std::execution::par_unseq, indexes.begin(), indexes.end(),
		[this, results, rho, qk, k, i, f](size_t index)
		{
			size_t index_qk = index / rho.size();
			size_t index_rho = index % rho.size();
			// printf("%zi, %zi, %zi, %zi\n", index_qk, index_rho, qk.size(), rho.size());


			std::complex<double> _result = 0.;
			for (size_t _f : f)
			{
				for (size_t _i : i)
				{
					for (size_t index_ki = 0; index_ki < k.size(); ++index_ki)
					{
						size_t _ki = k.at(index_ki);
						double ki_double = static_cast<double>(_ki);
						size_t i_el = (ek.at(_ki - 1)(_f, _i) >= mk.at(_ki - 1)(_f, _i)) ? 0 : 1;

						double delta_mi = states.at(_i)->get_m() - states.at(_f)->get_m();
						if (std::abs(delta_mi) > ki_double) continue;

						size_t i_qk = static_cast<size_t>(delta_mi + ki_double);
						std::complex<double> qk_i = qk.at(index_qk).at(index_ki)(i_qk, i_el);

						double a_if = A_einst.at(_ki - 1)(_f, _i);
						double d_fi = a_em.at(_ki - 1)(_f, _i);
						if (a_if * d_fi == 0.) continue;

						if (rho.at(index_rho).outerSize() == 1)
						{
							std::complex<double> y = a_if;
							// printf("y0: %s\n", std::format("{:.3e} + i{:.3e}", y.real(), y.imag()).c_str());
							y *= rho.at(index_rho)(_i);
							// printf("y1: %s\n", std::format("{:.3e} + i{:.3e}", y.real(), y.imag()).c_str());
							y *= pow(std::abs(d_fi * qk_i), 2);
							// printf("y2: %s\n", std::format("{:.3e} + i{:.3e}", y.real(), y.imag()).c_str());
							y *= (2 * ki_double + 1) / (8. * sc::pi);

							_result += y;

						}
						else
						{
							for (size_t _j : i)
							{
								for (size_t index_kj = 0; index_kj < k.size(); ++index_kj)
								{
									size_t _kj = k.at(index_kj);
									double kj_double = static_cast<double>(_kj);
									size_t j_el = (ek.at(_ki - 1)(_f, _j) >= mk.at(_ki - 1)(_f, _j)) ? 0 : 1;

									double delta_mj = states.at(_j)->get_m() - states.at(_f)->get_m();
									if (std::abs(delta_mj) > kj_double) continue;

									size_t j_qk = static_cast<size_t>(delta_mj + kj_double);
									std::complex<double> qk_j = qk.at(index_qk).at(index_kj)(j_qk, j_el);

									double a_jf = A_einst.at(_kj - 1)(_f, _j);
									double d_jf = a_em.at(_kj - 1)(_j, _f);
									if (a_jf * d_jf == 0.) continue;

									std::complex<double> y = sqrt(a_if * a_jf);  // *pow(-1., static_cast<double>(j_qk) - static_cast<double>(k))* pow(-1., static_cast<double>(i_qk) - static_cast<double>(k)); // * (2 * states.at(_f)->get_i() + 1) * (2 * states.at(_f)->get_j() + 1);
									// printf("y0: %s\n", std::format("{:.3e} + i{:.3e}", y.real(), y.imag()).c_str());
									y *= rho.at(index_rho)(_j, _i);
									// if (index_rho == 201) printf("rho: %s\n", std::format("({}, {}): {:.3e} + {:.3e}i", _i, _j, rho.at(index_rho)(_j, _i).real(), rho.at(index_rho)(_j, _i).imag()).c_str());
									// printf("y1: %s\n", std::format("{:.3e} + i{:.3e}", y.real(), y.imag()).c_str());
									y *= qk_i * d_fi * std::conj(qk_j) * d_jf;
									// printf("y2: %s\n", std::format("{:.3e} + i{:.3e}", y.real(), y.imag()).c_str());
									y *= sqrt((2 * ki_double + 1) * (2 * kj_double + 1)) / (8. * sc::pi);
									// printf("y3: %s\n", std::format("{:.3e} + i{:.3e}", y.real(), y.imag()).c_str());

									_result += y;
								}
							}
						}
					}
				}
			}
			// printf("%s\n", std::format("r{}: {:.3e} + {:.3e}i", index_rho, _result.real(), _result.imag()).c_str());
			results[index] += _result.real();
		}
	);
}

void Atom::scattering_rate(double* results, std::vector<size_t>& k, std::vector<MatrixXcd>& rho,
	std::vector<size_t>& i, std::vector<size_t>& f)
{
	size_t index = 0;
	for (MatrixXcd _rho : rho)
	{
		double _result = 0.;
		for (size_t _f : f)
		{
			for (size_t _i : i)
			{
				for (size_t index_k = 0; index_k < k.size(); ++index_k)
				{
					size_t _k = k.at(index_k);
					double k_double = static_cast<double>(_k);

					double delta_mi = states.at(_i)->get_m() - states.at(_f)->get_m();
					if (std::abs(delta_mi) > k_double) continue;

					if (_rho.outerSize() == 1) _result += L0(_f, _i) * _rho(_i).real();
					else _result += L0(_f, _i) * _rho(_i, _i).real();
				}
			}
		}

		results[index] += _result;
		index += 1;
	}
}

size_t Atom::get_min_k(size_t i, size_t j)
{
	State& s0 = *states.at(i);
	State& s1 = *states.at(j);
	bool parity_equal = get_parity_equal(i, j);
	size_t ne = s0.get_l().size();

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
