
#include "pch.h"
#include "Interaction.h"

struct push_back_state_and_time
{
	std::vector< VectorXd >& m_states;
	std::vector< double >& m_times;

	push_back_state_and_time(std::vector< VectorXd >& states, std::vector< double >& times)
		: m_states(states), m_times(times) { }

	void operator()(const VectorXd& x, double t)
	{
		m_states.push_back(x);
		m_times.push_back(t);
	}

	void operator()(const VectorXcd& x, double t)
	{
		m_states.push_back(x.cwiseAbs2());
		m_times.push_back(t);
	}

	void operator()(const MatrixXcd& x, double t)
	{
		m_states.push_back(x.diagonal().real());
		m_times.push_back(t);
	}
};

struct f_rate_equations
{
	MatrixXd& R;
	VectorXd& R_sum;
	MatrixXd& L0;
	VectorXd& L_sum;

	f_rate_equations(MatrixXd& _R, VectorXd& _R_sum, MatrixXd& _L0, VectorXd& _L_sum)
		: R(_R), R_sum(_R_sum), L0(_L0), L_sum(_L_sum) { }

	void operator()(const VectorXd& x, VectorXd& dxdt, double t)
	{
		dxdt = (R + L0) * x - (R_sum + L_sum).cwiseProduct(x);
	}
};

struct f_schroedinger
{
	MatrixXcd& H;

	f_schroedinger(MatrixXcd& _H) : H(_H) { }

	void operator()(const VectorXcd& x, VectorXcd& dxdt, double t)
	{
		dxdt = -sc::i * (H * x);
	}
};

struct f_schroedinger_t
{
	MatrixXcd& H;
	VectorXd& w0;
	VectorXd& w;
	Interaction& interaction;

	f_schroedinger_t(MatrixXcd& _H, VectorXd& _w0, VectorXd& _w, Interaction& _interaction)
		: H(_H), w0(_w0), w(_w), interaction(_interaction) { }

	void operator()(const VectorXcd& x, VectorXcd& dxdt, double t)
	{
		interaction.update_hamiltonian(H, w0, w, t);
		dxdt = -sc::i * (H * x);
	}
};

struct f_schroedinger_leaky_t
{
	MatrixXcd& H;
	VectorXd& w0;
	VectorXd& w;
	Interaction& interaction;

	f_schroedinger_leaky_t(MatrixXcd& _H, VectorXd& _w0, VectorXd& _w, Interaction& _interaction)
		: H(_H), w0(_w0), w(_w), interaction(_interaction) { }

	void operator()(const VectorXcd& x, VectorXcd& dxdt, double t)
	{
		interaction.update_hamiltonian_leaky(H, w0, w, t);
		dxdt = -sc::i * (H * x);
	}
};

struct f_master
{
	MatrixXcd& H;
	MatrixXd& L0;
	MatrixXd& L1;
	// size_t size;
	// std::complex<double> sum;

	f_master(MatrixXcd& _H, MatrixXd& _L0, MatrixXd& _L1) : H(_H), L0(_L0), L1(_L1) { }

	void operator()(const MatrixXcd& x, MatrixXcd& dxdt, double t)
	{

		dxdt = -sc::i * (H * x - x * H) + L1.cwiseProduct(x);
		dxdt.diagonal() += L0 * x.diagonal();

		/*size = H.outerSize();
		dxdt.fill(0);
		for (size_t j = 0; j < size; ++j)
		{
			for (size_t i = 0; i < j; ++i)
			{
				sum = 0;
				for (size_t k = 0; k < size; ++k)
				{
					sum += -sc::i * (std::conj(H(k, i)) * x(k, j) - std::conj(x(k, i)) * H(k, j));
				}
				dxdt(i, j) = sum + L1(i, j) * x(i, j);
				dxdt(j, i) = std::conj(dxdt(i, j));
			}
			sum = 0;
			for (size_t k = 0; k < size; ++k)
			{
				sum += -sc::i * (std::conj(H(k, j)) * x(k, j) - std::conj(x(k, j)) * H(k, j)) + L0(j, k) * x(k, k);
			}
			dxdt(j, j) = sum + L1(j, j) * x(j, j);
		}*/
	}
};

struct f_master_t
{
	MatrixXcd& H;
	MatrixXd& L0;
	MatrixXd& L1;
	VectorXd& w0;
	VectorXd& w;
	Interaction& interaction;

	f_master_t(MatrixXcd& _H, MatrixXd& _L0, MatrixXd& _L1, VectorXd& _w0, VectorXd& _w, Interaction& _interaction) 
		: H(_H), L0(_L0), L1(_L1), w0(_w0), w(_w), interaction(_interaction) { }

	void operator()(const MatrixXcd& x, MatrixXcd& dxdt, double t)
	{
		interaction.update_hamiltonian(H, w0, w, t);
		dxdt = -sc::i * (H * x - x * H) + L1.cwiseProduct(x);
		dxdt.diagonal() += L0 * x.diagonal();
	}
};


Interaction::Interaction()
{
	atom = nullptr;
	lasers = std::vector<Laser*>();
	env = new Environment();
}

Atom* Interaction::get_atom()
{
	return atom;
}

void Interaction::set_atom(Atom* _atom)
{
	atom = _atom;
}

void Interaction::clear_lasers()
{
	lasers.clear();
}

std::vector<Laser*>* Interaction::get_lasers()
{
	return &lasers;
}

void Interaction::add_laser(Laser* _laser)
{
	lasers.push_back(_laser);
	for (size_t q_i = 0; q_i < 3; ++q_i)
	{
		lasermap.at(q_i).push_back(MatrixXi::Zero(atom->get_size(), atom->get_size()));
		rabimap.push_back(MatrixXcd::Zero(atom->get_size(), atom->get_size()));
	}
}

Environment* Interaction::get_env()
{
	return env;
}

void Interaction::set_env(Environment* _env)
{
	env = _env;
	for (State* state : *atom->get_states()) state->update(env);
	atom->gen_w0();
	update();
}

double Interaction::get_delta_max()
{
	return delta_max;
}

void Interaction::set_delta_max(double _delta_max)
{
	delta_max = _delta_max;
}

bool Interaction::get_controlled()
{
	return controlled;
}

void Interaction::set_controlled(bool _controlled)
{
	controlled = _controlled;
}

double Interaction::get_dt()
{
	return dt;
}

void Interaction::set_dt(double _dt)
{
	dt = _dt;
}

bool Interaction::get_loop()
{
	return loop;
}

bool Interaction::get_time_dependent()
{
	return time_dependent;
}

void Interaction::set_time_dependent(bool _time_dependent)
{
	time_dependent = _time_dependent;
}

MatrixXi* Interaction::get_summap()
{
	return &summap;
}

std::vector<MatrixXcd>* Interaction::get_rabimap()
{
	return &rabimap;
}

MatrixXd* Interaction::get_atommap()
{
	return &atommap;
}

MatrixXd* Interaction::get_deltamap()
{
	return &deltamap;
}

void Interaction::update()
{
	gen_coordinates();
	gen_rabi();
	gen_trees();
	gen_conlist();
	gen_deltamap();
}

void Interaction::gen_coordinates()
{
	for (Laser* laser: lasers)
	{
		laser->get_polarization()->def_q_axis(*env->get_e_B(), false);
	}
}

void Interaction::gen_rabi()
{
	summap = MatrixXi::Zero(atom->get_size(), atom->get_size());
	// printf_s("%1.3f, %1.3f, %1.3f \n", (*lasers.at(0)->get_polarization()->get_q())(0).real(), (*lasers.at(0)->get_polarization()->get_q())(1).real(), (*lasers.at(0)->get_polarization()->get_q())(2).real());
	// printf_s("%1.3f, %1.3f, %1.3f \n", (*lasers.at(0)->get_polarization()->get_q())(0).imag(), (*lasers.at(0)->get_polarization()->get_q())(1).imag(), (*lasers.at(0)->get_polarization()->get_q())(2).imag());
	for (size_t q = 0; q < 3; ++q)
	{
		double q_val = static_cast<double>(q) - 1;
		for (size_t m = 0; m < lasers.size(); ++m)
		{
			// Which transitions are laser-driven?
			for (size_t col = 1; col < atom->get_size(); ++col)
			{
				for (size_t row = 0; row < col; ++row)
				{	
					State* lower = atom->get(col);
					State* upper = atom->get(row);
					if (lower->get_freq() > upper->get_freq())
					{
						lower = atom->get(row);
						upper = atom->get(col);
					}

					if (upper->get_m() - lower->get_m() != q_val) continue;  // Polarization match?
					if (abs(lasers.at(m)->get_freq() - (upper->get_freq() - lower->get_freq())) > delta_max) continue;  // In detuning range?

					std::complex<double> q_i = lasers.at(m)->get_polarization()->get_q()->array()[q];
					if (atom->get_m_dipole()->at(q)(row, col) * abs(q_i)  // q_i.real() * q_i.imag()  // (pow(q_i.real(), 2) + pow(q_i.imag(), 2))
						* lasers.at(m)->get_intensity() == 0) continue;  // Transition allowed/active?

					lasermap.at(q).at(m)(row, col) = 1;
					lasermap.at(q).at(m)(col, row) = 1;
					summap(row, col) = 1;
					summap(col, row) = 1;

					rabimap.at(m)(row, col) += 0.5 * atom->get_m_dipole()->at(q)(row, col)
					* sqrt(lasers[m]->get_intensity()) * q_i * pow(-1, q_val);  // Calc. Omega/2 for all transitions.
					rabimap.at(m)(col, row) = std::conj(rabimap.at(m)(row, col));
				}
			}
		}
	}
}

void Interaction::gen_trees()
{
	trees = std::vector<std::vector<size_t>>(0, std::vector<size_t>(0));
	size_t i = 0;
	size_t j = 0;
	std::set<size_t> visited;
	std::queue<size_t> queue;

	while (visited.size() < atom->get_size())
	{
		trees.push_back(std::vector<size_t>(1, i));
		visited.insert(i);
		for (size_t row = 0; row < atom->get_size(); ++row)
		{
			if (summap(row, i) != 0 && visited.find(row) == visited.end())
			{
				queue.push(row);
				visited.insert(row);
			}
		}
		while (!queue.empty())
		{
			j = queue.front();
			queue.pop();
			trees.back().push_back(j);
			for (size_t row = 0; row < atom->get_size(); ++row)
			{
				if (summap(row, j) != 0 && visited.find(row) == visited.end())
				{
					queue.push(row);
					visited.insert(row);
				}
			}
		}
		auto res = std::adjacent_find(visited.begin(), visited.end(), [](size_t a, size_t b) { return a + 1 != b; });
		if (res == visited.end()) i = *visited.rbegin() + 1;
		else i = *res + 1;
	}
}

void Interaction::gen_conlist()
{
	con_list = std::vector<std::vector<size_t>>(atom->get_size(), std::vector<size_t>(0));
	for (int i = 0; i < atom->get_size(); ++i)
	{
		for (int m = 0; m < lasers.size(); ++m)
		{
			for (size_t q = 0; q < 3; ++q)
			{
				if (lasermap.at(q).at(m).col(i).any())
				{
					con_list.at(i).push_back(m);
					break;
				}
			}
		}
	}
}

void Interaction::gen_deltamap()
{
	n_history = 0;
	loop = false;
	deltamap = MatrixXd::Zero(atom->get_size(), lasers.size());
	atommap = MatrixXd::Zero(atom->get_size(), atom->get_size());
	tmap.resize(0);
	for (size_t m = 0; m < lasers.size(); ++m)
	{
		tmap.push_back(MatrixXi::Zero(atom->get_size(), atom->get_size()));
	}

	std::set<size_t> visited;
	std::vector<MatrixXd> shifts(atom->get_size());
	for (size_t j = 0; j < atom->get_size(); ++j)
	{
		shifts.at(j) = MatrixXd::Zero(lasers.size(), lasers.size());
		atommap(j, j) = 1;
	}

	size_t i0 = 0;
	std::array<std::vector<size_t>, 2> path;
	for (const std::vector<size_t>& tree : trees)  // Go through all trees separately.
	{
		i0 = tree.at(0);
		propagate(i0, i0, visited, tree, path, shifts);
	}
	if (loop)
	{
		printf("\033[93mWARNING: The situations where two or more lasers form loops are "
			"not supported for time-independent coherent dynamics. Activating time dependence.\033[0m\n");
		time_dependent = true;
		/*for (size_t m = 0; m < lasers.size(); ++m)
		{
			printf("%zi:\n", m);
			for (size_t i = 0; i < atom->get_size(); ++i)
			{
				printf("[");
				for (size_t j = 0; j < atom->get_size(); ++j)
				{
					printf("%d, ", tmap.at(m)(i, j));
				}
				printf("]\n");
			}
		}*/
	};
}

void Interaction::propagate(size_t i, size_t i0, std::set<size_t>& visited, const std::vector<size_t>& tree,
	std::array<std::vector<size_t>, 2>& path, std::vector<MatrixXd>& shifts)
{
	history.push_back(i);
	n_history += 1;

	size_t j = 0;
	size_t k = 0;
	int pm = 1;
	std::queue<size_t> queue;

	visited.insert(i);
	path.at(0).push_back(i);
	atommap(i, i0) -= 1;  // Set the energy baseline to 0.
	for (size_t m = 0; m < lasers.size(); ++m)  // Iterate through all lasers m connected to i.
	{
		for (size_t _j : tree)  // Find all states which are connected to i via the laser m.
		{
			for (size_t q = 0; q < 3; ++q)
			{
				if (lasermap.at(q).at(m)(_j, i) != 0)
				{
					queue.push(_j);
					break;
				}
			}
		}
		while (!queue.empty())  // While there are states j connected to i via m, do this.
		{
			j = queue.front();
			queue.pop();
			if (atom->get(i)->get_freq() > atom->get(j)->get_freq()) pm = 1;
			else pm = -1;
			if (visited.find(j) != visited.end())  // If j already visited, do this.
			{
				//k = std::distance(path.at(0).begin(), std::find(path.at(0).begin(), path.at(0).end(), j));  // The index of j in the current path.
				//printf("%zi\n", k);
				//if (k >= path.at(1).size()) continue;  // If j was not exited via a laser in the current path, continue.
				//loop = loop || path.at(1).at(k) != m;  // If j was previously not left via m in the current path, recognize loop.
				//if (path.at(1).at(k) != m)  // If a loop is completed with the current path, ...
				//{
				//	printf("%zi, %zi, %zi\n", i, j, m);
				//	printf("shifts: %zi, %zi, %zi\n", i, j, m);
				//	tmap.at(m)(i, j) = pm;  // ... set tmap entry to true.
				//	tmap.at(m)(j, i) = -pm;
				//};
				bool _loop = check_loop(i, j, m, pm, shifts);  // Check whether a loop is not closed perfectly.
				loop = loop || _loop;
				if (_loop)  // If a loop is completed with the current path, ...
				{
					// printf("%zi, %zi\n", i, j);
					tmap.at(m)(i, j) = pm;  // ... set tmap entry to true.
					tmap.at(m)(j, i) = -pm;
				};
				continue;
			}
			deltamap.row(j) += shifts.at(i).row(m);  // Store the information for the transformation of the hamiltonian.
			deltamap(j, m) += pm;
			for (size_t _m : con_list.at(j))
			{
				shifts.at(j).row(_m) += shifts.at(i).row(m);  // All other lasers connected to the transformed state are affected as well.
				shifts.at(j)(_m, m) += pm;
			}
			path.at(1).push_back(m);
			propagate(j, i0, visited, tree, path, shifts);  // Continue from the transformed state.
			path.at(1).pop_back();
		}
	}
	path.at(0).pop_back();
}

VectorXd* Interaction::gen_w0()
{
	return atom->get_w0();
}

VectorXd* Interaction::gen_w0(Environment& env)
{
	return atom->get_w0();
}

VectorXd* Interaction::gen_w(const bool dynamics)
{
	VectorXd* w = new VectorXd(lasers.size());
	for (size_t m = 0; m < lasers.size(); ++m)
	{
		(*w)(m) = lasers.at(m)->get_freq();
		if (dynamics) (*w)(m) -= recoil((*w)(m), atom->get_mass());
		(*w)(m) *= 2 * sc::pi;
	}
	return w;
}

VectorXd* Interaction::gen_w(const VectorXd& delta, const bool dynamics)
{
	VectorXd* w = new VectorXd(lasers.size());
	for (size_t m = 0; m < lasers.size(); ++m)
	{
		(*w)(m) = (lasers.at(m)->get_freq() + delta(m));
		if (dynamics) (*w)(m) -= recoil((*w)(m), atom->get_mass());
		(*w)(m) *= 2 * sc::pi;
	}
	return w;
}

VectorXd* Interaction::gen_w(const Vector3d& v, const bool dynamics)
{
	VectorXd* w = new VectorXd(lasers.size());
	for (size_t m = 0; m < lasers.size(); ++m)
	{
		(*w)(m) = lasers.at(m)->get_detuned(v);
		if (dynamics) (*w)(m) -= recoil((*w)(m), atom->get_mass());
		(*w)(m) *= 2 * sc::pi;
	}
	return w;
}

VectorXd* Interaction::gen_w(const VectorXd& delta, const Vector3d& v, const bool dynamics)
{
	VectorXd* w = new VectorXd(lasers.size());
	for (size_t m = 0; m < lasers.size(); ++m)
	{
		(*w)(m) = lasers.at(m)->get_detuned(delta(m), v);
		if (dynamics) (*w)(m) -= recoil((*w)(m), atom->get_mass());
		(*w)(m) *= 2 * sc::pi;
	}
	return w;
}

void Interaction::update_w(VectorXd& w, const VectorXd& delta, const Vector3d& v, const bool dynamics)
{
	for (size_t m = 0; m < lasers.size(); ++m) w(m) = 2 * sc::pi * lasers.at(m)->get_detuned(delta(m), v);
}

//VectorXd Interaction::gen_delta(VectorXd& w0, VectorXd& w)
//{
//	VectorXd delta_diag(atom->get_size());
//	delta_diag = (atommap * w0) + (deltamap * w);
//	return delta_diag;
//}

VectorXd Interaction::gen_delta(const VectorXd& w0, const VectorXd& w)
{
	VectorXd delta_diag(atom->get_size());
	delta_diag = (atommap * w0) + (deltamap * w);
	return delta_diag;
}

MatrixXd* Interaction::gen_rates(VectorXd& w0, VectorXd& w)
{
	MatrixXd* R = new MatrixXd(atom->get_size(), atom->get_size());
	*R = MatrixXd::Zero(atom->get_size(), atom->get_size());
	update_rates(*R, w0, w);
	return R;
}

VectorXd* Interaction::gen_rates_sum(MatrixXd& R)
{
	VectorXd* R_sum = new VectorXd(atom->get_size());
	update_rates_sum(*R_sum, R);
	return R_sum;
}

void Interaction::update_rates(MatrixXd& R, VectorXd& w0, VectorXd& w)
{
	R.setZero();
	double _w0;
	double a;
	double r;
	for (int m = 0; m < lasers.size(); ++m)
	{
		for (int j = 1; j < atom->get_size(); ++j)
		{
			for (int i = 0; i < j; ++i)
			{
				a = atom->get_decay_map()->get_item(atom->get(i)->get_label(), atom->get(j)->get_label());  // (*atom->get_L0())(i, j) + (*atom->get_L0())(j, i);
				if (a == 0) continue;
				r = 4 * std::pow(std::abs(rabimap.at(m)(i, j)), 2);
				if (r == 0) continue;
				_w0 = abs(w0(i) - w0(j));
				R(i, j) += lorentz(w(m), _w0, a, r);
				R(j, i) = R(i, j);
			}
		}
	}
}

void Interaction::update_rates_sum(VectorXd& R_sum, MatrixXd& R)
{
	R_sum = R.colwise().sum();
}

MatrixXcd* Interaction::gen_hamiltonian(VectorXd& w0, VectorXd& w)
{
	MatrixXcd* H = new MatrixXcd(atom->get_size(), atom->get_size());
	*H = MatrixXcd::Zero(atom->get_size(), atom->get_size());
	H->diagonal() = gen_delta(w0, w);
	update_hamiltonian_off(*H);
	return H;
}

void Interaction::update_hamiltonian_off(MatrixXcd& H)
{
	for (int m = 0; m < lasers.size(); ++m)
	{
		H += rabimap.at(m);
	}
}

void Interaction::update_hamiltonian(MatrixXcd& H, VectorXd& w0, VectorXd& w, double t)
{
	H.fill(0);
	H.diagonal() = gen_delta(w0, w);
	update_hamiltonian_off(H, w, t);
}

void Interaction::update_hamiltonian_off(MatrixXcd& H, VectorXd& w, double t)
{
	VectorXd delta(atom->get_size());
	delta = deltamap * w;
	for (size_t m = 0; m < lasers.size(); ++m)
	{
		for (size_t j = 1; j < atom->get_size(); ++j)
		{
			for (size_t i = 0; i < j; ++i)
			{
				if (tmap.at(m)(i, j) == 0)
				{
					H(i, j) += rabimap.at(m)(i, j);
					H(j, i) = std::conj(H(i, j));
					continue;
				}
				// printf("(m = %zi, %zi, %zi) = %3.3f\n", m, i, j, delta(j) - tmap.at(m)(i, j) * w(m));  // if (delta(i) != 0) 
				H(i, j) += rabimap.at(m)(i, j) * std::exp(sc::i * (delta(j) - tmap.at(m)(i, j) * w(m)) * t);
				H(j, i) = std::conj(H(i, j));
			}
		}
	}
}

void Interaction::update_hamiltonian_diag(MatrixXcd& H, VectorXd& w0, VectorXd& w)
{
	H.diagonal() = gen_delta(w0, w);
}

MatrixXcd* Interaction::gen_hamiltonian_leaky(VectorXd& w0, VectorXd& w)
{
	MatrixXcd* H = new MatrixXcd(atom->get_size(), atom->get_size());
	*H = MatrixXcd::Zero(atom->get_size(), atom->get_size());
	H->diagonal() = gen_delta(w0, w) - 0.5 * sc::i * (*atom->get_Lsum());
	update_hamiltonian_off(*H);
	return H;
}

void Interaction::update_hamiltonian_leaky(MatrixXcd& H, VectorXd& w0, VectorXd& w, double t)
{
	H.fill(0);
	H.diagonal() = gen_delta(w0, w) - 0.5 * sc::i * (*atom->get_Lsum());
	update_hamiltonian_off(H, w, t);
}

void Interaction::update_hamiltonian_leaky_diag(MatrixXcd& H, VectorXd& w0, VectorXd& w)
{

	H.diagonal() = gen_delta(w0, w) - 0.5 * sc::i * (*atom->get_Lsum());
}

size_t Interaction::arange_t(double t)
{
	double n = t / dt;
	dt_var = dt;
	return static_cast<size_t>(n);
}

Result* Interaction::rate_equations(size_t n, VectorXd& x0, MatrixXd& R, VectorXd& R_sum)
{
	Result* _result = new Result();

	double t = 0;
	if (controlled)
	{
		d_dopri5_vd_type dopri5 = make_dense_output(1e-5, 1e-5, dt_var, dopri5_vd_type());
		t = integrate_n_steps(dopri5, f_rate_equations(R, R_sum, *atom->get_L0(), *atom->get_Lsum()),
			x0, 0.0, dt_var, n, push_back_state_and_time(*_result->get_y(), *_result->get_x()));
	}
	else
	{
		t = integrate_n_steps(rk4_vd_type(), f_rate_equations(R, R_sum, *atom->get_L0(), *atom->get_Lsum()),
			x0, 0.0, dt_var, n, push_back_state_and_time(*_result->get_y(), *_result->get_x()));
	}

	_result->update();
	return _result;
}

Result* Interaction::rate_equations(size_t n, VectorXd& x0, const VectorXd& delta, const Vector3d& v)
{
	VectorXd* w0 = gen_w0();
	VectorXd* w = gen_w(delta, v);
	MatrixXd* R = gen_rates(*w0, *w);
	VectorXd* R_sum = gen_rates_sum(*R);

	Result* _result = rate_equations(n, x0, *R, *R_sum);

	
	delete w;
	delete R;
	delete R_sum;
	return _result;
}

Result* Interaction::rate_equations(size_t n, VectorXd& x0, const VectorXd& delta)
{
	VectorXd* w0 = gen_w0();
	VectorXd* w = gen_w(delta);
	MatrixXd* R = gen_rates(*w0, *w);
	VectorXd* R_sum = gen_rates_sum(*R);

	Result* _result = rate_equations(n, x0, *R, *R_sum);

	
	delete w;
	delete R;
	delete R_sum;
	return _result;
}

Result* Interaction::rate_equations(size_t n, VectorXd& x0, const Vector3d& v)
{
	VectorXd* w0 = gen_w0();
	VectorXd* w = gen_w(v);
	MatrixXd* R = gen_rates(*w0, *w);
	VectorXd* R_sum = gen_rates_sum(*R);

	Result* _result = rate_equations(n, x0, *R, *R_sum);

	
	delete w;
	delete R;
	delete R_sum;
	return _result;
}

void Interaction::rate_equations(size_t n, const std::vector<VectorXd>& delta, const std::vector<Vector3d>& v, std::vector<VectorXd>& x0)
{
	VectorXd* w0 = gen_w0();

	std::vector<size_t> n_vec(x0.size());
	for (size_t i = 0; i < x0.size(); ++i) n_vec.at(i) = i;

	std::for_each(std::execution::par_unseq, n_vec.begin(), n_vec.end(),
		[this, n, &x0, &delta, &v, &w0](size_t i)
		{
			VectorXd* w = gen_w(delta.at(i), v.at(i));
			MatrixXd* R = gen_rates(*w0, *w);
			VectorXd* R_sum = gen_rates_sum(*R);
			/*update_w(*w, delta.at(i), v.at(i));
			update_rates(*R, *w0, *w);
			update_rates_sum(*R_sum, *R);*/
			double t = 0;
			if (controlled)
			{
				d_dopri5_vd_type dopri5 = make_dense_output(1e-5, 1e-5, dt_var, dopri5_vd_type());
				t = integrate_n_steps(dopri5, f_rate_equations(*R, *R_sum, *atom->get_L0(), *atom->get_Lsum()),
					x0.at(i), 0.0, dt_var, n);
			}
			else
			{
				t = integrate_n_steps(rk4_vd_type(), f_rate_equations(*R, *R_sum, *atom->get_L0(), *atom->get_Lsum()),
					x0.at(i), 0.0, dt_var, n);
			}
			delete w;
			delete R;
			delete R_sum;
			// printf("\r\033[92mProgress: %3.2f \033[0m", 100 * i / x0.size());
		}
	);
	// printf("\n");
}

Result* Interaction::rate_equations(size_t n, VectorXd& x0)
{
	VectorXd* w0 = gen_w0();
	VectorXd* w = gen_w();
	MatrixXd* R = gen_rates(*w0, *w);
	VectorXd* R_sum = gen_rates_sum(*R);

	Result* _result = rate_equations(n, x0, *R, *R_sum);

	
	delete w;
	delete R;
	delete R_sum;
	return _result;
}

Result* Interaction::rate_equations(size_t n)
{
	VectorXd x0(atom->get_size());
	x0 = VectorXd::Zero(atom->get_size());
	x0(*atom->get_gs()) = VectorXd::Ones(atom->get_gs()->size()) / atom->get_gs()->size();
	return rate_equations(n, x0);
}

Result* Interaction::rate_equations(double t)
{
	size_t n = arange_t(t);
	return rate_equations(n);
}


Result* Interaction::schroedinger(size_t n, VectorXcd& x0, VectorXd& w0, VectorXd& w, MatrixXcd& H)
{
	Result* _result = new Result();

	double t = 0;
	if (controlled)
	{
		d_dopri5_vcd_type dopri5 = make_dense_output(1e-5, 1e-5, dt_var, dopri5_vcd_type());
		t = integrate_n_steps(dopri5, f_schroedinger_t(H, w0, w, std::ref(*this)),
			x0, 0.0, dt_var, n, push_back_state_and_time(*_result->get_y(), *_result->get_x()));
	}
	else
	{
		t = integrate_n_steps(rk4_vcd_type(), f_schroedinger_t(H, w0, w, std::ref(*this)),
			x0, 0.0, dt_var, n, push_back_state_and_time(*_result->get_y(), *_result->get_x()));
	}

	_result->update();
	return _result;
}

Result* Interaction::schroedinger(size_t n, VectorXcd& x0, MatrixXcd& H)
{
	Result* _result = new Result();

	double t = 0;
	if (controlled)
	{
		d_dopri5_vcd_type dopri5 = make_dense_output(1e-5, 1e-5, dt_var, dopri5_vcd_type());
		t = integrate_n_steps(dopri5, f_schroedinger(H),
			x0, 0.0, dt_var, n, push_back_state_and_time(*_result->get_y(), *_result->get_x()));
	}
	else
	{
		t = integrate_n_steps(rk4_vcd_type(), f_schroedinger(H),
			x0, 0.0, dt_var, n, push_back_state_and_time(*_result->get_y(), *_result->get_x()));
	}

	_result->update();
	return _result;
}

Result* Interaction::schroedinger(size_t n, VectorXcd& x0, const VectorXd& delta, const Vector3d& v)
{
	VectorXd* w0 = gen_w0();
	VectorXd* w = gen_w(delta, v);
	MatrixXcd* H = gen_hamiltonian(*w0, *w);

	Result* _result = nullptr;
	if (time_dependent) _result = schroedinger(n, x0, *w0, *w, *H);
	else _result = schroedinger(n, x0, *H);

	
	delete w;
	delete H;
	return _result;
}

Result* Interaction::schroedinger(size_t n, VectorXcd& x0, const VectorXd& delta)
{
	VectorXd* w0 = gen_w0();
	VectorXd* w = gen_w(delta);
	MatrixXcd* H = gen_hamiltonian(*w0, *w);

	Result* _result = nullptr;
	if (time_dependent) _result = schroedinger(n, x0, *w0, *w, *H);
	else _result = schroedinger(n, x0, *H);

	
	delete w;
	delete H;
	return _result;
}

Result* Interaction::schroedinger(size_t n, VectorXcd& x0, const Vector3d& v)
{
	VectorXd* w0 = gen_w0();
	VectorXd* w = gen_w(v);
	MatrixXcd* H = gen_hamiltonian(*w0, *w);

	Result* _result = nullptr;
	if (time_dependent) _result = schroedinger(n, x0, *w0, *w, *H);
	else _result = schroedinger(n, x0, *H);

	
	delete w;
	delete H;
	return _result;
}

Result* Interaction::schroedinger(size_t n, VectorXcd& x0)
{
	VectorXd* w0 = gen_w0();
	VectorXd* w = gen_w();
	MatrixXcd* H = gen_hamiltonian(*w0, *w);

	Result* _result = nullptr;
	if (time_dependent) _result = schroedinger(n, x0, *w0, *w, *H);
	else _result = schroedinger(n, x0, *H);

	
	delete w;
	delete H;
	return _result;
}

Result* Interaction::schroedinger(size_t n)
{
	VectorXcd x0(atom->get_size());
	x0 = VectorXcd::Zero(atom->get_size());
	x0(0) = 1;
	return schroedinger(n, x0);
}

Result* Interaction::schroedinger(double t)
{
	size_t n = arange_t(t);
	return schroedinger(n);
}

Result* Interaction::master(size_t n, MatrixXcd& x0, VectorXd& w0, VectorXd& w, MatrixXcd& H)
{
	Result* _result = new Result();

	double t = 0;
	if (controlled)
	{
		d_dopri5_mcd_type dopri5 = make_dense_output(1e-5, 1e-5, dt_var, dopri5_mcd_type());
		t = integrate_n_steps(dopri5, f_master_t(H, *atom->get_L0(), *atom->get_L1(), w0, w, std::ref(*this)),
			x0, 0.0, dt_var, n, push_back_state_and_time(*_result->get_y(), *_result->get_x()));
	}
	else
	{
		t = integrate_n_steps(rk4_mcd_type(), f_master_t(H, *atom->get_L0(), *atom->get_L1(), w0, w, std::ref(*this)),
			x0, 0.0, dt_var, n, push_back_state_and_time(*_result->get_y(), *_result->get_x()));
	}
	
	_result->update();
	return _result;
}

Result* Interaction::master(size_t n, MatrixXcd& x0, MatrixXcd& H)
{
	Result* _result = new Result();

	double t = 0;
	if (controlled)
	{
		d_dopri5_mcd_type dopri5 = make_dense_output(1e-5, 1e-5, dt_var, dopri5_mcd_type());
		t = integrate_n_steps(dopri5, f_master(H, *atom->get_L0(), *atom->get_L1()),
			x0, 0.0, dt_var, n, push_back_state_and_time(*_result->get_y(), *_result->get_x()));
	}
	else
	{
		t = integrate_n_steps(rk4_mcd_type(), f_master(H, *atom->get_L0(), *atom->get_L1()),
			x0, 0.0, dt_var, n, push_back_state_and_time(*_result->get_y(), *_result->get_x()));
	}

	_result->update();
	return _result;
}

Result* Interaction::master(size_t n, MatrixXcd& x0, const VectorXd& delta, const Vector3d& v)
{
	VectorXd* w0 = gen_w0();
	VectorXd* w = gen_w(delta, v);
	MatrixXcd* H = gen_hamiltonian(*w0, *w);

	Result* _result = nullptr;
	if (time_dependent) _result = master(n, x0, *w0, *w, *H);
	else _result = master(n, x0, *H);

	
	delete w;
	delete H;
	return _result;
}

Result* Interaction::master(size_t n, MatrixXcd& x0, const VectorXd& delta)
{
	VectorXd* w0 = gen_w0();
	VectorXd* w = gen_w(delta);
	MatrixXcd* H = gen_hamiltonian(*w0, *w);


	Result* _result = nullptr;
	if (time_dependent) _result = master(n, x0, *w0, *w, *H);
	else _result = master(n, x0, *H);

	
	delete w;
	delete H;
	return _result;
}

Result* Interaction::master(size_t n, MatrixXcd& x0, const Vector3d& v)
{
	VectorXd* w0 = gen_w0();
	VectorXd* w = gen_w(v);
	MatrixXcd* H = gen_hamiltonian(*w0, *w);

	Result* _result = nullptr;
	if (time_dependent) _result = master(n, x0, *w0, *w, *H);
	else _result = master(n, x0, *H);

	
	delete w;
	delete H;
	return _result;
}

Result* Interaction::master(size_t n, MatrixXcd& x0)
{
	VectorXd* w0 = gen_w0();
	VectorXd* w = gen_w();
	MatrixXcd* H = gen_hamiltonian(*w0, *w);

	Result* _result = nullptr;
	if (time_dependent) _result = master(n, x0, *w0, *w, *H);
	else _result = master(n, x0, *H);

	
	delete w;
	delete H;
	return _result;
}

Result* Interaction::master(size_t n)
{
	MatrixXcd x0(atom->get_size(), atom->get_size());
	x0 = MatrixXcd::Zero(atom->get_size(), atom->get_size());
	x0.diagonal()(*atom->get_gs()) = VectorXcd::Ones(atom->get_gs()->size()) / atom->get_gs()->size();
	return master(n, x0);
}

Result* Interaction::master(double t)
{
	size_t n = arange_t(t);
	return master(n);
}

Result* Interaction::master_mc(size_t n, std::vector<VectorXcd>& x0, const VectorXd& delta, const std::vector<Vector3d>& v, const bool dynamics)
{
	std::vector<size_t> c_i;
	std::vector<size_t> c_j;
	std::vector<double> c_a;
	for (size_t j = 0; j < atom->get_size(); ++j)
	{
		for (size_t i = 0; i < atom->get_size(); ++i)
		{
			if ((*atom->get_L0())(i, j) != 0)
			{
				c_i.push_back(j);
				c_j.push_back(i);
				c_a.push_back((*atom->get_L0())(i, j));
			}
		}
	}

	std::vector<double> t = std::vector<double>(n + 1);
	std::vector<VectorXd> x = std::vector<VectorXd>(n + 1);
	std::vector<Vector3d> v_res;
	if (dynamics) v_res = std::vector<Vector3d>(v.size() * x0.size(), Vector3d::Zero());
	for (int n_t = 0; n_t < n + 1; ++n_t)
	{
		t.at(n_t) = n_t * dt_var;
		x.at(n_t) = VectorXd::Zero(atom->get_size());
	}

	std::vector<size_t> n_psi(x0.size());
	for (size_t i = 0; i < x0.size(); ++i) n_psi.at(i) = i;

	std::for_each(std::execution::par_unseq, n_psi.begin(), n_psi.end(),
		[this, n, &x0, &delta, &v, dynamics, &c_i, &c_j, &c_a, &t, &x, &v_res](size_t _n_psi)
		{
			// d_dopri5_vcd_type dopri5 = make_dense_output(1e-5, 1e-5, dt_var, dopri5_vcd_type());

			VectorXd* w0 = gen_w0();
			VectorXd* w = gen_w(delta, v.at(0), dynamics);
			MatrixXcd H(atom->get_size(), atom->get_size());
			H = MatrixXcd::Zero(atom->get_size(), atom->get_size());
			update_hamiltonian_off(H);

			thread_local std::random_device rd;
			thread_local std::mt19937 gen(rd());
			thread_local std::uniform_real_distribution<double> d(0, 1);

			double r = 0;
			double _t = 0;
			double p = 0;
			double p_n = 0;

			VectorXcd _psi0(atom->get_size());
			_psi0 = x0.at(_n_psi);

			size_t i_v = 0;
			x.at(0) += x0.at(_n_psi).cwiseAbs2() * v.size();
			for (const Vector3d& _v: v)
			{
				Vector3d v_temp = _v;
				update_w(*w, delta, v_temp, dynamics);
				if (!time_dependent) update_hamiltonian_leaky_diag(H, *w0, *w);

				size_t i = gen_index(x0.at(_n_psi).cwiseAbs2(), d, gen);
				_psi0 = x0.at(_n_psi);
				r = d(gen);
				for (size_t n_t = 0; n_t < n; ++n_t)
				{
					if (time_dependent) rk4_vcd_type().do_step(
						f_schroedinger_leaky_t(H, *w0, *w, std::ref(*this)), _psi0, t.at(n_t), dt_var);
					else rk4_vcd_type().do_step(f_schroedinger(H), _psi0, t.at(n_t), dt_var);

					x.at(n_t + 1) += _psi0.cwiseAbs2() / _psi0.cwiseAbs2().sum();
					if (_psi0.cwiseAbs2().sum() < r)
					{
						p = 0;
						p_n = 0;
						for (size_t j = 0; j < c_i.size(); ++j)
						{
							p += c_a.at(j) * std::pow(std::abs(_psi0(c_i.at(j))), 2);
						}
						r = d(gen);
						for (size_t j = 0; j < c_i.size(); ++j)
						{
							p_n += c_a.at(j) * std::pow(std::abs(_psi0(c_i.at(j))), 2) / p;
							if (p_n >= r)
							{
								if (dynamics)
								{
									v_temp += atom->gen_velocity_change(i, c_i.at(j), c_j.at(j), *lasers.at(0)->get_k(), gen);
									update_w(*w, delta, v_temp, dynamics);
									if (!time_dependent) update_hamiltonian_leaky_diag(H, *w0, *w);
								}

								i = c_j.at(j);
								_psi0.fill(0);
								_psi0(c_j.at(j)) = 1.;
								break;
							}
						}
						r = d(gen);
					}
				}
				if (dynamics) v_res.at(_n_psi * v.size() + i_v++) = v_temp;
			}
			
			delete w;
		});

	Result* _result = new Result();
	if (dynamics) _result->set_v(v_res);
	for (size_t n_t = 0; n_t < n + 1; ++n_t)
	{
		_result->add(t.at(n_t), x.at(n_t) / v.size() / atom->get_gs()->size());
	}
	_result->update();
	return _result;
}

Result* Interaction::master_mc(size_t n, std::vector<VectorXcd>& x0, const VectorXd& delta, const Vector3d& v, size_t num)
{
	std::vector<size_t> c_i;
	std::vector<size_t> c_j;
	std::vector<double> c_a;
	for (size_t j = 0; j < atom->get_size(); ++j)
	{
		for (size_t i = 0; i < atom->get_size(); ++i)
		{
			if ((*atom->get_L0())(i, j) != 0)
			{
				c_i.push_back(j);
				c_j.push_back(i);
				c_a.push_back((*atom->get_L0())(i, j));
			}
		}
	}
	
	std::vector<double> t = std::vector<double>(n + 1);
	std::vector<VectorXd> x = std::vector<VectorXd>(n + 1);
	for (int n_t = 0; n_t < n + 1; ++n_t)
	{
		t.at(n_t) = n_t * dt_var;
		x.at(n_t) = VectorXd::Zero(atom->get_size());
	}

	std::for_each(std::execution::par_unseq, x0.begin(), x0.end(), [this, n, num, &delta, &v, &c_i, &c_j, &c_a, &t, &x](auto&& psi0)
		{
			// d_dopri5_vcd_type dopri5 = make_dense_output(1e-5, 1e-5, dt_var, dopri5_vcd_type());

			VectorXd* w0 = gen_w0();
			VectorXd* w = gen_w(delta, v);
			MatrixXcd* H = gen_hamiltonian_leaky(*w0, *w);

			thread_local std::random_device rd;
			thread_local std::mt19937 gen(rd());
			thread_local std::uniform_real_distribution<double> d(0, 1);

			double r = 0;
			double _t = 0;
			double p = 0;
			double p_n = 0;

			VectorXcd _psi0(atom->get_size());
			_psi0 = psi0;

			x.at(0) += psi0.cwiseAbs2() * num;
			for (size_t i = 0; i < num; ++i)
			{
				_psi0 = psi0;
				r = d(gen);
				for (size_t n_t = 0; n_t < n; ++n_t)
				{
					if (time_dependent) rk4_vcd_type().do_step(
						f_schroedinger_leaky_t(*H, *w0, *w, std::ref(*this)), _psi0, t.at(n_t), dt_var);
					else rk4_vcd_type().do_step(f_schroedinger(*H), _psi0, t.at(n_t), dt_var);

					x.at(n_t + 1) += _psi0.cwiseAbs2() / _psi0.cwiseAbs2().sum();
					if (_psi0.cwiseAbs2().sum() < r)
					{
						p = 0;
						p_n = 0;
						for (size_t j = 0; j < c_i.size(); ++j)
						{
							p += c_a.at(j) * std::pow(std::abs(_psi0(c_i.at(j))), 2);
						}
						r = d(gen);
						for (size_t j = 0; j < c_i.size(); ++j)
						{
							p_n += c_a.at(j) * std::pow(std::abs(_psi0(c_i.at(j))), 2) / p;
							if (p_n >= r)
							{
								_psi0.fill(0);
								_psi0(c_j.at(j)) = 1.;
								break;
							}
						}
						r = d(gen);
					}
				}
			}
			
			delete w;
			delete H;
		});

	Result* _result = new Result();
	for (size_t n_t = 0; n_t < n + 1; ++n_t)
	{
		_result->add(t.at(n_t), x.at(n_t) / num / atom->get_gs()->size());
	}
	_result->update();
	return _result;
}

Result* Interaction::master_mc(size_t n, const VectorXd& delta, std::vector<Vector3d>& v, bool dynamics)
{
	std::vector<VectorXcd> x0(atom->get_gs()->size(), VectorXcd::Zero(atom->get_size()));
	size_t i = 0;
	for (size_t gs : *atom->get_gs())
	{
		x0.at(i)(gs) = 1.;
		++i;
	}
	return master_mc(n, x0, delta, v, dynamics);
}

Result* Interaction::master_mc(size_t n, std::vector<VectorXcd>& x0, size_t num, bool dynamics)
{
	VectorXd delta(lasers.size());
	delta = VectorXd::Zero(lasers.size());
	if (dynamics)
	{
		std::vector<Vector3d> v;
		v = std::vector<Vector3d>(num, Vector3d::Zero());
		return master_mc(n, x0, delta, v, true);
	}
	else
	{
		Vector3d v;
		v = Vector3d::Zero();
		return master_mc(n, x0, delta, v, num);
	}
}

Result* Interaction::master_mc(size_t n, size_t num, bool dynamics)
{
	std::vector<VectorXcd> x0(atom->get_gs()->size(), VectorXcd::Zero(atom->get_size()));
	size_t i = 0;
	for (size_t gs: *atom->get_gs())
	{
		x0.at(i)(gs) = 1.;
		++i;
	}
	return master_mc(n, x0, num, dynamics);
}

Result* Interaction::master_mc(double t, size_t num, bool dynamics)
{
	size_t n = arange_t(t);
	return master_mc(n, num, dynamics);
}

Result* Interaction::call_solver_v(size_t n, VectorXd& x0, const VectorXd& delta, const Vector3d& v, int solver)
{
	return rate_equations(n, x0, delta, v);
}

Result* Interaction::call_solver_v(size_t n, VectorXcd& x0, const VectorXd& delta, const Vector3d& v, int solver)
{
	return schroedinger(n, x0, delta, v);
}

Result* Interaction::call_solver_v(size_t n, MatrixXcd& x0, const VectorXd& delta, const Vector3d& v, int solver)
{
	return master(n, x0, delta, v);
}

template<typename T>
Result* Interaction::mean_v(const VectorXd& delta, const std::vector<Vector3d>& v, size_t n, T& y0, int solver)
{
	std::vector<double> t = std::vector<double>(n + 1);
	std::vector<VectorXd> x = std::vector<VectorXd>(n + 1);
	for (size_t n_t = 0; n_t < n + 1; ++n_t)
	{
		t.at(n_t) = n_t * dt_var;
		x.at(n_t) = VectorXd::Zero(atom->get_size());
	}
	std::vector<size_t> n_v_vector(v.size());
	for (size_t n_v = 0; n_v < v.size(); ++n_v)
	{
		n_v_vector.at(n_v) = n_v;
	}

	// const auto it = v.colwise();
	std::for_each(n_v_vector.begin(), n_v_vector.end(), [this, n, solver, &y0, &delta, &v, &t, &x](size_t n_v)
		{
			T _y0 = y0;
			Result* _result_temp = call_solver_v(n, _y0, delta, v.at(n_v), solver);

			for (size_t n_t = 0; n_t < n + 1; ++n_t) x.at(n_t) += _result_temp->get_y()->at(n_t);
			delete _result_temp;
		}
	);

	Result* _result = new Result();

	for (size_t n_t = 0; n_t < n + 1; ++n_t)
	{
		_result->add(t.at(n_t), x.at(n_t) / v.size());
	}
	_result->update();
	return _result;
}

template<typename T>
Result* Interaction::mean_v(const std::vector<Vector3d>& v, size_t n, T& y0, int solver)
{
	std::vector<double> t = std::vector<double>(n + 1);
	std::vector<VectorXd> x = std::vector<VectorXd>(n + 1);
	for (size_t n_t = 0; n_t < n + 1; ++n_t)
	{
		t.at(n_t) = n_t * dt_var;
		x.at(n_t) = VectorXd::Zero(atom->get_size());
	}
	std::vector<size_t> n_v_vector(v.size());
	for (size_t n_v = 0; n_v < v.size(); ++n_v)
	{
		n_v_vector.at(n_v) = n_v;
	}
	VectorXd delta(lasers.size());
	delta = VectorXd::Zero(lasers.size());

	// const auto it = v.colwise();
	std::for_each(std::execution::par_unseq, n_v_vector.begin(), n_v_vector.end(), [this, n, solver, &y0, &delta, &v, &t, &x](size_t n_v)
		{
			T _y0 = y0;
			Result* _result_temp = call_solver_v(n, _y0, delta, v.at(n_v), solver);

			for (size_t n_t = 0; n_t < n + 1; ++n_t) x.at(n_t) += _result_temp->get_y()->at(n_t);
			delete _result_temp;
			printf("\r\033[92mProgress: %zi / %zi. \033[0m", n_v + 1, v.size());
		}
	);
	printf("\r\033[92mProgress: %zi / %zi. \033[0m", v.size(), v.size());
	printf("\n");

	Result* _result = new Result();

	for (size_t n_t = 0; n_t < n + 1; ++n_t)
	{
		_result->add(t.at(n_t), x.at(n_t) / v.size());
	}
	_result->update();
	return _result;
}

Result* Interaction::mean_v(const std::vector<Vector3d>& v, size_t n, int solver)
{
	if (solver == 0)
	{
		VectorXd y0(atom->get_size());
		y0 = VectorXd::Zero(atom->get_size());
		y0(*atom->get_gs()) = VectorXd::Ones(atom->get_gs()->size()) / atom->get_gs()->size();
		return mean_v(v, n, y0, solver);
	}
	else if (solver == 1)
	{
		VectorXcd y0(atom->get_size());
		y0 = VectorXcd::Zero(atom->get_size());
		y0(0) = 1;
		return mean_v(v, n, y0, solver);
	}
	else
	{
		MatrixXcd y0(atom->get_size(), atom->get_size());
		y0 = MatrixXcd::Zero(atom->get_size(), atom->get_size());
		y0.diagonal()(*atom->get_gs()) = VectorXcd::Ones(atom->get_gs()->size()) / atom->get_gs()->size();
		return mean_v(v, n, y0, solver);
	}
}

Result* Interaction::mean_v(const std::vector<Vector3d>& v, double t, int solver)
{
	size_t n = arange_t(t);
	return mean_v(v, n, solver);
}

Result* Interaction::call_solver_d(size_t n, VectorXd& x0, const VectorXd& delta, int solver)
{
	return rate_equations(n, x0, delta);
}

Result* Interaction::call_solver_d(size_t n, VectorXcd& x0, const VectorXd& delta, int solver)
{
	return schroedinger(n, x0, delta);
}

Result* Interaction::call_solver_d(size_t n, MatrixXcd& x0, const VectorXd& delta, int solver)
{
	return master(n, x0, delta);
}

template<typename T>
Spectrum* Interaction::spectrum(const std::vector<VectorXd>& delta, size_t n, T& y0, int solver)
{
	std::vector<Result*> results(delta.size());
	std::vector<size_t> n_d_vector(delta.size());
	std::vector<VectorXd> x = std::vector<VectorXd>(delta.size());
	for (size_t n_d = 0; n_d < delta.size(); ++n_d)
	{
		n_d_vector.at(n_d) = n_d;
		x.at(n_d) = VectorXd::Zero(atom->get_size());
	}

	std::for_each(std::execution::par_unseq, n_d_vector.begin(), n_d_vector.end(), [this, n, solver, &y0, &delta, &x, &results](size_t n_d)
		{
			T _y0 = y0;
			Result* _result_temp = call_solver_d(n, _y0, delta.at(n_d), solver);
			results.at(n_d) = _result_temp;
			printf("\r\033[92mProgress: %zi / %zi. \033[0m", n_d + 1, delta.size());
		}
	);
	printf("\r\033[92mProgress: %zi / %zi. \033[0m", delta.size(), delta.size());
	printf("\n");

	Spectrum* _spectrum = new Spectrum(delta, results);
	for (auto p : results)
	{
		delete p;
	}
	results.clear();
	return _spectrum;
}

Spectrum* Interaction::spectrum(const std::vector<VectorXd>& delta, size_t n, int solver)
{
	if (solver == 0)
	{
		VectorXd y0(atom->get_size());
		y0 = VectorXd::Zero(atom->get_size());
		y0(*atom->get_gs()) = VectorXd::Ones(atom->get_gs()->size()) / atom->get_gs()->size();
		return spectrum(delta, n, y0, solver);
	}
	else if (solver == 1)
	{
		VectorXcd y0(atom->get_size());
		y0 = VectorXcd::Zero(atom->get_size());
		y0(0) = 1;
		return spectrum(delta, n, y0, solver);
	}
	else
	{
		MatrixXcd y0(atom->get_size(), atom->get_size());
		y0 = MatrixXcd::Zero(atom->get_size(), atom->get_size());
		y0.diagonal()(*atom->get_gs()) = VectorXcd::Ones(atom->get_gs()->size()) / atom->get_gs()->size();
		return spectrum(delta, n, y0, solver);
	}
}

Spectrum* Interaction::spectrum(const std::vector<VectorXd>& delta, double t, int solver)
{
	size_t n = arange_t(t);
	return spectrum(delta, n, solver);
}

template<typename T>
Spectrum* Interaction::spectrum(const std::vector<VectorXd>& delta, const std::vector<Vector3d>& v, size_t n, T& y0, int solver)
{
	std::vector<Result*> results(delta.size());
	std::vector<size_t> n_d_vector(delta.size());
	std::vector<VectorXd> x = std::vector<VectorXd>(delta.size());
	for (size_t n_d = 0; n_d < delta.size(); ++n_d)
	{
		n_d_vector.at(n_d) = n_d;
		x.at(n_d) = VectorXd::Zero(atom->get_size());
	}

	std::for_each(std::execution::par_unseq, n_d_vector.begin(), n_d_vector.end(), [this, n, solver, &y0, &delta, &v, &x, &results](size_t n_d)
		{
			T _y0 = y0;
			Result* _result_temp = mean_v(delta.at(n_d), v, n, y0, solver);
			results.at(n_d) = _result_temp;
			printf("\r\033[92mProgress: %zi / %zi. \033[0m", n_d + 1, delta.size());
		}
	);
	printf("\r\033[92mProgress: %zi / %zi. \033[0m", delta.size(), delta.size());
	printf("\n");

	Spectrum* _spectrum = new Spectrum(delta, results);
	for (auto p : results)
	{
		delete p;
	}
	results.clear();
	return _spectrum;
}

Spectrum* Interaction::spectrum(const std::vector<VectorXd>& delta, const std::vector<Vector3d>& v, size_t n, int solver)
{
	if (solver == 0)
	{
		VectorXd y0(atom->get_size());
		y0 = VectorXd::Zero(atom->get_size());
		y0(*atom->get_gs()) = VectorXd::Ones(atom->get_gs()->size()) / atom->get_gs()->size();
		return spectrum(delta, v, n, y0, solver);
	}
	else if (solver == 1)
	{
		VectorXcd y0(atom->get_size());
		y0 = VectorXcd::Zero(atom->get_size());
		y0(0) = 1;
		return spectrum(delta, v, n, y0, solver);
	}
	else
	{
		MatrixXcd y0(atom->get_size(), atom->get_size());
		y0 = MatrixXcd::Zero(atom->get_size(), atom->get_size());
		y0.diagonal()(*atom->get_gs()) = VectorXcd::Ones(atom->get_gs()->size()) / atom->get_gs()->size();
		return spectrum(delta, v, n, y0, solver);
	}
}

Spectrum* Interaction::spectrum(const std::vector<VectorXd>& delta, const std::vector<Vector3d>& v, double t, int solver)
{
	size_t n = arange_t(t);
	return spectrum(delta, v, n, solver);
}

Spectrum* Interaction::spectrum_mc(const std::vector<VectorXd>& delta, const std::vector<Vector3d>& v, size_t n, std::vector<VectorXcd>& y0, int solver, bool dynamics)
{
	std::vector<Result*> results(delta.size());
	std::vector<size_t> n_d_vector(delta.size());
	std::vector<VectorXd> x = std::vector<VectorXd>(delta.size());
	for (size_t n_d = 0; n_d < delta.size(); ++n_d)
	{
		n_d_vector.at(n_d) = n_d;
		x.at(n_d) = VectorXd::Zero(atom->get_size());
	}

	std::for_each(std::execution::par_unseq, n_d_vector.begin(), n_d_vector.end(), [this, n, solver, &y0, &delta, &v, dynamics, &x, &results](size_t n_d)
		{
			Result* _result_temp = master_mc(n, y0, delta.at(n_d), v, dynamics);
			results.at(n_d) = _result_temp;
			printf("\r\033[92mProgress: %zi / %zi. \033[0m", n_d + 1, delta.size());
		}
	);
	printf("\r\033[92mProgress: %zi / %zi. \033[0m", delta.size(), delta.size());
	printf("\n");

	Spectrum* _spectrum = new Spectrum(delta, results);
	for (auto p : results)
	{
		delete p;
	}
	results.clear();
	return _spectrum;
}
