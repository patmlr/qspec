
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
		// m_states.push_back(x);
		m_times.push_back(t);
	}

	void operator()(const MatrixXcd& x, double t)
	{
		m_states.push_back(x.diagonal().real());
		m_times.push_back(t);
	}
};

struct push_back_VectorXd
{
	std::vector< VectorXd >& m_states;

	push_back_VectorXd(std::vector< VectorXd >& states)
		: m_states(states) { }

	void operator()(const VectorXd& x, double t)
	{
		m_states.push_back(x);
	}
};

struct push_back_VectorXcd
{
	std::vector< VectorXcd >& m_states;

	push_back_VectorXcd(std::vector< VectorXcd >& states)
		: m_states(states) { }

	void operator()(const VectorXcd& x, double t)
	{
		m_states.push_back(x);
	}
};

struct push_back_MatrixXcd
{
	std::vector< MatrixXcd >& m_states;

	push_back_MatrixXcd(std::vector< MatrixXcd >& states)
		: m_states(states) { }

	void operator()(const MatrixXcd& x, double t)
	{
		m_states.push_back(x);
	}
};

struct push_back_VectorXcd_as_MatrixXcd
{
	std::vector< MatrixXcd >& m_states;
	size_t rows, cols;

	push_back_VectorXcd_as_MatrixXcd(std::vector< MatrixXcd >& states, size_t r, size_t c)
		: m_states(states), rows(r), cols(c) { }

	void operator()(const VectorXcd& x, double t)
	{
		Map<const MatrixXcd> X(x.data(), rows, cols);
		m_states.push_back(X);
	}
};

/* struct f_rate_equations_full
{
	MatrixXd& R;
	VectorXd& R_sum;
	MatrixXd& L0;
	VectorXd& L_sum;

	f_rate_equations_full(MatrixXd& _R, VectorXd& _R_sum, MatrixXd& _L0, VectorXd& _L_sum)
		: R(_R), R_sum(_R_sum), L0(_L0), L_sum(_L_sum) { }

	void operator()(const VectorXd& x, VectorXd& dxdt, double t)
	{
		dxdt = (R + L0) * x - (R_sum + L_sum).cwiseProduct(x);
	}
};

struct f_schroedinger_full
{
	MatrixXcd& H;

	f_schroedinger_full(MatrixXcd& _H) : H(_H) { }

	void operator()(const VectorXcd& x, VectorXcd& dxdt, double t)
	{
		dxdt = -sc::i * (H * x);
	}
};

struct f_schroedinger_t_full
{
	MatrixXcd& H;
	VectorXd& w0;
	VectorXd& w;
	Interaction& interaction;

	f_schroedinger_t_full(MatrixXcd& _H, VectorXd& _w0, VectorXd& _w, Interaction& _interaction)
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


struct f_master_full
{
	MatrixXcd& H;
	MatrixXd& L0;
	MatrixXd& L1;
	// size_t size;
	// std::complex<double> sum;

	f_master_full(MatrixXcd& _H, MatrixXd& _L0, MatrixXd& _L1) : H(_H), L0(_L0), L1(_L1) { }

	void operator()(const VectorXcd& x, VectorXcd& dxdt, double t)
	{
		Map<const MatrixXcd> X(x.data(), H.rows(), H.cols());
    	Map<MatrixXcd> dXdt(dxdt.data(), H.rows(), H.cols());

		dXdt = -sc::i * (H * X - X * H) + L1.cwiseProduct(X);
		dXdt.diagonal() += L0 * X.diagonal();
	}
};

struct f_master_t_full
{
	MatrixXcd& H;
	MatrixXd& L0;
	MatrixXd& L1;
	VectorXd& w0;
	VectorXd& w;
	Interaction& interaction;

	f_master_t_full(MatrixXcd& _H, MatrixXd& _L0, MatrixXd& _L1, VectorXd& _w0, VectorXd& _w, Interaction& _interaction) 
		: H(_H), L0(_L0), L1(_L1), w0(_w0), w(_w), interaction(_interaction) { }

	void operator()(const VectorXcd& x, VectorXcd& dxdt, double t)
	{
		Map<const MatrixXcd> X(x.data(), H.rows(), H.cols());
    	Map<MatrixXcd> dXdt(dxdt.data(), H.rows(), H.cols());

		interaction.update_hamiltonian(H, w0, w, t);
		dXdt = -sc::i * (H * X - X * H) + L1.cwiseProduct(X);
		dXdt.diagonal() += L0 * X.diagonal();
	}
}; */


VectorXd rate_exponential(double t, VectorXd x0, MatrixXd R)
{
	return (R * t).exp() * x0;
}

struct RateExponential
{
    Eigen::RealSchur<MatrixXd> schur;
    VectorXd tmp;

    RateExponential(const MatrixXd& R_exp) : schur(R_exp), tmp(R_exp.rows()) {}

    VectorXd operator()(double t, const VectorXd& x0)
    {
        tmp.noalias() = schur.matrixU().transpose() * x0;
        tmp = (schur.matrixT() * t).exp() * tmp;

        return schur.matrixU() * tmp;
    }
};

struct f_rate_equations
{
    const MatrixXd& R;
    const MatrixXd& L0;
    const VectorXd& R_sum;
    const VectorXd& L_sum;

    MatrixXd A;
    VectorXd D;
    VectorXd Ax;

    f_rate_equations(const MatrixXd& _R, const VectorXd& _R_sum, const MatrixXd& _L0, const VectorXd& _L_sum)
        : R(_R), L0(_L0), R_sum(_R_sum), L_sum(_L_sum), A(_R.rows(), _R.cols()), D(_R_sum.size()), Ax(_R_sum.size())
    {
        A.noalias() = R + L0;
        D.noalias() = R_sum + L_sum;
    }

    void operator()(const VectorXd& x, VectorXd& dxdt, double t)
    {
        Ax.noalias() = A * x;

        dxdt.noalias() = Ax;
        dxdt.array() -= D.array() * x.array();
    }
};

struct f_schroedinger
{
    const MatrixXcd& H;

    VectorXcd Hx;

    f_schroedinger(const MatrixXcd& _H) 
		: H(_H), Hx(_H.rows()) {}

    void operator()(const VectorXcd& x, VectorXcd& dxdt, double t)
    {
        Hx.noalias() = H.selfadjointView<Lower>() * x;
        dxdt.noalias() = -sc::i * Hx;
    }
};

struct f_schroedinger_t
{
	MatrixXcd& H;
	const VectorXd& w0;
	const VectorXd& w;
	Interaction& interaction;

    VectorXcd Hx;

	f_schroedinger_t(MatrixXcd& _H, const VectorXd& _w0, const VectorXd& _w, Interaction& _interaction)
		: H(_H), w0(_w0), w(_w), interaction(_interaction), Hx(_H.rows()) { }

	void operator()(const VectorXcd& x, VectorXcd& dxdt, double t)
	{
		interaction.update_hamiltonian(H, w0, w, t);

        Hx.noalias() = H.selfadjointView<Lower>() * x;
        dxdt.noalias() = -sc::i * Hx;
	}
};

struct f_schroedinger_leaky
{
    MatrixXcd& H;

    f_schroedinger_leaky(MatrixXcd& _H)
		: H(_H) {}

    void operator()(const VectorXcd& x, VectorXcd& dxdt, double t)
    {
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
    const MatrixXcd& H;
    const MatrixXd&  L0;
    const MatrixXd&  L1;

    MatrixXcd HX;
    MatrixXcd XH;

    f_master(const MatrixXcd& _H, const MatrixXd&  _L0, const MatrixXd&  _L1) 
		: H(_H), L0(_L0), L1(_L1), HX(_H.rows(), _H.cols()), XH(_H.rows(), _H.cols()) {}

    void operator()(const VectorXcd& x, VectorXcd& dxdt, double t)
    {
        const size_t n = H.rows();

        Map<const MatrixXcd> X(x.data(), n, n);
        Map<MatrixXcd> dXdt(dxdt.data(), n, n);

        HX.noalias() = H.selfadjointView<Lower>() * X;
        XH.noalias() = X * H.selfadjointView<Lower>();

        dXdt.noalias() = -sc::i * (HX - XH);

        dXdt.array() += L1.array() * X.array();

        dXdt.diagonal().noalias() += L0 * X.diagonal();
    }
};

struct f_master_t
{
    MatrixXcd& H;
    const MatrixXd& L0;
    const MatrixXd& L1;
    const VectorXd& w0;
    const VectorXd& w;
    Interaction& interaction;

    MatrixXcd HX;
    MatrixXcd XH;

    f_master_t(
		MatrixXcd& _H, const MatrixXd& _L0,const MatrixXd& _L1, const
		VectorXd& _w0, const VectorXd& _w, Interaction& _interaction
	) 
	: H(_H), L0(_L0), L1(_L1), w0(_w0), w(_w), interaction(_interaction),
	  HX(_H.rows(), _H.cols()), XH(_H.rows(), _H.cols()) {}

    void operator()(const VectorXcd& x, VectorXcd& dxdt, double t)
    {
        const int n = H.rows();

        Map<const MatrixXcd> X(x.data(), n, n);
        Map<MatrixXcd> dXdt(dxdt.data(), n, n);

        interaction.update_hamiltonian(H, w0, w, t);

        HX.noalias() = H.selfadjointView<Lower>() * X;
        XH.noalias() = X * H.selfadjointView<Lower>();

        dXdt.noalias() = -sc::i * (HX - XH);

        dXdt.array() += L1.array() * X.array();

        dXdt.diagonal().noalias() += L0 * X.diagonal();
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
	summap = MatrixXi::Zero(atom->get_size(), atom->get_size());
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
	std::vector<MatrixXi> q_map;
	for (size_t q = 0; q < 2 * atom->get_decay_map()->get_k_em_max() + 1; ++q)
	{
		q_map.push_back(MatrixXi::Zero(atom->get_size(), atom->get_size()));
	}
	lasermap.push_back(q_map);
	rabimap.push_back(MatrixXcd::Zero(atom->get_size(), atom->get_size()));
}

Environment* Interaction::get_env()
{
	return env;
}

void Interaction::set_env(Environment* _env)
{
	env = _env;
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

bool Interaction::get_dense()
{
	return dense;
}

void Interaction::set_dense(bool _dense)
{
	dense = _dense;
}

double Interaction::get_dt()
{
	return dt;
}

void Interaction::set_dt(double _dt)
{
	dt = _dt;
}

double Interaction::get_dt_max()
{
	return dt_max;
}

void Interaction::set_dt_max(double _dt_max)
{
	dt_max = _dt_max;
}

double Interaction::get_atol()
{
	return atol;
}

void Interaction::set_atol(double _atol)
{
	atol = _atol;
}

double Interaction::get_rtol()
{
	return rtol;
}

void Interaction::set_rtol(double _rtol)
{
	rtol = _rtol;
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

MatrixXcd Interaction::get_hamiltonian(const double t, const VectorXd& delta, const Vector3d& v)
{
	VectorXd w0 = *atom->get_w0();
	VectorXd w = gen_w(delta, v);
	MatrixXcd H = gen_hamiltonian(w0, w);
	if (time_dependent) update_hamiltonian(H, w0, w, t);
	return H;
}

void Interaction::resonance_info()
{
	printf("\033[95mResonance info:\033[0m\n");

	for (size_t k = 0; k < lasers.size(); ++k)
	{
		size_t n = 0;
		Laser* laser = lasers.at(k);
		printf("\033[94mLaser %zi @ %.3f MHz:\033[0m\n", k, laser->get_freq());

		for (size_t i = 1; i < atom->get_size(); ++i)
		{
			for (size_t j = 0; j < i; ++j)
			{
				if (abs(rabimap.at(k)(i, j)) == 0.) continue;

				State* lower = atom->get(i);
				State* upper = atom->get(j);
				if (lower->get_freq() > upper->get_freq())
				{
					lower = atom->get(j);
					upper = atom->get(i);
				}
				// printf("%s\n", fmt::format("{} -> {}: {:.3e} MHz", lower->repr(), upper->repr(), upper->get_freq() - lower->get_freq() - laser->get_freq()).c_str());
				printf("%s\n", fmt::format("{} -> {}: {:.10f} MHz", lower->repr(), upper->repr(), upper->get_freq() - lower->get_freq() - laser->get_freq()).c_str());
				n += 1;

			}
		}
		if (n == 0) printf("No resonances!");
		printf("\n");
	}
}

int Interaction::update()
{
	try {
		gen_coordinates();
		atom->set_env(env);
		atom->update();
		gen_rabi();
		gen_trees();
		gen_conlist();
		gen_deltamap();
	}
	catch (const polarization_error& e)
	{
		return -1;
	}
	return 0;
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
	summap.setZero();
	for (size_t m = 0; m < lasers.size(); ++m)
	{
		rabimap.at(m).setZero();
		for (size_t q = 0; q < 2 * atom->get_decay_map()->get_k_em_max() + 1; ++q)
		{
			lasermap.at(m).at(q).setZero();
		}
		
		std::array<std::vector<VectorXcd>, 2> kpol_list;  // List of polarization-k-vector tensors in helicity basis for electric and magnetic k-multipole orders.
		for (size_t _k = 1; _k <= atom->get_decay_map()->get_k_em_max(); ++_k)
		{
			kpol_list.at(0).push_back(lasers.at(m)->get_kpol(true, _k, *env->get_e_B()));
			kpol_list.at(1).push_back(lasers.at(m)->get_kpol(false, _k, *env->get_e_B()));
			
			
			/*printf("%zi(%zi): ", m, _k);
			for (size_t i = 0; i < kpol_list.at(0).back().size(); ++i)
			{
				printf("%.3f + %.3fj, ", kpol_list.at(0).back()(i).real(), kpol_list.at(0).back()(i).imag());
			}
			printf("\n");*/
		}


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

				for (size_t _k = 1; _k <= atom->get_decay_map()->get_k_em_max(); ++_k)
				{
					size_t k = atom->get_emk(_k, row, col);
					size_t ke = atom->get_ek(_k, row, col);
					size_t km = atom->get_mk(_k, row, col);
					bool electric = ke != 0;
					size_t i_em = (electric) ? 0 : 1;

					// printf("\nk(%zi, %zi): e%zi, m%zi", row, col, ke, km);
					if (k == 0) continue;
					if (abs(lasers.at(m)->get_freq() - (upper->get_freq() - lower->get_freq())) > delta_max) continue;  // In detuning range?

					double q_val = upper->get_m() - lower->get_m();
					if (abs(q_val) > k) continue;
					size_t q = k + q_val;
					// printf("\n(k, q): %zi, %3.3f", k, q_val);

					std::complex<double> q_i = kpol_list.at(i_em).at(k - 1)(q);
					//printf("\nqi: %3.3f + %3.3fj", q_i.real(), q_i.imag());

					if ((atom->get_d_em(k, row, col)) * abs(q_i)  // q_i.real() * q_i.imag()  // (pow(q_i.real(), 2) + pow(q_i.imag(), 2))
						* lasers.at(m)->get_intensity() == 0) continue;  // Transition allowed/active?

					lasermap.at(m).at(q)(row, col) = 1;
					lasermap.at(m).at(q)(col, row) = 1;
					summap(row, col) = 1;
					summap(col, row) = 1;

					rabimap.at(m)(row, col) += -0.5 * (atom->get_d_em(k, row, col))
					* sqrt(lasers[m]->get_intensity()) * q_i * pow(-1, q_val) * pow(sc::i, k - 1);  // Calc. Omega/2 for all transitions.
					rabimap.at(m)(col, row) = std::conj(rabimap.at(m)(row, col));

					break;  // Use only leading order for now.
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
			for (size_t q = 0; q < 2 * atom->get_decay_map()->get_k_em_max() + 1; ++q)
			{
				if (lasermap.at(m).at(q).col(i).any())
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
			for (size_t q = 0; q < 2 * atom->get_decay_map()->get_k_em_max() + 1; ++q)
			{
				if (lasermap.at(m).at(q)(_j, i) != 0)
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

VectorXd Interaction::gen_w(const bool dynamics)
{
	VectorXd w = VectorXd(lasers.size());
	for (size_t m = 0; m < lasers.size(); ++m)
	{
		w(m) = lasers.at(m)->get_freq();
		if (dynamics) w(m) -= recoil(w(m), atom->get_mass());
		w(m) *= 2 * sc::pi;
	}
	return w;
}

VectorXd Interaction::gen_w(const VectorXd& delta, const bool dynamics)
{
	VectorXd w = VectorXd(lasers.size());
	for (size_t m = 0; m < lasers.size(); ++m)
	{
		w(m) = (lasers.at(m)->get_freq() + delta(m));
		if (dynamics) w(m) -= recoil(w(m), atom->get_mass());
		w(m) *= 2 * sc::pi;
	}
	return w;
}

VectorXd Interaction::gen_w(const Vector3d& v, const bool dynamics)
{
	VectorXd w = VectorXd(lasers.size());
	for (size_t m = 0; m < lasers.size(); ++m)
	{
		w(m) = lasers.at(m)->get_detuned(v);
		if (dynamics) w(m) -= recoil(w(m), atom->get_mass());
		w(m) *= 2 * sc::pi;
	}
	return w;
}

VectorXd Interaction::gen_w(const VectorXd& delta, const Vector3d& v, const bool dynamics)
{
	VectorXd w = VectorXd(lasers.size());
	for (size_t m = 0; m < lasers.size(); ++m)
	{
		w(m) = lasers.at(m)->get_detuned(delta(m), v);
		if (dynamics) w(m) -= recoil(w(m), atom->get_mass());
		w(m) *= 2 * sc::pi;
	}
	return w;
}

void Interaction::update_w(VectorXd& w, const VectorXd& delta, const Vector3d& v, const bool dynamics)
{
	for (size_t m = 0; m < lasers.size(); ++m)
	{
		w(m) = lasers.at(m)->get_detuned(delta(m), v);
		if (dynamics) w(m) -= recoil(w(m), atom->get_mass());
		w(m) *= 2 * sc::pi;
	}
}

VectorXd Interaction::get_delta(const VectorXd& w0, const VectorXd& w)
{
	return (atommap * w0) + (deltamap * w);
}

void Interaction::update_delta(VectorXd& delta_diag, const VectorXd& w0, const VectorXd& w)
{
	delta_diag = get_delta(w0, w);
}

std::vector<MatrixXd> Interaction::gen_R_k(VectorXd& w0, VectorXd& w)
{
	std::vector<MatrixXd> Rk(lasers.size());
	double _w0;
	double gamma;
	double r;

	for (size_t m = 0; m < lasers.size(); ++m)
	{
		//Rk.at(m).setZero();
		MatrixXd R = MatrixXd::Zero(atom->get_size(), atom->get_size());
		for (size_t j = 1; j < atom->get_size(); ++j)
		{
			for (size_t i = 0; i < j; ++i)
			{
				/*size_t i_decay = atom->get_decay_map()->get_index(atom->get(i)->get_label(), atom->get(j)->get_label());
				double a = atom->get_decay_map()->get_item(atom->get(i)->get_label(), atom->get(j)->get_label());  // (*atom->get_L0())(i, j) + (*atom->get_L0())(j, i);
				if (a == 0.) continue;*/

				r = 4 * std::pow(std::abs(rabimap.at(m)(i, j)), 2);
				if (r == 0.) continue;

				_w0 = abs(w0(i) - w0(j));
				gamma = atom->get_gamma(i) + atom->get_gamma(j);
				R(i, j) += lorentz(w(m), _w0, gamma, r);
				R(j, i) = R(i, j);
			}
		}
		Rk.at(m) = R;
	}
	return Rk;
}

Vector3d Interaction::gen_k_up(std::mt19937& gen, VectorXd& w0, VectorXd& w, size_t i, size_t j)
{
	Vector3d k_up = Vector3d::Zero();
	std::vector<MatrixXd> Rk = gen_R_k(w0, w);
	std::uniform_real_distribution<double> d(0, 1);
	size_t c = i;
	bool outer_flag = false;
	while (true)
	{
		double choice = d(gen);
		double chance = 0.;
		double norm = 0.;
		bool inner_flag = false;
		for (size_t m = 0; m < lasers.size(); ++m)
		{
			for (size_t l = 0; l < atom->get_size(); ++l) norm += Rk.at(m)(c, l);
		}
		for (size_t m = 0; m < lasers.size(); ++m)
		{
			for (size_t l = 0; l < atom->get_size(); ++l)
			{
				chance += Rk.at(m)(c, l) / norm;
				if (choice < chance)
				{
					k_up += lasers.at(m)->get_k_si();
					c = l;
					inner_flag = true;
					if (c == j) outer_flag = true;
					break;
				}
			}
			if (inner_flag) break;
		}
		if (outer_flag) break;
	}
	return k_up;
}

Vector3d Interaction::gen_velocity_change(std::mt19937& gen, VectorXd& w0, VectorXd& w, size_t i, size_t j, size_t f)
{
	std::normal_distribution<double> d(0, 1);
	Vector3d e_r;
	e_r << d(gen), d(gen), d(gen);
	e_r /= e_r.norm();

	Vector3d k_up = gen_k_up(gen, w0, w, i, j);

	double freq_down = atom->get(j)->get_freq() - atom->get(f)->get_freq();
	Vector3d k_down = freq_down / sc::c * e_r;

	return sc::h / (atom->get_mass() * sc::amu) * (k_up - k_down) * 1e6;
}

MatrixXd Interaction::gen_rates(const VectorXd& w0, const VectorXd& w)
{
	MatrixXd R = MatrixXd::Zero(atom->get_size(), atom->get_size());
	update_rates(R, w0, w);
	return R;
}

VectorXd Interaction::gen_rates_sum(const MatrixXd& R)
{
	VectorXd R_sum = VectorXd::Zero(atom->get_size());
	update_rates_sum(R_sum, R);
	return R_sum;
}

void Interaction::update_rates(MatrixXd& R, const VectorXd& w0, const VectorXd& w)
{
	R.setZero();
	double _w0;
	double gamma;
	double r;
	for (size_t m = 0; m < lasers.size(); ++m)
	{
		for (size_t j = 1; j < atom->get_size(); ++j)
		{
			for (size_t i = 0; i < j; ++i)
			{
				
				/*double a = atom->get_decay_map()->get_item(atom->get(i)->get_label(), atom->get(j)->get_label());  // (*atom->get_L0())(i, j) + (*atom->get_L0())(j, i);
				if (a == 0.) continue;*/

				r = 4 * std::pow(std::abs(rabimap.at(m)(i, j)), 2);
				if (r == 0) continue;

				_w0 = abs(w0(i) - w0(j));
				gamma = atom->get_gamma(i) + atom->get_gamma(j);
				R(i, j) += lorentz(w(m), _w0, gamma, r);
				R(j, i) = R(i, j);
			}
		}
	}
}

void Interaction::update_rates_sum(VectorXd& R_sum, const MatrixXd& R)
{
	R_sum = R.colwise().sum();
}

MatrixXcd Interaction::gen_hamiltonian(const VectorXd& w0, const VectorXd& w)
{
	MatrixXcd H = MatrixXcd::Zero(atom->get_size(), atom->get_size());
	H.diagonal() = get_delta(w0, w);
	update_hamiltonian_off(H);
	return H;
}

void Interaction::update_hamiltonian(MatrixXcd& H, const VectorXd& w0, const VectorXd& w)
{
	H.setZero();
	update_hamiltonian_diag(H, w0, w);
	update_hamiltonian_off(H);
}

void Interaction::update_hamiltonian_off(MatrixXcd& H)
{
	for (int m = 0; m < lasers.size(); ++m)
	{
		H += rabimap.at(m);
	}
}

void Interaction::update_hamiltonian_diag(MatrixXcd& H, const VectorXd& w0, const VectorXd& w)
{
	H.diagonal() = get_delta(w0, w);
}

void Interaction::update_hamiltonian(MatrixXcd& H, const VectorXd& w0, const VectorXd& w, double t)
{
	H.setZero();
	update_hamiltonian_diag(H, w0, w);
	update_hamiltonian_off(H, w, t);
}

void Interaction::update_hamiltonian_off(MatrixXcd& H, const VectorXd& w, double t)
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

MatrixXcd Interaction::gen_hamiltonian_leaky(const VectorXd& w0, const VectorXd& w)
{
	MatrixXcd H = MatrixXcd::Zero(atom->get_size(), atom->get_size());
	update_hamiltonian_leaky_diag(H, w0, w);
	update_hamiltonian_off(H);
	return H;
}

void Interaction::update_hamiltonian_leaky(MatrixXcd& H, const VectorXd& w0, const VectorXd& w, double t)
{
	H.setZero();
	update_hamiltonian_leaky_diag(H, w0, w);
	update_hamiltonian_off(H, w, t);
}

void Interaction::update_hamiltonian_leaky_diag(MatrixXcd& H, const VectorXd& w0, const VectorXd& w)
{
	H.diagonal() = get_delta(w0, w) - sc::i * 0.5 * (*atom->get_Lsum());
}

std::vector<std::vector<VectorXd>> Interaction::rates(
	const std::vector<double>& t, const std::vector<VectorXd>& delta, const std::vector<Vector3d>& v, std::vector<VectorXd>& x0, const bool analytic)
{
	std::vector<std::vector<VectorXd>> results(x0.size());
	for (auto& vec : results) vec.reserve(t.size());

	VectorXd w0 = *atom->get_w0();

	std::vector<size_t> n_vec(x0.size());
	for (size_t i = 0; i < x0.size(); ++i)
	{
		n_vec.at(i) = i;
	}

	std::atomic<size_t> progress{0};
	std::thread worker([this, &t, &x0, &delta, &v, &analytic, &w0, &results, &progress, &n_vec]
	{
		std::for_each(std::execution::par, n_vec.begin(), n_vec.end(),
			[this, &t, &x0, &delta, &v, &analytic, &w0, &results, &progress](size_t i)
			{
				thread_local VectorXd w;
				thread_local MatrixXd R;
				thread_local VectorXd R_sum;

				const Index w_size = lasers.size();
				const Index H_size = atom->get_size();

				if (w.size() != w_size)
					w.resize(w_size);

				if (R.rows() != H_size || R.cols() != H_size)
					R.resize(H_size, H_size);

				if (R_sum.size() != H_size)
					R_sum.resize(H_size);

				update_w(w, delta.at(i), v.at(i));
				update_rates(R, w0, w);
				update_rates_sum(R_sum, R);

				size_t n = 0;

				if (analytic)
				{
					MatrixXd R_diag = (R_sum + *atom->get_Lsum()).asDiagonal();
					MatrixXd R_exp = (R + *atom->get_L0()) - R_diag;

					RateExponential rate_op(R_exp);
					for (auto it = t.cbegin(); it != t.cend(); ++it)
					{
						results.at(i).push_back(rate_op(*it, x0.at(i)));
					}
				}
				else
				{
					if (dense)
					{
						d_dopri5_vd_type dopri5 = make_dense_output(atol, rtol, dt_max, dopri5_vd_type());
						n = integrate_times(dopri5, f_rate_equations(R, R_sum, *atom->get_L0(), *atom->get_Lsum()),
							x0.at(i), t.begin(), t.end(), dt, push_back_VectorXd(results.at(i)));
					}
					else if (controlled)
					{
						c_dopri5_vd_type dopri5 = make_controlled(atol, rtol, dt_max, dopri5_vd_type());
						n = integrate_times(dopri5, f_rate_equations(R, R_sum, *atom->get_L0(), *atom->get_Lsum()),
							x0.at(i), t.begin(), t.end(), dt, push_back_VectorXd(results.at(i)));
					}
					else
					{
						n = integrate_times(rk4_vd_type(), f_rate_equations(R, R_sum, *atom->get_L0(), *atom->get_Lsum()),
							x0.at(i), t.begin(), t.end(), dt, push_back_VectorXd(results.at(i)));
					}
				}
				progress.fetch_add(1, std::memory_order_relaxed);
			}
		);
	});

	while (progress.load() < x0.size()) {
		printf("\r\033[92mSolving rate equations ... %3.2f %%", 100.0 * progress.load() / x0.size());
		std::this_thread::sleep_for(std::chrono::milliseconds(100));
	}

	worker.join();

	printf("\r\033[92mSolving rate equations ... 100.00 %%\033[0m");
	printf("\n");

	return results;
}

std::vector<std::vector<VectorXcd>> Interaction::schroedinger(
	const std::vector<double>& t, const std::vector<VectorXd>& delta, const std::vector<Vector3d>& v, std::vector<VectorXcd>& x0)
{
	std::vector<std::vector<VectorXcd>> results(x0.size());
	for (auto& vec : results) vec.reserve(t.size());

	VectorXd w0 = *atom->get_w0();

	std::vector<size_t> n_vec(x0.size());
	for (size_t i = 0; i < x0.size(); ++i)
	{
		n_vec.at(i) = i;
	}

	std::atomic<size_t> progress{0};
	std::thread worker([this, &t, &x0, &delta, &v, &w0, &results, &progress, &n_vec]
	{
		std::for_each(std::execution::par, n_vec.begin(), n_vec.end(),
			[this, &t, &x0, &delta, &v, &w0, &results, &progress](size_t i)
			{
				thread_local VectorXd w;
				thread_local MatrixXcd H;

				const Index w_size = lasers.size();
				const Index H_size = atom->get_size();

				if (w.size() != w_size)
					w.resize(w_size);

				if (H.rows() != H_size || H.cols() != H_size)
					H.resize(H_size, H_size);

				update_w(w, delta.at(i), v.at(i));
				update_hamiltonian(H, w0, w);

				size_t n = 0;

				if (dense)
				{
					if (time_dependent)
					{
						d_dopri5_vcd_type d_dopri5 = make_dense_output(atol, rtol, dt_max, dopri5_vcd_type());
						n = integrate_times(d_dopri5, f_schroedinger_t(H, w0, w, *this),
							x0.at(i), t.begin(), t.end(), dt, push_back_VectorXcd(results.at(i)));
					}
					else
					{
						d_dopri5_vcd_type d_dopri5 = make_dense_output(atol, rtol, dt_max, dopri5_vcd_type());
						n = integrate_times(d_dopri5, f_schroedinger(H),
							x0.at(i), t.begin(), t.end(), dt, push_back_VectorXcd(results.at(i)));
					}
				}
				else if (controlled)
				{
					if (time_dependent)
					{
						c_dopri5_vcd_type c_dopri5 = make_controlled(atol, rtol, dt_max, dopri5_vcd_type());
						n = integrate_times(c_dopri5, f_schroedinger_t(H, w0, w, *this),
							x0.at(i), t.begin(), t.end(), dt, push_back_VectorXcd(results.at(i)));
					}
					else
					{
						c_dopri5_vcd_type c_dopri5 = make_controlled(atol, rtol, dt_max, dopri5_vcd_type());
						n = integrate_times(c_dopri5, f_schroedinger(H),
							x0.at(i), t.begin(), t.end(), dt, push_back_VectorXcd(results.at(i)));
					}
				}
				else
				{
					if (time_dependent)
					{
						n = integrate_times(rk4_vcd_type(), f_schroedinger_t(H, w0, w, *this),
							x0.at(i), t.begin(), t.end(), dt, push_back_VectorXcd(results.at(i)));
					}
					else
					{
						n = integrate_times(rk4_vcd_type(), f_schroedinger(H),
							x0.at(i), t.begin(), t.end(), dt, push_back_VectorXcd(results.at(i)));
					}
				}
				progress.fetch_add(1, std::memory_order_relaxed);
			}
		);
	});

	while (progress.load() < x0.size()) {
		printf("\r\033[92mSolving Schroedinger equation ... %3.2f %%", 100.0 * progress.load() / x0.size());
		std::this_thread::sleep_for(std::chrono::milliseconds(100));
	}

	worker.join();

	printf("\r\033[92mSolving Schroedinger equation ... 100.00 %%\033[0m");
	printf("\n");

	return results;
}

std::vector<std::vector<MatrixXcd>> Interaction::master(
	const std::vector<double>& t, const std::vector<VectorXd>& delta, const std::vector<Vector3d>& v, std::vector<MatrixXcd>& x0)
{
	std::vector<std::vector<MatrixXcd>> results(x0.size());
	for (auto& vec : results) vec.reserve(t.size());

	VectorXd w0 = *atom->get_w0();
	MatrixXd L0 = *atom->get_L0();
	MatrixXd L1 = *atom->get_L1();

	std::vector<size_t> n_vec(x0.size());
	for (size_t i = 0; i < x0.size(); ++i)
	{
		n_vec.at(i) = i;
	}

	std::atomic<size_t> progress{0};
	std::thread worker([this, &t, &x0, &delta, &v, &w0, &L0, &L1, &results, &progress, &n_vec]
	{
		std::for_each(std::execution::par, n_vec.begin(), n_vec.end(),
			[this, &t, &x0, &delta, &v, &w0, &L0, &L1, &results, &progress](size_t i)
			{
				thread_local VectorXd w;
				thread_local MatrixXcd H;

				const Index w_size = lasers.size();
				const Index H_size = atom->get_size();

				if (w.size() != w_size)
					w.resize(w_size);

				if (H.rows() != H_size || H.cols() != H_size)
					H.resize(H_size, H_size);

				update_w(w, delta.at(i), v.at(i));
				update_hamiltonian(H, w0, w);

				size_t n = 0;
				Eigen::Map<Eigen::VectorXcd> x0_vec(x0.at(i).data(), x0.at(i).size());

				if (dense)
				{
					if (time_dependent)
					{
						d_dopri5_vcd_type d_dopri5 = make_dense_output(atol, rtol, dt_max, dopri5_vcd_type());
						n = integrate_times(d_dopri5, f_master_t(H, L0, L1, w0, w, *this),
							x0_vec, t.begin(), t.end(), dt,
							push_back_VectorXcd_as_MatrixXcd(results.at(i), H.rows(), H.cols()));
					}
					else
					{
						d_dopri5_vcd_type d_dopri5 = make_dense_output(atol, rtol, dt_max, dopri5_vcd_type());
						n = integrate_times(d_dopri5, f_master(H, L0, L1),
							x0_vec, t.begin(), t.end(), dt,
							push_back_VectorXcd_as_MatrixXcd(results.at(i), H.rows(), H.cols()));
					}
				}
				else if (controlled)
				{
					if (time_dependent)
					{
						c_dopri5_vcd_type c_dopri5 = make_controlled(atol, rtol, dt_max, dopri5_vcd_type());
						n = integrate_times(c_dopri5, f_master_t(H, L0, L1, w0, w, *this),
							x0_vec, t.begin(), t.end(), dt,
							push_back_VectorXcd_as_MatrixXcd(results.at(i), H.rows(), H.cols()));
					}
					else
					{
						c_dopri5_vcd_type c_dopri5 = make_controlled(atol, rtol, dt_max, dopri5_vcd_type());
						n = integrate_times(c_dopri5, f_master(H, L0, L1),
							x0_vec, t.begin(), t.end(), dt,
							push_back_VectorXcd_as_MatrixXcd(results.at(i), H.rows(), H.cols()));
					}
				}
				else
				{
					if (time_dependent)
					{
						n = integrate_times(rk4_vcd_type(), f_master_t(H, L0, L1, w0, w, *this),
							x0_vec, t.begin(), t.end(), dt,
							push_back_VectorXcd_as_MatrixXcd(results.at(i), H.rows(), H.cols()));
					}
					else
					{
						n = integrate_times(rk4_vcd_type(), f_master(H, L0, L1),
							x0_vec, t.begin(), t.end(), dt,
							push_back_VectorXcd_as_MatrixXcd(results.at(i), H.rows(), H.cols()));
					}
				}
				progress.fetch_add(1, std::memory_order_relaxed);
			}
		);
	});

	while (progress.load() < x0.size()) {
		printf("\r\033[92mSolving master equation ... %3.2f %%", 100.0 * progress.load() / x0.size());
		std::this_thread::sleep_for(std::chrono::milliseconds(100));
	}

	worker.join();

	printf("\r\033[92mSolving master equation ... 100.00 %%\033[0m");
	printf("\n");

	return results;
}

std::vector<std::vector<VectorXcd>> Interaction::mc_master(
	const std::vector<double>& t, const std::vector<VectorXd>& delta, std::vector<Vector3d>& v, std::vector<VectorXcd>& x0, const bool dynamics)
{
	// if (controlled || dense) printf("\r\033[93mWarning: Interaction.mc_master does not support controlled or dense steppers.\033[0m");
	std::vector<std::vector<VectorXcd>> results(x0.size());
	for (auto& vec : results) vec.reserve(t.size());

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

	std::vector<size_t> n_vec(x0.size());
	for (size_t i = 0; i < x0.size(); ++i)
	{
		n_vec.at(i) = i;
	}

	std::atomic<size_t> progress{0};
	std::thread worker([this, &x0, &delta, &v, dynamics, &c_i, &c_j, &c_a, &t, &results, &progress, &n_vec]
	{
		std::for_each(std::execution::par, n_vec.begin(), n_vec.end(),
			[this, &x0, &delta, &v, dynamics, &c_i, &c_j, &c_a, &t, &results, &progress](size_t n)
			{
				thread_local std::random_device rd;
				thread_local std::mt19937 gen(rd());
				thread_local std::uniform_real_distribution<double> d(0, 1);

				thread_local VectorXd w0;
				thread_local VectorXd w;
				thread_local MatrixXcd H;

				const Index w_size = lasers.size();
				const Index H_size = atom->get_size();

				if (w0.size() != H_size)
					w0.resize(H_size);

				if (w.size() != w_size)
					w.resize(w_size);

				if (H.rows() != H_size || H.cols() != H_size)
					H.resize(H_size, H_size);
				
				w0 = *atom->get_w0();
				update_w(w, delta.at(n), v.at(n), dynamics);
				H.setZero();
				update_hamiltonian_off(H);
				if (!time_dependent) update_hamiltonian_leaky_diag(H, w0, w);

				Vector3d v_temp = Vector3d::Zero();
				v_temp += v.at(n);
				size_t i = gen_index(x0.at(n).cwiseAbs2(), d, gen);
				double r = d(gen);
				double p = 0;
				double p_n = 0;

				size_t i_t = 1;
				double _t = t.front();
				double _t1 = 0;
				double _dt = dt;
				results.at(n).push_back(x0.at(n));
				bool break_loop = false;
				while (true)
				{
					_t1 = _t + dt;
					_dt = dt;
					while (t.at(i_t) < _t + _dt)
					{
						_dt = t.at(i_t) - _t;
						if (time_dependent) rk4_vcd_type().do_step(
							f_schroedinger_leaky_t(H, w0, w, std::ref(*this)), x0.at(n), _t, _dt);
						else rk4_vcd_type().do_step(f_schroedinger_leaky(H), x0.at(n), _t, _dt);

						results.at(n).push_back(x0.at(n) / sqrt(x0.at(n).cwiseAbs2().sum()));
						_t += _dt;
						_dt = _t1 - _t;
						if (++i_t == t.size())
						{
							break_loop = true;
							break;
						}
					}
					if (break_loop) break;
					if (_dt > 0)
					{
						if (time_dependent) rk4_vcd_type().do_step(
							f_schroedinger_leaky_t(H, w0, w, std::ref(*this)), x0.at(n), _t, _dt);
						else rk4_vcd_type().do_step(f_schroedinger_leaky(H), x0.at(n), _t, _dt);
						_t = _t1;
					}
					if (x0.at(n).cwiseAbs2().sum() < r)
					{
						p = 0;
						p_n = 0;
						for (size_t j = 0; j < c_i.size(); ++j)
						{
							p += c_a.at(j) * std::pow(std::abs(x0.at(n)(c_i.at(j))), 2);
						}
						r = d(gen);
						for (size_t j = 0; j < c_i.size(); ++j)
						{
							p_n += c_a.at(j) * std::pow(std::abs(x0.at(n)(c_i.at(j))), 2) / p;
							if (p_n >= r)
							{
								if (dynamics)
								{
									v_temp += gen_velocity_change(gen, w0, w, i, c_i.at(j), c_j.at(j));
									update_w(w, delta.at(n), v_temp, dynamics);
									if (!time_dependent) update_hamiltonian_leaky_diag(H, w0, w);
								}

								i = c_j.at(j);
								x0.at(n).fill(0);
								x0.at(n)(c_j.at(j)) = 1.;
								break;
							}
						}
						r = d(gen);
					}
				}
				if (dynamics) v.at(n) += v_temp;
				progress.fetch_add(1, std::memory_order_relaxed);
			}
		);
	});

	while (progress.load() < x0.size())
	{
		printf("\r\033[92mSolving MC master equation ... %3.2f %%", 100.0 * progress.load() / x0.size());
		std::this_thread::sleep_for(std::chrono::milliseconds(100));
	}

	worker.join();

	printf("\r\033[92mSolving MC master equation ... 100.00 %%\033[0m");
	printf("\n");

	return results;
}
