// dllmain.cpp : Defines the entry point for the DLL application.
#include "qspec_cpp.h"

// Polarization
void* polarization_construct()
{
    return static_cast<void*>(new Polarization());
}

void polarization_destruct(void* polarization)
{
    delete static_cast<Polarization*>(polarization);
}

void polarization_init(void* polarization, std::complex<double>* vec,
    double* q_axis, bool vec_as_q)
{
    Polarization* _polarization = static_cast<Polarization*>(polarization);

    Vector3cd _vec;
    _vec << vec[0], vec[1], vec[2];
    Vector3d _q_axis;
    _q_axis << q_axis[0], q_axis[1], q_axis[2];
    _polarization->init(_vec, _q_axis, vec_as_q);
}

void polarization_def_q_axis(void* polarization, double* q_axis, bool q_fixed)
{
    Polarization* _polarization = static_cast<Polarization*>(polarization);

    Vector3d _q_axis;
    _q_axis << q_axis[0], q_axis[1], q_axis[2];
    _polarization->def_q_axis(_q_axis, q_fixed);
}

double* polarization_get_q_axis(void* polarization)
{
    Polarization* _polarization = static_cast<Polarization*>(polarization);
    return _polarization->get_q_axis()->data();
}

std::complex<double>* polarization_get_x(void* polarization)
{
    Polarization* _polarization = static_cast<Polarization*>(polarization);
    return _polarization->get_x()->data();
}

std::complex<double>* polarization_get_q(void* polarization)
{
    Polarization* _polarization = static_cast<Polarization*>(polarization);
    return _polarization->get_q()->data();
}


// Laser
void* laser_construct()
{
    return static_cast<void*>(new Laser());
}

void laser_destruct(void* laser)
{
    delete static_cast<Laser*>(laser);
}

void laser_init(void* laser, double freq, double intensity, void* polarization, double* k)
{
    Laser* _laser = static_cast<Laser*>(laser);
    Polarization* _polarization = static_cast<Polarization*>(polarization);

    Vector3d _k = cast_Vector3d(k);
    _laser->init(freq, intensity, _polarization, _k);
}

double laser_get_freq(void* laser)
{
    Laser* _laser = static_cast<Laser*>(laser);
    return _laser->get_freq();
}

void laser_set_freq(void* laser, double freq)
{
    Laser* _laser = static_cast<Laser*>(laser);
    return _laser->set_freq(freq);
}

double laser_get_intensity(void* laser)
{
    Laser* _laser = static_cast<Laser*>(laser);
    return _laser->get_intensity();
}

void laser_set_intensity(void* laser, double intensity)
{
    Laser* _laser = static_cast<Laser*>(laser);
    return _laser->set_intensity(intensity);
}

void* laser_get_polarization(void* laser)
{
    Laser* _laser = static_cast<Laser*>(laser);
    return _laser->get_polarization();
}

void laser_set_polarization(void* laser, void* polarization)
{
    Laser* _laser = static_cast<Laser*>(laser);
    Polarization* _polarization = static_cast<Polarization*>(polarization);

    return _laser->set_polarization(_polarization);
}

double* laser_get_k(void* laser)
{
    Laser* _laser = static_cast<Laser*>(laser);
    return _laser->get_k()->data();
}

void laser_set_k(void* laser, double* k)
{
    Laser* _laser = static_cast<Laser*>(laser);

    Vector3d _k = cast_Vector3d(k);
    return _laser->set_k(_k);
}

std::complex<double>* laser_get_kpol(void* laser, bool electric, size_t k, double* q_axis)
{
    Laser* _laser = static_cast<Laser*>(laser);

    Vector3d _q_axis = cast_Vector3d(q_axis);
    return _laser->get_kpol(electric, k, _q_axis).array().data();
}


//Environment
void* environment_construct()
{
    return static_cast<void*>(new Environment());
}

void environment_destruct(void* env)
{
    delete static_cast<Environment*>(env);
}

double environment_get_E(void* env)
{
    Environment* _env = static_cast<Environment*>(env);
    return _env->get_E();
}

double environment_get_B(void* env)
{
    Environment* _env = static_cast<Environment*>(env);
    return _env->get_B();
}

double* environment_get_e_E(void* env)
{
    Environment* _env = static_cast<Environment*>(env);
    return _env->get_e_E()->data();
}

double* environment_get_e_B(void* env)
{
    Environment* _env = static_cast<Environment*>(env);
    return _env->get_e_B()->data();
}

void environment_set_E(void* env, double* E)
{
    Environment* _env = static_cast<Environment*>(env);

    Vector3d Evec;
    Evec << E[0], E[1], E[2];
    _env->set_E(Evec);
}

void environment_set_B(void* env, double* B)
{
    Environment* _env = static_cast<Environment*>(env);

    Vector3d Bvec;
    Bvec << B[0], B[1], B[2];
    _env->set_B(Bvec);
}

void environment_set_E_double(void* env, double E)
{
    Environment* _env = static_cast<Environment*>(env);
    _env->set_E(E);
}

void environment_set_B_double(void* env, double B)
{
    Environment* _env = static_cast<Environment*>(env);
    _env->set_B(B);
}


// State
void* state_construct()
{
    return static_cast<void*>(new State());
}

void state_destruct(void* state)
{
    delete static_cast<State*>(state);
}

void state_init(void* state, double freq_0, double* s, double* l, double j, double i, double f, double m,
    bool parity, double* jj, size_t ls_size, double* hyper_const, double gj, double gi, char* label)
{
    State* _state = static_cast<State*>(state);

    std::vector<double> _s = cast_samples_double(s, ls_size);
    std::vector<double> _l = cast_samples_double(l, ls_size);
    std::vector<double> _jj = cast_samples_double(jj, ls_size);
    _state->init(freq_0, j, i, f, m, parity, _s, _l, _jj, hyper_const, gj, gi, std::string(label));
}

void state_reset(void* state)
{
    State* _state = static_cast<State*>(state);
    return _state->reset();
}

double state_get_shift(void* state)
{
    State* _state = static_cast<State*>(state);
    return _state->get_shift();
}

double state_get_freq_j(void* state)
{
    State* _state = static_cast<State*>(state);
    return _state->get_freq_j();
}

void state_set_freq_j(void* state, double freq_j)
{
    State* _state = static_cast<State*>(state);
    _state->set_freq_j(freq_j);
}

double state_get_freq(void* state)
{
    State* _state = static_cast<State*>(state);
    return _state->get_freq();
}

void state_set_freq(void* state, double freq)
{
    State* _state = static_cast<State*>(state);
    _state->set_freq(freq);
}

double state_get_j(void* state)
{
    State* _state = static_cast<State*>(state);
    return _state->get_j();
}

void state_set_j(void* state, double j)
{
    State* _state = static_cast<State*>(state);
    _state->set_j(j);
}

double state_get_i(void* state)
{
    State* _state = static_cast<State*>(state);
    return _state->get_i();
}

void state_set_i(void* state, double i)
{
    State* _state = static_cast<State*>(state);
    _state->set_i(i);
}

double state_get_f(void* state)
{
    State* _state = static_cast<State*>(state);
    return _state->get_f();
}

void state_set_f(void* state, double f)
{
    State* _state = static_cast<State*>(state);
    _state->set_f(f);
}

double state_get_m(void* state)
{
    State* _state = static_cast<State*>(state);
    return _state->get_m();
}

void state_set_m(void* state, double m)
{
    State* _state = static_cast<State*>(state);
    _state->set_m(m);
}

double* state_get_hyper_const(void* state)
{
    State* _state = static_cast<State*>(state);
    return _state->get_hyper_const();
}

void state_set_hyper_const(void* state, double* hyper_const)
{
    State* _state = static_cast<State*>(state);
    _state->set_hyper_const(hyper_const);
}

double state_get_gj(void* state)
{
    State* _state = static_cast<State*>(state);
    return _state->get_gj();
}

void state_set_gj(void* state, double gj)
{
    State* _state = static_cast<State*>(state);
    _state->set_gj(gj);
}

double state_get_gi(void* state)
{
    State* _state = static_cast<State*>(state);
    return _state->get_gi();
}

void state_set_gi(void* state, double gi)
{
    State* _state = static_cast<State*>(state);
    _state->set_gi(gi);
}

const char* state_get_label(void* state)
{
    State* _state = static_cast<State*>(state);

    char* ret = new char[_state->get_label().length() + 1];
    std::strcpy(ret, _state->get_label().c_str());
    return ret;
}

void state_set_label(void* state, char* label)
{
    State* _state = static_cast<State*>(state);
    _state->set_label(std::string(label));
}


// DecayMap
void* decaymap_construct()
{
    return static_cast<void*>(new DecayMap());
}

void decaymap_destruct(void* decays)
{
    delete static_cast<DecayMap*>(decays);
}

void decaymap_add_decay(void* decays, char* state_0, char* state_1, double* ae, size_t ae_size, double* am, size_t am_size, bool single_leading_order)
{
    DecayMap* _decays = static_cast<DecayMap*>(decays);

    std::vector<double> _ae = cast_samples_double(ae, ae_size);
    std::vector<double> _am = cast_samples_double(am, am_size);
    _decays->add_decay(std::string(state_0), std::string(state_1), _ae, _am, single_leading_order);
}

const char* decaymap_get_label(void* decays, size_t i, size_t j)
{
    DecayMap* _decays = static_cast<DecayMap*>(decays);

    if (i == 0)
    {
        char* ret = new char[_decays->get_states_0()->at(j).length() + 1];
        std::strcpy(ret, _decays->get_states_0()->at(j).c_str());
        return ret;
    }
    else if (i == 1)
    {
        char* ret = new char[_decays->get_states_1()->at(j).length() + 1];
        std::strcpy(ret, _decays->get_states_1()->at(j).c_str());
        return ret;
    }
    else return "";

}

size_t decaymap_get_size(void* decays)
{
    DecayMap* _decays = static_cast<DecayMap*>(decays);
    return _decays->get_size();
}

size_t decaymap_get_k_em_max(void* decays)
{
    DecayMap* _decays = static_cast<DecayMap*>(decays);
    return _decays->get_k_em_max();
}

void decaymap_set_k_em_max(void* decays, size_t k_em_max)
{
    DecayMap* _decays = static_cast<DecayMap*>(decays);
    return _decays->set_k_em_max(k_em_max);
}

double decaymap_get_a_i(void* decays, char* state_0, char* state_1, bool parity_equal)
{
    DecayMap* _decays = static_cast<DecayMap*>(decays);
    return _decays->get_a(std::string(state_0), std::string(state_1), parity_equal);
}

double decaymap_get_ae_ik(void* decays, char* state_0, char* state_1, size_t k)
{
    DecayMap* _decays = static_cast<DecayMap*>(decays);
    return _decays->get_ae(std::string(state_0), std::string(state_1), k);
}

double decaymap_get_am_ik(void* decays, char* state_0, char* state_1, size_t k)
{
    DecayMap* _decays = static_cast<DecayMap*>(decays);
    return _decays->get_am(std::string(state_0), std::string(state_1), k);
}


// Atom
void* atom_construct()
{
    return static_cast<void*>(new Atom());
}

void atom_destruct(void* atom)
{
    delete static_cast<Atom*>(atom);
}

void atom_update(void* atom)
{

    Atom* _atom = static_cast<Atom*>(atom);
    _atom->update();
}

void atom_set_env(void* atom, void* env)
{
    Atom* _atom = static_cast<Atom*>(atom);
    Environment* _env = static_cast<Environment*>(env);

    _atom->set_env(_env);
}

void atom_add_state(void* atom, void* state)
{
    Atom* _atom = static_cast<Atom*>(atom);
    State* _state = static_cast<State*>(state);

    _atom->add_state(_state);
}

void atom_clear_states(void* atom)
{
    Atom* _atom = static_cast<Atom*>(atom);
    _atom->clear_states();
}

void* atom_get_decay_map(void* atom)
{
    Atom* _atom = static_cast<Atom*>(atom);
    return _atom->get_decay_map();
}

void atom_set_decay_map(void* atom, void* decays)
{
    Atom* _atom = static_cast<Atom*>(atom);
    DecayMap* _decays = static_cast<DecayMap*>(decays);

    return _atom->set_decay_map(_decays);
}

double atom_get_gamma(void* atom, size_t i)
{
    Atom* _atom = static_cast<Atom*>(atom);
    return _atom->get_gamma(i);
}

double atom_get_mass(void* atom)
{
    Atom* _atom = static_cast<Atom*>(atom);
    return _atom->get_mass();
}

void atom_set_mass(void* atom, double mass)
{
    Atom* _atom = static_cast<Atom*>(atom);
    _atom->set_mass(mass);
}

size_t atom_get_size(void* atom)
{
    Atom* _atom = static_cast<Atom*>(atom);
    return _atom->get_size();
}

size_t atom_get_gs_size(void* atom)
{
    Atom* _atom = static_cast<Atom*>(atom);
    return _atom->get_gs()->size();
}

size_t* atom_get_gs(void* atom)
{
    Atom* _atom = static_cast<Atom*>(atom);
    return _atom->get_gs()->data();
}

int* atom_get_ek(void* atom, size_t k)
{
    Atom* _atom = static_cast<Atom*>(atom);

    MatrixXi* ek = new MatrixXi(_atom->get_size(), _atom->get_size());
    *ek = _atom->get_ek(k);
    return ek->data();
}

int* atom_get_mk(void* atom, size_t k)
{
    Atom* _atom = static_cast<Atom*>(atom);

    MatrixXi* mk = new MatrixXi(_atom->get_size(), _atom->get_size());
    *mk = _atom->get_mk(k);
    return mk->data();
}

int* atom_get_emk(void* atom, size_t k)
{
    Atom* _atom = static_cast<Atom*>(atom);

    MatrixXi* emk = new MatrixXi(_atom->get_size(), _atom->get_size());
    *emk = _atom->get_emk(k);
    return emk->data();
}

double* atom_get_d_em(void* atom, size_t k)
{
    Atom* _atom = static_cast<Atom*>(atom);

    MatrixXd* d_em = new MatrixXd(_atom->get_size(), _atom->get_size());
    *d_em = _atom->get_d_em(k);
    return d_em->data();
}

double* atom_get_L0(void* atom)
{
    Atom* _atom = static_cast<Atom*>(atom);
    return _atom->get_L0()->data();
}

double* atom_get_L1(void* atom)
{
    Atom* _atom = static_cast<Atom*>(atom);
    return _atom->get_L1()->data();
}

int atom_scattering_rate_4pi(void* atom, double* results, size_t* k, size_t k_size,
    std::complex<double>* rho, size_t rho_size, bool as_density_matrix,
    size_t* i, size_t i_size, size_t* f, size_t f_size)
{
    Atom* _atom = static_cast<Atom*>(atom);

    std::vector<size_t> _k = cast_samples_size_t(k, k_size);
    for (size_t _ki : _k) if (_ki < 1 || _ki > _atom->get_decay_map()->get_k_em_max()) return -1;

    std::vector<MatrixXcd> _rho;
    if (as_density_matrix) _rho = cast_samples_MatrixXcd(rho, rho_size, _atom->get_size());
    else _rho = cast_samples_VectorXcd_as_MatrixXcd(rho, rho_size, _atom->get_size());

    std::vector<size_t> _i = cast_samples_size_t(i, i_size);
    std::vector<size_t> _f = cast_samples_size_t(f, f_size);
    _atom->scattering_rate(results, _k, _rho, _i, _f);

    return 0;
}

int atom_scattering_rate_k(
    void* atom, double* results, size_t* k, size_t k_size,
    std::complex<double>* rho, size_t rho_size, bool as_density_matrix,
    double* k_vec, size_t k_vec_size,
    size_t* i, size_t i_size, size_t* f, size_t f_size)
{
    Atom* _atom = static_cast<Atom*>(atom);

    std::vector<size_t> _k = cast_samples_size_t(k, k_size);
    for (size_t _ki : _k) if (_ki < 1 || _ki > _atom->get_decay_map()->get_k_em_max()) return -1;

    std::vector<MatrixXcd> _rho;
    if (as_density_matrix) _rho = cast_samples_MatrixXcd(rho, rho_size, _atom->get_size());
    else _rho = cast_samples_VectorXcd_as_MatrixXcd(rho, rho_size, _atom->get_size());
    
    std::vector<Vector3d> _k_vec = cast_samples_Vector3d(k_vec, k_vec_size);
    std::vector<size_t> _i = cast_samples_size_t(i, i_size);
    std::vector<size_t> _f = cast_samples_size_t(f, f_size);

    _atom->scattering_rate(results, _k, _rho, _k_vec, _i, _f);

    return 0;
}

int atom_scattering_rate_k_tp(
    void* atom, double* results, size_t* k, size_t k_size,
    std::complex<double>* rho, size_t rho_size, bool as_density_matrix,
    double* theta, double* phi, size_t k_vec_size,
    size_t* i, size_t i_size, size_t* f, size_t f_size)
{
    Atom* _atom = static_cast<Atom*>(atom);

    std::vector<size_t> _k = cast_samples_size_t(k, k_size);
    for (size_t _ki : _k) if (_ki < 1 || _ki > _atom->get_decay_map()->get_k_em_max()) return -1;

    std::vector<MatrixXcd> _rho;
    if (as_density_matrix) _rho = cast_samples_MatrixXcd(rho, rho_size, _atom->get_size());
    else _rho = cast_samples_VectorXcd_as_MatrixXcd(rho, rho_size, _atom->get_size());

    std::vector<Vector3d> _k_vec = cast_samples_theta_phi_er(theta, phi, k_vec_size);
    std::vector<size_t> _i = cast_samples_size_t(i, i_size);
    std::vector<size_t> _f = cast_samples_size_t(f, f_size);

    _atom->scattering_rate(results, _k, _rho, _k_vec, _i, _f);

    return 0;
}

int atom_scattering_rate_qk(
    void* atom, double* results, size_t* k, size_t k_size,
    std::complex<double>* rho, size_t rho_size, bool as_density_matrix,
    double* k_vec, std::complex<double>* x_vec, size_t k_vec_size,
    size_t* i, size_t i_size, size_t* f, size_t f_size)
{
    Atom* _atom = static_cast<Atom*>(atom);

    std::vector<size_t> _k = cast_samples_size_t(k, k_size);
    for (size_t _ki : _k) if (_ki < 1 || _ki > _atom->get_decay_map()->get_k_em_max()) return -1;

    std::vector<MatrixXcd> _rho;
    if (as_density_matrix) _rho = cast_samples_MatrixXcd(rho, rho_size, _atom->get_size());
    else _rho = cast_samples_VectorXcd_as_MatrixXcd(rho, rho_size, _atom->get_size());
    
    std::vector<Vector3d> _k_vec = cast_samples_Vector3d(k_vec, k_vec_size);
    std::vector<Vector3cd> _x_vec = cast_samples_Vector3cd(x_vec, k_vec_size);
    std::vector<size_t> _i = cast_samples_size_t(i, i_size);
    std::vector<size_t> _f = cast_samples_size_t(f, f_size);
    _atom->scattering_rate(results, _k, _rho, _k_vec, _x_vec, _i, _f);

    return 0;
}

int atom_scattering_rate_qk_xb(
    void* atom, double* results, size_t* k, size_t k_size,
    std::complex<double>* rho, size_t rho_size, bool as_density_matrix,
    double* k_vec, size_t x_vec, size_t k_vec_size,
    size_t* i, size_t i_size, size_t* f, size_t f_size)
{
    Atom* _atom = static_cast<Atom*>(atom);

    std::vector<size_t> _k = cast_samples_size_t(k, k_size);
    for (size_t _ki : _k) if (_ki < 1 || _ki > _atom->get_decay_map()->get_k_em_max()) return -1;

    std::vector<MatrixXcd> _rho;
    if (as_density_matrix) _rho = cast_samples_MatrixXcd(rho, rho_size, _atom->get_size());
    else _rho = cast_samples_VectorXcd_as_MatrixXcd(rho, rho_size, _atom->get_size());

    std::vector<Vector3d> _k_vec = cast_samples_Vector3d(k_vec, k_vec_size);
    std::vector<Vector3cd> _x_vec(k_vec_size, Vector3cd::Zero());
    for (size_t index_qk = 0; index_qk < k_vec_size; ++index_qk)
    {
        double _theta = rotation_theta(_k_vec.at(index_qk));
        double _phi = rotation_phi(_k_vec.at(index_qk));
        if (x_vec == 0) _x_vec.at(index_qk).real() = cast_theta_phi_et(_theta, _phi);
        else if (x_vec == 1) _x_vec.at(index_qk).real() = cast_theta_phi_ep(_theta, _phi);
        else
        {
            Vector3d t = cast_theta_phi_et(_theta, _phi);
            Vector3d p = cast_theta_phi_ep(_theta, _phi);
            if (x_vec == 2) _x_vec.at(index_qk) = t + sc::i * p;
            else if (x_vec == 3) _x_vec.at(index_qk) = t - sc::i * p;
            else return -2;
        }
    }

    std::vector<size_t> _i = cast_samples_size_t(i, i_size);
    std::vector<size_t> _f = cast_samples_size_t(f, f_size);
    _atom->scattering_rate(results, _k, _rho, _k_vec, _x_vec, _i, _f);

    return 0;
}

int atom_scattering_rate_qk_tp(
    void* atom, double* results, size_t* k, size_t k_size,
    std::complex<double>* rho, size_t rho_size, bool as_density_matrix,
    double* theta, double* phi, std::complex<double>* x_vec, size_t k_vec_size,
    size_t* i, size_t i_size, size_t* f, size_t f_size)
{
    Atom* _atom = static_cast<Atom*>(atom);

    std::vector<size_t> _k = cast_samples_size_t(k, k_size);
    for (size_t _ki : _k) if (_ki < 1 || _ki > _atom->get_decay_map()->get_k_em_max()) return -1;

    std::vector<MatrixXcd> _rho;
    if (as_density_matrix) _rho = cast_samples_MatrixXcd(rho, rho_size, _atom->get_size());
    else _rho = cast_samples_VectorXcd_as_MatrixXcd(rho, rho_size, _atom->get_size());
    
    std::vector<Vector3d> _k_vec = cast_samples_theta_phi_er(theta, phi, k_vec_size);
    std::vector<Vector3cd> _x_vec = cast_samples_Vector3cd(x_vec, k_vec_size);
    std::vector<size_t> _i = cast_samples_size_t(i, i_size);
    std::vector<size_t> _f = cast_samples_size_t(f, f_size);
    _atom->scattering_rate(results, _k, _rho, _k_vec, _x_vec, _i, _f);

    return 0;
}

int atom_scattering_rate_qk_tp_xb(
    void* atom, double* results, size_t* k, size_t k_size,
    std::complex<double>* rho, size_t rho_size, bool as_density_matrix,
    double* theta, double* phi, size_t x_vec, size_t k_vec_size,
    size_t* i, size_t i_size, size_t* f, size_t f_size)
{
    Atom* _atom = static_cast<Atom*>(atom);

    std::vector<size_t> _k = cast_samples_size_t(k, k_size);
    for (size_t _ki : _k) if (_ki < 1 || _ki > _atom->get_decay_map()->get_k_em_max()) return -1;

    std::vector<MatrixXcd> _rho;
    if (as_density_matrix) _rho = cast_samples_MatrixXcd(rho, rho_size, _atom->get_size());
    else _rho = cast_samples_VectorXcd_as_MatrixXcd(rho, rho_size, _atom->get_size());

    std::vector<Vector3d> _k_vec = cast_samples_theta_phi_er(theta, phi, k_vec_size);
    std::vector<Vector3cd> _x_vec(k_vec_size, Vector3cd::Zero());
    for (size_t index_qk = 0; index_qk < k_vec_size; ++index_qk)
    {
        double _theta = theta[index_qk];
        double _phi = phi[index_qk];
        if (x_vec == 0) _x_vec.at(index_qk).real() = cast_theta_phi_et(_theta, _phi);
        else if (x_vec == 1) _x_vec.at(index_qk).real() = cast_theta_phi_ep(_theta, _phi);
        else
        {
            Vector3d t = cast_theta_phi_et(_theta, _phi);
            Vector3d p = cast_theta_phi_ep(_theta, _phi);
            if (x_vec == 2) _x_vec.at(index_qk) = t + sc::i * p;
            else if (x_vec == 3) _x_vec.at(index_qk) = t - sc::i * p;
            else return -2;
        }
    }


    std::vector<size_t> _i = cast_samples_size_t(i, i_size);
    std::vector<size_t> _f = cast_samples_size_t(f, f_size);
    _atom->scattering_rate(results, _k, _rho, _k_vec, _x_vec, _i, _f);

    return 0;
}


// Interaction
void* interaction_construct()
{
    return static_cast<void*>(new Interaction());
}

void interaction_destruct(void* interaction)
{
    delete static_cast<Interaction*>(interaction);
}

void interaction_resonance_info(void* interaction)
{
    Interaction* _interaction = static_cast<Interaction*>(interaction);
    return _interaction->resonance_info();
}

int interaction_update(void* interaction)
{
    Interaction* _interaction = static_cast<Interaction*>(interaction);
    return _interaction->update();
}

void* interaction_get_environment(void* interaction)
{
    Interaction* _interaction = static_cast<Interaction*>(interaction);
    return static_cast<void*>(_interaction->get_env());
}

void interaction_set_environment(void* interaction, void* env)
{
    Interaction* _interaction = static_cast<Interaction*>(interaction);
    Environment* _env = static_cast<Environment*>(env);

    _interaction->set_env(_env);
}

void* interaction_get_atom(void* interaction)
{
    Interaction* _interaction = static_cast<Interaction*>(interaction);
    return static_cast<void*>(_interaction->get_atom());
}

void interaction_set_atom(void* interaction, void* atom)
{
    Interaction* _interaction = static_cast<Interaction*>(interaction);
    Atom* _atom = static_cast<Atom*>(atom);

    _interaction->set_atom(_atom);
}

void interaction_add_laser(void* interaction, void* laser)
{
    Interaction* _interaction = static_cast<Interaction*>(interaction);
    Laser* _laser = static_cast<Laser*>(laser);

    _interaction->add_laser(_laser);
}

void interaction_clear_lasers(void* interaction)
{
    Interaction* _interaction = static_cast<Interaction*>(interaction);
    _interaction->clear_lasers();

}

size_t interaction_get_lasers_size(void* interaction)
{
    Interaction* _interaction = static_cast<Interaction*>(interaction);
    return _interaction->get_lasers()->size();
}

void* interaction_get_laser(void* interaction, size_t m)
{
    Interaction* _interaction = static_cast<Interaction*>(interaction);
    return static_cast<void*>(_interaction->get_lasers()->at(m));
}

double interaction_get_delta_max(void* interaction)
{
    Interaction* _interaction = static_cast<Interaction*>(interaction);
    return _interaction->get_delta_max();
}

void interaction_set_delta_max(void* interaction, double delta_max)
{
    Interaction* _interaction = static_cast<Interaction*>(interaction);
    _interaction->set_delta_max(delta_max);
}

bool interaction_get_controlled(void* interaction)
{
    Interaction* _interaction = static_cast<Interaction*>(interaction);
    return _interaction->get_controlled();
}

void interaction_set_controlled(void* interaction, bool controlled)
{
    Interaction* _interaction = static_cast<Interaction*>(interaction);
    _interaction->set_controlled(controlled);
}

bool interaction_get_dense(void* interaction)
{
    Interaction* _interaction = static_cast<Interaction*>(interaction);
    return _interaction->get_dense();
}

void interaction_set_dense(void* interaction, bool dense)
{
    Interaction* _interaction = static_cast<Interaction*>(interaction);
    _interaction->set_dense(dense);
}

double interaction_get_dt(void* interaction)
{
    Interaction* _interaction = static_cast<Interaction*>(interaction);
    return _interaction->get_dt();
}

void interaction_set_dt(void* interaction, double dt)
{
    Interaction* _interaction = static_cast<Interaction*>(interaction);
    _interaction->set_dt(dt);
}

double interaction_get_dt_max(void* interaction)
{
    Interaction* _interaction = static_cast<Interaction*>(interaction);
    return _interaction->get_dt_max();
}

void interaction_set_dt_max(void* interaction, double dt_max)
{
    Interaction* _interaction = static_cast<Interaction*>(interaction);
    _interaction->set_dt_max(dt_max);
}

double interaction_get_atol(void* interaction)
{
    Interaction* _interaction = static_cast<Interaction*>(interaction);
    return _interaction->get_atol();
}

void interaction_set_atol(void* interaction, double atol)
{
    Interaction* _interaction = static_cast<Interaction*>(interaction);
    _interaction->set_atol(atol);
}

double interaction_get_rtol(void* interaction)
{
    Interaction* _interaction = static_cast<Interaction*>(interaction);
    return _interaction->get_rtol();
}

void interaction_set_rtol(void* interaction, double rtol)
{
    Interaction* _interaction = static_cast<Interaction*>(interaction);
    _interaction->set_rtol(rtol);
}

int* interaction_get_summap(void* interaction)
{
    Interaction* _interaction = static_cast<Interaction*>(interaction);
    return _interaction->get_summap()->data();
}

std::complex<double>* interaction_get_rabi(void* interaction, size_t m)
{
    Interaction* _interaction = static_cast<Interaction*>(interaction);

    MatrixXcd* rabi = new MatrixXcd(_interaction->get_atom()->get_size(), _interaction->get_atom()->get_size());
    *rabi = _interaction->get_rabimap()->at(m) * 2;
    return rabi->data();
}

double* interaction_get_atommap(void* interaction)
{
    Interaction* _interaction = static_cast<Interaction*>(interaction);
    return _interaction->get_atommap()->data();
}

double* interaction_get_deltamap(void* interaction)
{
    Interaction* _interaction = static_cast<Interaction*>(interaction);
    return _interaction->get_deltamap()->data();
}

double* interaction_get_delta(void* interaction)
{
    Interaction* _interaction = static_cast<Interaction*>(interaction);

    VectorXd* delta = new VectorXd(_interaction->get_atom()->get_size());
    *delta = _interaction->get_delta(*_interaction->get_atom()->get_w0(), _interaction->gen_w());
    return delta->data();
}

bool interaction_get_loop(void* interaction)
{
    Interaction* _interaction = static_cast<Interaction*>(interaction);
    return _interaction->get_loop();
}

bool interaction_get_time_dependent(void* interaction)
{
    Interaction* _interaction = static_cast<Interaction*>(interaction);
    return _interaction->get_time_dependent();
}

void interaction_set_time_dependent(void* interaction, bool time_dependent)
{
    Interaction* _interaction = static_cast<Interaction*>(interaction);
    _interaction->set_time_dependent(time_dependent);
}

size_t* interaction_get_history(void* interaction)
{
    Interaction* _interaction = static_cast<Interaction*>(interaction);
    return _interaction->history.data();
}

int interaction_get_n_history(void* interaction)
{
    Interaction* _interaction = static_cast<Interaction*>(interaction);
    return _interaction->n_history;
}

void interaction_get_hamiltonian(void* interaction, double* t, double* delta, double* v, std::complex<double>* h, size_t t_size, size_t sample_size)
{
    Interaction* _interaction = static_cast<Interaction*>(interaction);

    const std::vector<double> _t = cast_samples_double(t, t_size);
    const std::vector<VectorXd> _delta = cast_samples_VectorXd(delta, sample_size, _interaction->get_lasers()->size());
    const std::vector<Vector3d> _v = cast_samples_Vector3d(v, sample_size);
    size_t size = _interaction->get_atom()->get_size();
    for (size_t i = 0; i < sample_size; ++i)
    {
        for (size_t k = 0; k < t_size; ++k)
        {
            MatrixXcd _h = _interaction->get_hamiltonian(_t.at(k), _delta.at(i), _v.at(i));
            for (size_t m = 0; m < size; ++m)
            {
                for (size_t n = 0; n < size; ++n)
                {
                    h[i * size * size * t_size + m * size * t_size + n * t_size + k] = _h(n, m);
                }
            }
        }
    }
}

void interaction_rates(
    void* interaction, double* t, double* delta, double* v, double* x0, double* results, size_t t_size, size_t sample_size, bool analytic)
{
    Interaction* _interaction = static_cast<Interaction*>(interaction);

    const std::vector<double> _t = cast_samples_double(t, t_size);
    const std::vector<VectorXd> _delta = cast_samples_VectorXd(delta, sample_size, _interaction->get_lasers()->size());
    const std::vector<Vector3d> _v = cast_samples_Vector3d(v, sample_size);
    size_t size = _interaction->get_atom()->get_size();
    std::vector<VectorXd> _x0 = cast_samples_VectorXd(x0, sample_size, size);
    std::vector<std::vector<VectorXd>> _results = _interaction->rates(_t, _delta, _v, _x0, analytic);
    for (size_t i = 0; i < sample_size; ++i)
    {
        for (size_t j = 0; j < size; ++j)
        {
            for (size_t k = 0; k < t_size; ++k)
            results[i * size * t_size + j * t_size + k] = _results.at(i).at(k)(j);
        }
    }
}

void interaction_schroedinger(
    void* interaction, double* t, double* delta, double* v, std::complex<double>* x0, std::complex<double>* results, size_t t_size, size_t sample_size)
{
    Interaction* _interaction = static_cast<Interaction*>(interaction);

    const std::vector<double> _t = cast_samples_double(t, t_size);
    const std::vector<VectorXd> _delta = cast_samples_VectorXd(delta, sample_size, _interaction->get_lasers()->size());
    const std::vector<Vector3d> _v = cast_samples_Vector3d(v, sample_size);
    size_t size = _interaction->get_atom()->get_size();
    std::vector<VectorXcd> _x0 = cast_samples_VectorXcd(x0, sample_size, size);
    std::vector<std::vector<VectorXcd>> _results = _interaction->schroedinger(_t, _delta, _v, _x0);
    for (size_t i = 0; i < sample_size; ++i)
    {
        for (size_t j = 0; j < size; ++j)
        {
            for (size_t k = 0; k < t_size; ++k)
                results[i * size * t_size + j * t_size + k] = _results.at(i).at(k)(j);
        }
    }
}

void interaction_master(
    void* interaction, double* t, double* delta, double* v, std::complex<double>* x0, std::complex<double>* results, size_t t_size, size_t sample_size)
{
    Interaction* _interaction = static_cast<Interaction*>(interaction);

    const std::vector<double> _t = cast_samples_double(t, t_size);
    const std::vector<VectorXd> _delta = cast_samples_VectorXd(delta, sample_size, _interaction->get_lasers()->size());
    const std::vector<Vector3d> _v = cast_samples_Vector3d(v, sample_size);
    size_t size = _interaction->get_atom()->get_size();
    std::vector<MatrixXcd> _x0 = cast_samples_MatrixXcd(x0, sample_size, size);
    std::vector<std::vector<MatrixXcd>> _results = _interaction->master(_t, _delta, _v, _x0);
    for (size_t i = 0; i < sample_size; ++i)
    {
        for (size_t m = 0; m < size; ++m)
        {
            for (size_t n = 0; n < size; ++n)
            {
                for (size_t k = 0; k < t_size; ++k)
                    results[i * size * size * t_size + m * size * t_size + n * t_size + k] = _results.at(i).at(k)(n, m);
            }
        }
    }
}

void interaction_mc_master(
    void* interaction, double* t, double* delta, double* v, std::complex<double>* x0, bool dynamics, std::complex<double>* results, size_t t_size, size_t sample_size)
{
    Interaction* _interaction = static_cast<Interaction*>(interaction);

    const std::vector<double> _t = cast_samples_double(t, t_size);
    const std::vector<VectorXd> _delta = cast_samples_VectorXd(delta, sample_size, _interaction->get_lasers()->size());
    std::vector<Vector3d> _v = cast_samples_Vector3d(v, sample_size);
    size_t size = _interaction->get_atom()->get_size();
    std::vector<VectorXcd> _x0 = cast_samples_VectorXcd(x0, sample_size, size);
    std::vector<std::vector<VectorXcd>> _results = _interaction->mc_master(_t, _delta, _v, _x0, dynamics);
    for (size_t i = 0; i < sample_size; ++i)
    {   
        if (dynamics)
        {
            for (size_t j = 0; j < 3; ++j) v[i * 3 + j] = _v.at(i)(j);
        }
        for (size_t n = 0; n < size; ++n)
        {
            for (size_t k = 0; k < t_size; ++k)
                results[i * size * t_size + n * t_size + k] = _results.at(i).at(k)(n);
        }
    
    }
}

// @ScatteringRate
void sr_generate_y(std::complex<double>* denominator, std::complex<double>* f_theta,
    std::complex<double>* f_phi, size_t* counts, size_t* shape, double* y)
{
    size_t s0 = shape[0];
    size_t s1 = shape[1];
    size_t len_counts = shape[2];
    size_t len_y = s0 * s1;
    size_t sum_counts = 0;
    for (int i = 0; i < len_counts; ++i) {
        sum_counts += counts[i];
    }
    size_t i = 0;
    size_t ij = 0;
    for (size_t x = 0; x < s0; ++x) {
        for (size_t a = 0; a < s1; ++a) {
            i = 0;
            for (size_t c = 0; c < len_counts; ++c) {
                std::complex<double> c_theta(0., 0.);
                std::complex<double> c_phi(0., 0.);
                for (size_t j = 0; j < counts[c]; ++j) {
                    ij = i + j;
                    c_theta += denominator[x * sum_counts + ij] * f_theta[a * sum_counts + ij];
                    c_phi += denominator[x * sum_counts + ij] * f_phi[a * sum_counts + ij];
                }
                i = ij + 1;
                //y[x * s1 + a] += (c_theta * std::conj(c_theta) + c_phi * std::conj(c_phi)).real();
                y[x * s1 + a] += std::norm(c_theta) + std::norm(c_phi);
            }
        }
    }
}


void* multivariatenormal_construct(double* mean, double* cov, size_t size)
{
    return static_cast<void*>(new MultivariateNormal(cast_VectorXd(mean, size), cast_MatrixXd(cov, size)));
}

void multivariatenormal_destruct(void* mvn)
{
    delete static_cast<MultivariateNormal*>(mvn);
}

size_t multivariatenormal_size(void* mvn)
{
    MultivariateNormal* _mvn = static_cast<MultivariateNormal*>(mvn);
    return _mvn->get_size();
}

void multivariatenormal_rvs(void* mvn, double* ret)
{
    MultivariateNormal* _mvn = static_cast<MultivariateNormal*>(mvn);

    VectorXd _ret = _mvn->rvs();
    for (size_t i = 0; i < _ret.size(); ++i) ret[i] = _ret(i);
}

void gen_collinear(double* x, double* mean, double* cov, size_t* n, size_t size, size_t dim,
    size_t* n_max, bool user_seed, size_t seed, bool report)
{
    std::vector<VectorXd> _mean = cast_samples_VectorXd(mean, size, dim);
    std::vector<MatrixXd> _cov = cast_samples_MatrixXd(cov, size, dim);
    unsigned int _seed = std::random_device{}();
    if (user_seed) _seed = seed;
    CollinearResult res = collinear(_mean, _cov, *n, *n_max, _seed, report);
    std::vector<std::vector<VectorXd>>& p = res.get_p();
    for (size_t i = 0; i < *n; ++i)
    {
        for (size_t j = 0; j < size; ++j)
        {
            for (size_t k = 0; k < dim; ++k)
                x[i * size * dim + j * dim + k] = p.at(i).at(j)(k);
        }
    }
    *n = res.get_n_accepted();
    *n_max = res.get_n_samples();
}
