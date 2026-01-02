#pragma once

#include "Utility.h"
#include "Light.h"
#include "Matter.h"
#include "Interaction.h"
#include "King.h"

#include <complex>


#ifdef __cplusplus
extern "C" {
#endif

    #if defined(_WIN32) || defined(__CYGWIN__)
    #ifdef QSPEC_EXPORTS
        #define QSPEC_API __declspec(dllexport)
    #else
        #define QSPEC_API __declspec(dllimport)
    #endif
    #else
    #if __GNUC__ >= 4
        #define QSPEC_API __attribute__((visibility("default")))
    #else
        #define QSPEC_API
    #endif
    #endif


    // Polarization
    QSPEC_API void* polarization_construct();
    QSPEC_API void polarization_destruct(void* polarization);
    QSPEC_API void polarization_init(void* polarization, std::complex<double>* vec, double* q_axis, bool vec_as_q);
    QSPEC_API void polarization_def_q_axis(void* polarization, double* q_axis, bool q_fixed);
    QSPEC_API double* polarization_get_q_axis(void* polarization);
    QSPEC_API std::complex<double>* polarization_get_x(void* polarization);
    QSPEC_API std::complex<double>* polarization_get_q(void* polarization);

    // Laser
    QSPEC_API void* laser_construct();
    QSPEC_API void laser_destruct(void* laser);
    QSPEC_API void laser_init(void* laser, double freq, double intensity, void* polarization, double* k);
    QSPEC_API double laser_get_freq(void* laser);
    QSPEC_API void laser_set_freq(void* laser, double freq);
    QSPEC_API double laser_get_intensity(void* laser);
    QSPEC_API void laser_set_intensity(void* laser, double intensity);
    QSPEC_API void* laser_get_polarization(void* laser);
    QSPEC_API void laser_set_polarization(void* laser, void* polarization);
    QSPEC_API double* laser_get_k(void* laser);
    QSPEC_API void laser_set_k(void* laser, double* k);
    QSPEC_API std::complex<double>* laser_get_kpol(void* laser, bool electric, size_t k, double* q_axis);

    //Environment
    QSPEC_API void* environment_construct();
    QSPEC_API void environment_destruct(void* env);
    QSPEC_API double environment_get_E(void* env);
    QSPEC_API double environment_get_B(void* env);
    QSPEC_API double* environment_get_e_E(void* env);
    QSPEC_API double* environment_get_e_B(void* env);
    QSPEC_API void environment_set_E(void* env, double* E);
    QSPEC_API void environment_set_B(void* env, double* B);
    QSPEC_API void environment_set_E_double(void* env, double E);
    QSPEC_API void environment_set_B_double(void* env, double B);

    // State
    QSPEC_API void* state_construct();
    QSPEC_API void state_destruct(void* state);
    QSPEC_API void state_init(void* state, double freq_0, double* s, double* l, double j, double i, double f, double m,
        bool parity, double* jj, size_t ls_size, double* hyper_const, double gj, double gi, char* label);
    QSPEC_API void state_reset(void* state);
    QSPEC_API double state_get_shift(void* state);
    QSPEC_API double state_get_freq_j(void* state);
    QSPEC_API void state_set_freq_j(void* state, double freq_j);
    QSPEC_API double state_get_freq(void* state);
    QSPEC_API void state_set_freq(void* state, double freq);
    QSPEC_API double state_get_j(void* state);
    QSPEC_API void state_set_j(void* state, double j);
    QSPEC_API double state_get_i(void* state);
    QSPEC_API void state_set_i(void* state, double i);
    QSPEC_API double state_get_f(void* state);
    QSPEC_API void state_set_f(void* state, double f);
    QSPEC_API double state_get_m(void* state);
    QSPEC_API void state_set_m(void* state, double m);
    QSPEC_API double* state_get_hyper_const(void* state);
    QSPEC_API void state_set_hyper_const(void* state, double* hyper_const);
    QSPEC_API double state_get_gj(void* state);
    QSPEC_API void state_set_gj(void* state, double gj);
    QSPEC_API double state_get_gi(void* state);
    QSPEC_API void state_set_gi(void* state, double gi);
    QSPEC_API const char* state_get_label(void* state);
    QSPEC_API void state_set_label(void* state, char* label);

    // DecayMap
    QSPEC_API void* decaymap_construct();
    QSPEC_API void decaymap_destruct(void* decays);
    QSPEC_API void decaymap_add_decay(void* decays, char* state_0, char* state_1, double* ae, size_t ae_size,
        double* am, size_t am_size, bool single_leading_order);
    QSPEC_API const char* decaymap_get_label(void* decays, size_t i, size_t j);
    QSPEC_API size_t decaymap_get_size(void* decays);
    QSPEC_API size_t decaymap_get_k_em_max(void* decays);
    QSPEC_API void decaymap_set_k_em_max(void* decays, size_t k_em_max);
    QSPEC_API double decaymap_get_a_i(void* decays, char* state_0, char* state_1, bool parity_equal);
    QSPEC_API double decaymap_get_ae_ik(void* decays, char* state_0, char* state_1, size_t k);
    QSPEC_API double decaymap_get_am_ik(void* decays, char* state_0, char* state_1, size_t k);

    // Atom
    QSPEC_API void* atom_construct();
    QSPEC_API void atom_destruct(void* atom);
    QSPEC_API void atom_update(void* atom);
    QSPEC_API void atom_set_env(void* atom, void* env);
    QSPEC_API void atom_add_state(void* atom, void* state);
    QSPEC_API void atom_clear_states(void* atom);
    QSPEC_API void* atom_get_decay_map(void* atom);
    QSPEC_API void atom_set_decay_map(void* atom, void* decays);
    QSPEC_API double atom_get_gamma(void* atom, size_t i);
    QSPEC_API double atom_get_mass(void* atom);
    QSPEC_API void atom_set_mass(void* atom, double mass);
    QSPEC_API size_t atom_get_size(void* atom);
    QSPEC_API size_t atom_get_gs_size(void* atom);
    QSPEC_API size_t* atom_get_gs(void* atom);
    QSPEC_API int* atom_get_ek(void* atom, size_t k);
    QSPEC_API int* atom_get_mk(void* atom, size_t k);
    QSPEC_API int* atom_get_emk(void* atom, size_t k);
    QSPEC_API double* atom_get_d_em(void* atom, size_t k);
    QSPEC_API double* atom_get_L0(void* atom);
    QSPEC_API double* atom_get_L1(void* atom);
    QSPEC_API int atom_scattering_rate_4pi(void* atom, double* results, size_t* k, size_t k_size,
        std::complex<double>* rho, size_t rho_size, bool as_density_matrix,
        size_t* i, size_t i_size, size_t* f, size_t f_size);
    QSPEC_API int atom_scattering_rate_k(
        void* atom, double* results, size_t* k, size_t k_size,
        std::complex<double>* rho, size_t rho_size, bool as_density_matrix,
        double* k_vec, size_t k_vec_size,
        size_t* i, size_t i_size, size_t* f, size_t f_size);
    QSPEC_API int atom_scattering_rate_k_tp(
        void* atom, double* results, size_t* k, size_t k_size,
        std::complex<double>* rho, size_t rho_size, bool as_density_matrix,
        double* theta, double* phi, size_t k_vec_size,
        size_t* i, size_t i_size, size_t* f, size_t f_size);
    QSPEC_API int atom_scattering_rate_qk(
        void* atom, double* results, size_t* k, size_t k_size,
        std::complex<double>* rho, size_t rho_size, bool as_density_matrix,
        double* k_vec, std::complex<double>* x_vec, size_t k_vec_size,
        size_t* i, size_t i_size, size_t* f, size_t f_size);
    QSPEC_API int atom_scattering_rate_qk_xb(
        void* atom, double* results, size_t* k, size_t k_size,
        std::complex<double>* rho, size_t rho_size, bool as_density_matrix,
        double* k_vec, size_t x_vec, size_t k_vec_size,
        size_t* i, size_t i_size, size_t* f, size_t f_size);
    QSPEC_API int atom_scattering_rate_qk_tp(
        void* atom, double* results, size_t* k, size_t k_size,
        std::complex<double>* rho, size_t rho_size, bool as_density_matrix,
        double* theta, double* phi, std::complex<double>* x_vec, size_t k_vec_size,
        size_t* i, size_t i_size, size_t* f, size_t f_size);
    QSPEC_API int atom_scattering_rate_qk_tp_xb(
        void* atom, double* results, size_t* k, size_t k_size,
        std::complex<double>* rho, size_t rho_size, bool as_density_matrix,
        double* theta, double* phi, size_t x_vec, size_t k_vec_size,
        size_t* i, size_t i_size, size_t* f, size_t f_size);

    // Interaction
    QSPEC_API void* interaction_construct();
    QSPEC_API void interaction_destruct(void* interaction);
    QSPEC_API void interaction_resonance_info(void* interaction);
    QSPEC_API int interaction_update(void* interaction);
    QSPEC_API void* interaction_get_environment(void* interaction);
    QSPEC_API void interaction_set_environment(void* interaction, void* environment);
    QSPEC_API void* interaction_get_atom(void* interaction);
    QSPEC_API void interaction_set_atom(void* interaction, void* atom);
    QSPEC_API void interaction_add_laser(void* interaction, void* laser);
    QSPEC_API void interaction_clear_lasers(void* interaction);
    QSPEC_API size_t interaction_get_lasers_size(void* interaction);
    QSPEC_API void* interaction_get_laser(void* interaction, size_t m);
    QSPEC_API double interaction_get_delta_max(void* interaction);
    QSPEC_API void interaction_set_delta_max(void* interaction, double delta_max);
    QSPEC_API bool interaction_get_controlled(void* interaction);
    QSPEC_API void interaction_set_controlled(void* interaction, bool controlled);
    QSPEC_API bool interaction_get_dense(void* interaction);
    QSPEC_API void interaction_set_dense(void* interaction, bool dense);
    QSPEC_API double interaction_get_dt(void* interaction);
    QSPEC_API void interaction_set_dt(void* interaction, double dt);
    QSPEC_API double interaction_get_dt_max(void* interaction);
    QSPEC_API void interaction_set_dt_max(void* interaction, double dt_max);
    QSPEC_API double interaction_get_atol(void* interaction);
    QSPEC_API void interaction_set_atol(void* interaction, double atol);
    QSPEC_API double interaction_get_rtol(void* interaction);
    QSPEC_API void interaction_set_rtol(void* interaction, double rtol);
    QSPEC_API int* interaction_get_summap(void* interaction);
    QSPEC_API std::complex<double>* interaction_get_rabi(void* interaction, size_t m);
    QSPEC_API double* interaction_get_atommap(void* interaction);
    QSPEC_API double* interaction_get_deltamap(void* interaction);
    QSPEC_API double* interaction_get_delta(void* interaction);
    QSPEC_API bool interaction_get_loop(void* interaction);
    QSPEC_API bool interaction_get_time_dependent(void* interaction);
    QSPEC_API void interaction_set_time_dependent(void* interaction, bool time_dependent);
    QSPEC_API size_t* interaction_get_history(void* interaction);
    QSPEC_API int interaction_get_n_history(void* interaction);
    QSPEC_API void interaction_get_hamiltonian(void* interaction, double* t, double* delta, double* v,
        std::complex<double>* h, size_t t_size, size_t sample_size);
    QSPEC_API void interaction_rates(void* interaction, double* t, double* delta, double* v,
        double* x0, double* results, size_t t_size, size_t sample_size, bool analytic);
    QSPEC_API void interaction_schroedinger(void* interaction, double* t, double* delta, double* v,
        std::complex<double>* x0, std::complex<double>* results, size_t t_size, size_t sample_size);
    QSPEC_API void interaction_master(void* interaction, double* t, double* delta, double* v,
        std::complex<double>* x0, std::complex<double>* results, size_t t_size, size_t sample_size);
    QSPEC_API void interaction_mc_master(void* interaction, double* t, double* delta, double* v,
        std::complex<double>* x0, bool dynamics, std::complex<double>* results, size_t t_size, size_t sample_size);

    // @ScatteringRate
    QSPEC_API void sr_generate_y(std::complex<double>* denominator, std::complex<double>* f_theta,
        std::complex<double>* f_phi, size_t* counts, size_t* shape, double* y);
    
    // MultivariateNormal
    QSPEC_API void* multivariatenormal_construct(double* mean, double* cov, size_t size);
    QSPEC_API void multivariatenormal_destruct(void* mvn);
    QSPEC_API size_t multivariatenormal_size(void* mvn);
    QSPEC_API void multivariatenormal_rvs(void* mvn, double* ret);
    QSPEC_API void gen_collinear(double* x, double* mean, double* cov, size_t* n, size_t size, size_t dim,
        size_t* n_max, bool user_seed, size_t seed, bool report);

#ifdef __cplusplus
};
#endif
