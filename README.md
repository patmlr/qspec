qspec
=====

A python package for calculations surrounding laser spectroscopy.

The [_qspec_](https://github.com/lasersphere/qspec) Python package provides mathematical and physical functions
frequently used in laser spectroscopy but also more general methods for data processing. 
Most functions are compatible with numpy arrays and are able to process n-dimensional arrays,
even in the case of arrays of vector- or matrix-objects. This enables fast calculations with large samples of data,
e.g., facilitating Monte-Carlo simulations.

Exemplary, two exciting use cases could be:
- Coherently evolve atomic state population in a classical laser field. 
In contrast to powerful packages such as [_qutip_](https://qutip.org/),
the quantum mechanical system is set up automatically by just providing atomic state and laser information.
- Generate modular lineshape models for fitting. The modular system can be used
to sum, convolve, link models and share parameters, fit hyperfine structure spectra, etc.


### Included modules

- _tools_: General helper, print, data shaping and mathematical functions.
- _stats_: Contains functions for the statistical analysis of data.
- _physics_: Library of physical functions.
- _algebra_: Contains functions to calculate dipole coefficients and Wigner-j symbols.
- _analyze_: Contains optimization functions and a class for King-plots.
- _lineshapes_: A framework to generate modular lineshape models for fitting.
- _simulate_: An intuitive framework to simulate laser-atom interactions.
