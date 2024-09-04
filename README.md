qspec
=====

![Static Badge](https://img.shields.io/badge/OS-Windows-yellow)
![Static Badge](https://img.shields.io/badge/Python_version-3.8%2B-blue?logo=python&logoColor=white)
[![Static Badge](https://img.shields.io/badge/License-MIT-orange)](https://opensource.org/license/mit)

[![Static Badge](https://img.shields.io/badge/DOI-10.48550/arXiv.2409.01417-blue)](https://www.doi.org/10.48550/arXiv.2409.01417)
[![Static Badge](https://img.shields.io/badge/arXiv-2409.01417-red)](https://arxiv.org/abs/2409.01417)

The [_qspec_](https://pypi.org/project/qspec/) Python package provides mathematical and physical functions
frequently used in laser spectroscopy but also more general methods for data processing. 
Most functions are compatible with numpy arrays and are able to process *n*-dimensional arrays.
This enables fast calculations with large samples of data, e.g., facilitating Monte-Carlo simulations.
The API documentation and example scripts are available on the [_GitHub_](https://github.com/patmlr/qspec) page.

### Dependencies

- [_Matplotlib_](http://matplotlib.org/)
- [_NumPy_](http://www.numpy.org/)
- [_SciPy_](http://www.scipy.org/)
- [_SymPy_](http://www.sympy.org/)

### Modules

- _algebra_: Contains functions to calculate dipole coefficients and Wigner-*j* symbols.
- _analyze_: Contains optimization functions and a class for King-plots.
- _models_: Framework to generate modular fit models.
- _physics_: Library of physical functions.
- _simulate_: Intuitive framework to simulate coherent laser-atom interactions.
- _stats_: Contains functions for the statistical analysis of data.
- _tools_: General helper, print, data shaping and mathematical functions.

### Use cases
- Coherently evolve atomic state population in a classical laser field. 
In contrast to powerful packages such as [_qutip_](https://qutip.org/),
the quantum mechanical system is set up automatically by just providing atomic state and laser information.
- Generate modular lineshape models for fitting. The modular system can be used
to sum, convolve, link models and share parameters, fit hyperfine structure spectra, etc. This module is similar to the [_satlas2_](https://iks-nm.github.io/satlas2/) Python package.
- Perform multidimensional King-plot analyses.
- Calculate frequently used physical observables such as kinetic energies, velocities, Doppler shifts, 
hyperfine structure splittings, etc.
