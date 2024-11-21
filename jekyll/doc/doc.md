---
layout: article
title: API Doc
---

API Doc
=======

## qspec's module structure

Currently, there are seven modules in qspec, divided by their purpose.

- [_algebra_]() &emsp; Angular momentum algebra and transition matrix elements.
- [_analyze_]() &emsp; Linear and nonlinear optimization routines.
- [_models_]() &emsp; System of modular fit models.
- [_physics_]() &emsp; Physical functions.
- [_simulate_]() &emsp; Simulation of laser-atom interactions
- [_stats_]() &emsp; Statistical functions.
- [_tools_](/doc/functions/tools/print_colored.html) &emsp; Mathematical and data management functions.

## qspec's namespaces

For the ease of access of most of the functions implemented in qspec,
the seven modules are summarized in only three namespaces,

- [_qspec_]()
- [_qspec.models_]()
- [_qspec.simulate_]()

which allows most of the functions to be accessed by simply importing qspec.
