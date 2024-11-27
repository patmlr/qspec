---
layout: article
title: API Doc
---

API Doc
=======

## qspec's module structure

Currently, there are seven modules in qspec, divided by their purpose.

- [_algebra_]({{ site.baseurl }}{% link doc/modules/algebra.html %}) &emsp; Angular momentum algebra and transition matrix elements.
- [_analyze_]({{ site.baseurl }}{% link doc/modules/analyze.html %}) &emsp; Linear and nonlinear optimization routines.
- [_models_]({{ site.baseurl }}{% link doc/modules/models.html %}) &emsp; System of modular fit models.
- [_physics_]({{ site.baseurl }}{% link doc/modules/physics.html %}) &emsp; Physical functions.
- [_simulate_]({{ site.baseurl }}{% link doc/modules/simulate.html %}) &emsp; Simulation of laser-atom interactions
- [_stats_]({{ site.baseurl }}{% link doc/modules/stats.html %}) &emsp; Statistical functions.
- [_tools_]({{ site.baseurl }}{% link doc/modules/tools.html %}) &emsp; Mathematical and data management functions.

## qspec's namespaces

For the ease of access of most of the functions implemented in qspec,
the seven modules are summarized in only three namespaces,

- <code>qspec</code>
- <code>qspec.models</code>
- <code>qspec.simulate</code>

which allows most of the functions to be accessed by simply importing qspec.
