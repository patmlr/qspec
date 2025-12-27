"""
examples.tutorial_3
===================

Tutorial 3 from the website: Construction and fitting of modular lineshape models.
"""

import matplotlib.pyplot as plt
import numpy as np

import qspec.models as mod

# Generate the model used in this tutorial
model = mod.Offset(  # Add a y-axis shift (off0e0)

    mod.NPeak(  # Add a single (n_peaks=1)
        # x-axis shift (x0) and an intensity (p0)

        mod.Voigt(), n_peaks=1))  # Add a Voigt lineshape
# with Lorentzian (Gamma) and Gaussian (sigma) widths

print(f"\nParameter names: {model.names}\n")
# >>> Parameter names: ["Gamma", "sigma", "x0", "p0", "off0e0"]

# Change all parameter values and if they stay fixed during fitting at once
model.set_vals([20., 6., 0., 10., 3.])
model.set_fixes([False, "5(0.7)", False, False, False])

# Generate some random data for this tutorial
x = np.linspace(-80., 80., 81)  # x-values
sigma_y = np.full_like(x, 0.5)  # y-uncertainties
y = np.random.default_rng().normal(model(x, *model.vals), sigma_y)
# Random y-values around the model

# Change the initial value of the peak position and intensity
model.set_val(2, 15.)  # Parameter 2 (x0), the peak position
model.set_val(3, 5.)  # Parameter 3 (p0), the peak intensity
p_init = model.vals.copy()  # Copy the initial values for the plot

# Fit the constructed model to the data
popt, pcov, info = mod.fit(
    model, x, y, sigma_y=sigma_y, report=True)

# We plot the data, the initial model and the fitted model
plt.errorbar(x, y, yerr=sigma_y, fmt=".k", label="Data")
plt.plot(x, model(x, *p_init), "-C0", label="Initial model")
plt.plot(x, model(x, *popt), "-C1", label="Fitted model")

# Improve the plot
plt.legend()
plt.xlabel("x"), plt.ylabel("y")
plt.subplots_adjust(left=0.08, bottom=0.09, top=0.99, right=0.99)
plt.show()
