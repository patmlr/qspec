
import numpy as np
import qspec as qs
import qspec.models as mod
import matplotlib.pyplot as plt

I, J0, J1 = 9/2, 1/2, 3/2  # Angular momentum quantum numbers

x, y = np.load('data/87Sr+_partial.npy')
sigma_y = np.sqrt(y)  # uncertainty of the photon counts
x_cont = np.linspace(x[0], x[-1], 1001)  # Generate a dense x array for plotting

model_qi = mod.HyperfineQI(None, I, J0, J1, label='87Sr+')
# generate a quantum interference model.
# Note that the first argument has no effect
# and is just there for compatibility reasons.
# The 'name' keyword is optional for unique console outputs.

model = mod.Offset(mod.NPeak(model_qi, n_peaks=1))
# Add a single x-axis shift and amplitude parameter (NPeak)
# as well as a y-axis shift parameter (Offset)

# model = mod.gen_model((I, J0, J1), qi=True, n_peaks=None)
# generate a quantum interference hyperfine structure model.
print(f'\nParameter names: {model.names}\n')

model.set_vals([20., -1000.5, -36., 86., 0., 0., 0.1, -20., 2000., 0.])
model.set_fixes([False, True, False, '86(2.0)', True, 'Au / Al', False, False, False, False])

popt, pcov, info = mod.fit(
    model, x, y, sigma_y=sigma_y, report=True, guess_offset=True)

res = y - model(x, *popt)
y_qi = model(x_cont, *popt)
sigma_y_qi = qs.propagate_fit(model, x_cont, popt, pcov, sample_size=10000)

popt[6] = 0
y_0 = model(x_cont, *popt)

fig, (m, r) = plt.subplots(2, 1, sharex='all', height_ratios=[3, 1], figsize=(6, 5))

m.errorbar(x, y, yerr=sigma_y, fmt='.k', label='Data')
m.plot(x_cont, y_qi, '-C1', label='QI fit', lw=2)
m.plot(x_cont, y_0, '--C0', label=r'$geo = 0$', lw=2)
m.fill_between(x_cont, sigma_y_qi[1], sigma_y_qi[2], color='C1', alpha=0.3)

r.errorbar(x, res, yerr=sigma_y, fmt='.k', label='Residuals', zorder=-100)
r.plot(x_cont, y_0 -  y_qi, '-C0', lw=2)

x_lim = x_cont[0], x_cont[-1]

r.hlines(0, *x_lim, colors='C1', lw=2)

m.set_xlim(x_lim)
m.set_ylabel('Photon events (counts)')
m.legend()

r.set_xlabel('Relative frequency (GHz)')
r.set_ylabel('Residuals (counts)')

plt.subplots_adjust(left=0.13, bottom=0.11, top=0.99, right=0.98, hspace=0.04)
plt.show()
