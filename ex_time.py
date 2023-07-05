import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def chi_squared_reduced(data, model, sigma, dof=None):
    """
    Calculate the reduced chi-squared value for a fit.

    If no dof is given, returns the chi-squared (non-reduced) value.

    Parameters
    ----------
    data : array_like
        The observed data.
    model : array_like
        The model data.
    sigma : array_like
        The uncertainty in the data.
    dof : int
        Degrees of freedom (len(data) - # of free parameters).
    """
    sq_residual = (data - model)**2
    chi_sq = np.sum(sq_residual / sigma**2)
    if dof is None:
        return chi_sq
    else:
        nu = len(data) - dof
        return chi_sq / nu

data = pd.read_excel(r'.\energies_kp.xlsx', sheet_name = "time_kp")

n_qubits = np.array(data['n_qubits'])
time_qa = np.array(data['mean_qa'])
time_err = np.array(data['err_qa'])
#time_c = np.array(data['mean_c'])

def f(x, a, b):
    return a * x + b

x_model = np.linspace(min(n_qubits), max(n_qubits), 11)

popt, pcov = curve_fit(f, n_qubits, time_qa, p0 = [0.5, 0.5])

print(popt)
print(pcov)

# Define our data, model, uncertainty, and degrees of freedom
I_data = time_qa.copy()  # observed data
I_modeled = f(x_model, *popt) # model fitted result
I_sigma = time_err.copy() # uncertainty in the data

# Calculate the Chi-Squared value (no dof)
chisq = chi_squared_reduced(I_data, I_modeled, I_sigma)
print(f"chi-squared statistic = {chisq:1.2f}")

# Calculate the Reduced Chi-Squared value (with dof)
dof = 9  # We have two free parameters
rechisq = chi_squared_reduced(I_data, I_modeled, I_sigma, dof)
print(f"reduced chi-squared = {rechisq:1.2f}")

fig = plt.figure(1, dpi=120)
ax = fig.add_subplot(1,1,1)

tqa = ax.scatter(n_qubits, time_qa, label="Quantum annealer compilation time")
fit, = plt.plot(n_qubits, f(n_qubits, popt[0], popt[1]), label="Linear fit", color = 'r')
plt.errorbar(n_qubits, time_qa, yerr=time_err, fmt="o")
#tc, = plt.plot(n_qubits, time_c, label="Classical computer compilation time")
#ax.set_yscale('log')
plt.xlabel("Number of qubits")
plt.ylabel("Time (s)")

plt.savefig('compilation_time_qa.pdf')
plt.savefig('compilation_time_qa.png')

plt.legend(handles = [tqa, fit], loc = 'upper left')

plt.show()