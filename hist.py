import numpy as np
import pandas as pd
import math as mt
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

#data = pd.read_excel(r'.\hist_kp.xlsx')

data = pd.read_excel(r'.\complete_graph_10.xlsx', sheet_name = "21 qubits")

energy = np.array(data['energy'])
deg = np.array(data['deg'])
num_occ = np.array(data['num_occ.'])

energy = energy * 6.62607015e-25 * 7.56622700

# Boltzmann distribution

k_B = 1.380649E-23

def Z(x, T):
    E = 0
    for i in range(0, len(x[0])-1):
        x1 = np.float64(-(x[0][i])/(k_B * T))
        E += x[1][i] * (mt.e ** x1)
    return E 
    
def boltzmann_distribution(x,T):
    y = np.float64(-(x[0])/(k_B * T))
    return x[1]*(mt.e ** y) / Z(x, T)

x_model = np.linspace(min(energy), max(energy), 121)
y_data = num_occ/2000

popt, pcov = curve_fit(boltzmann_distribution, [energy, deg], y_data, p0 = [10])

print(popt)
print(pcov)

# drawing

plt.figure(1, dpi=120)
plt.plot(x_model, boltzmann_distribution([x_model, deg], 10), color = "red")
plt.scatter(energy, num_occ/2000, label="Energy Distribution")
plt.xlabel("Energy (J)")
plt.ylabel("Occurrences")

plt.show()