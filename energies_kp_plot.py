import numpy as np
import pandas as pd
import math as mt
import matplotlib.pyplot as plt

data = pd.read_excel(r'.\energies_kp.xlsx', sheet_name = "probability")

n_qubits = np.array(data['n_qubits'])
e0 = np.array(data['e0'])/5000
e1 = np.array(data['e1'])/5000
e2 = np.array(data['e2'])/5000
e3 = np.array(data['e3'])/5000

x_model = np.linspace(min(n_qubits), max(n_qubits), len(n_qubits))

# drawing

plt.figure(1, dpi=120)

e_0, = plt.plot(n_qubits, e0, label="ground state")
e_1, = plt.plot(n_qubits, e1, label="first excited state")
e_2, = plt.plot(n_qubits, e2, label="second excited state")
e_3, = plt.plot(n_qubits, e3, label="third excited state")

plt.xlabel("Number of qubits used")
plt.ylabel("Probability")
plt.legend(handles=[e_0, e_1, e_2, e_3])

plt.savefig('energies_kp.png')
plt.savefig('energies_kp.pdf')

plt.show()